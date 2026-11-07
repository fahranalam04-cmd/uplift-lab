"""Heterogeneous treatment effects on Criteo: meta-learners with LightGBM base learners.

Splits come from the stratified ``bucket`` column (stratified by treatment x visit):
train = buckets 0-3 (a seeded 50% subsample is used for fitting, ~2.8M users),
validation = buckets 4-5 (all ~2.8M users) for choosing configs and the final model,
test = buckets 6-9 (all ~5.6M users), scored once with the chosen configurations.

Baselines: random targeting and a *response model* (P(visit | x), ignoring treatment),
which is what a team without uplift modelling would typically target by.
"""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import psutil

from abkit.uplift import (
    SLearner,
    TLearner,
    TransformedOutcome,
    XLearner,
    incremental_capture_at_k,
    poisson_bootstrap,
    qini_auc_score,
    qini_curve,
    uplift_at_k,
    uplift_auc_score,
    uplift_curve,
)
from abkit.variance_reduction import check_pretreatment

from . import config, data
from .data import SPLITS

BASE = {"learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.9, "n_jobs": config.THREADS, "verbose": -1,
        "random_state": config.SEED}
GRID = [
    {"num_leaves": 31, "min_child_samples": 1_000},
    {"num_leaves": 127, "min_child_samples": 3_000},
]
KS = (0.1, 0.3, 0.5)


def _clf(cfg: dict[str, Any]) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective="binary", **BASE, **cfg)


def _reg(cfg: dict[str, Any]) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="regression", **BASE, **cfg)


class ResponseModel:
    """Baseline: rank users by predicted outcome probability, ignoring treatment."""

    name = "response_model"

    def __init__(self, model: Any) -> None:
        self.model = model

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> ResponseModel:
        self.model.fit(X, y)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def make_learner(kind: str, cfg: dict[str, Any]) -> Any:
    if kind == "s_learner":
        return SLearner(_clf(cfg))
    if kind == "t_learner":
        return TLearner(_clf(cfg))
    if kind == "x_learner":
        return XLearner(_clf(cfg), _reg(cfg))
    if kind == "transformed_outcome":
        return TransformedOutcome(_reg(cfg))
    if kind == "response_model":
        return ResponseModel(_clf(cfg))
    raise ValueError(kind)


KINDS = ["s_learner", "t_learner", "x_learner", "transformed_outcome", "response_model"]


def _load(split: str, frac: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cols = [*config.FEATURES, "treatment", *config.OUTCOMES]
    check_pretreatment(config.FEATURES)
    d = data.load_buckets(SPLITS[split], cols, sample_frac=frac, seed=config.SEED + 7)
    X = np.column_stack([d.pop(f) for f in config.FEATURES])
    return X, d.pop("treatment").astype(np.int8), d


def _metrics(y: np.ndarray, t: np.ndarray, s: np.ndarray,
             w: np.ndarray | None = None) -> dict[str, float]:
    out = {"qini_auc": qini_auc_score(y, t, s, sample_weight=w),
           "uplift_auc": uplift_auc_score(y, t, s, sample_weight=w)}
    for k in KS:
        out[f"uplift_at_{int(k * 100)}"] = uplift_at_k(y, t, s, k, sample_weight=w)
        out[f"capture_at_{int(k * 100)}"] = incremental_capture_at_k(y, t, s, k,
                                                                     sample_weight=w)
    return out


def _ipw_weights(split: str, frac: float, t: np.ndarray, clip: float = 0.01) -> np.ndarray:
    """1/e for treated, 1/(1-e) for control, from the cross-fitted propensity e(x)."""
    e = data.load_buckets(SPLITS[split], ["e_hat"], sample_frac=frac, seed=config.SEED + 7,
                          path=config.CUPAC_PARQUET)["e_hat"].astype(float)
    if e.size != t.size:
        raise RuntimeError("propensity rows are not aligned with the test split")
    e = np.clip(e, clip, 1 - clip)
    return np.where(t == 1, 1 / e, 1 / (1 - e))


def _curve_points(y: np.ndarray, t: np.ndarray, s: np.ndarray, n_points: int = 201
                  ) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    grid = np.linspace(0, 1, n_points)
    for name, fn in (("uplift", uplift_curve), ("qini", qini_curve)):
        x, v = fn(y, t, s)
        out[name] = np.interp(grid * x[-1], x, v).tolist()
    out["fraction"] = grid.tolist()
    return out


def run(quick: bool = False, outcomes: tuple[str, ...] = ("visit", "conversion")) -> dict[str, Any]:
    t0 = time.time()
    train_frac = 0.05 if quick else 0.5
    X_tr, t_tr, Y_tr = _load("train", train_frac)
    X_va, t_va, Y_va = _load("val", 0.1 if quick else 1.0)
    result: dict[str, Any] = {
        "splits": {
            "train": {"buckets": SPLITS["train"], "sample_frac": train_frac,
                      "n": int(t_tr.size), "treatment_share": float(t_tr.mean())},
            "val": {"buckets": SPLITS["val"], "n": int(t_va.size),
                    "treatment_share": float(t_va.mean())},
        },
        "base_params": BASE, "grid": GRID, "hardware": "CPU only, 4 threads",
    }
    chosen: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        val_rows = []
        fitted: dict[str, tuple[float, Any, int]] = {}
        for kind in KINDS:
            for gi, cfg in enumerate(GRID):
                t1 = time.time()
                learner = make_learner(kind, cfg).fit(X_tr, t_tr, Y_tr[outcome])
                s = learner.predict_uplift(X_va)
                q = qini_auc_score(Y_va[outcome], t_va, s)
                val_rows.append({"learner": kind, "config": gi, "val_qini_auc": q,
                                 "val_capture_at_30": incremental_capture_at_k(
                                     Y_va[outcome], t_va, s, 0.3),
                                 "fit_predict_seconds": round(time.time() - t1, 1)})
                print(outcome, val_rows[-1], flush=True)
                if kind not in fitted or q > fitted[kind][0]:
                    fitted[kind] = (q, learner, gi)
        uplift_kinds = [k for k in KINDS if k != "response_model"]
        best_kind = max(uplift_kinds, key=lambda k: fitted[k][0])
        chosen[outcome] = {"learners": {k: fitted[k] for k in KINDS}, "best": best_kind}
        result[outcome] = {
            "validation": val_rows,
            "selected_config": {k: fitted[k][2] for k in KINDS},
            "chosen_model": best_kind,
            "selection_rule": "highest validation Qini AUC among uplift learners",
        }
    del X_tr, t_tr, Y_tr, X_va, t_va, Y_va

    test_frac = 0.1 if quick else 1.0
    X_te, t_te, Y_te = _load("test", test_frac)
    w_ipw = _ipw_weights("test", test_frac, t_te)
    result["splits"]["test"] = {"buckets": SPLITS["test"], "n": int(t_te.size),
                                "treatment_share": float(t_te.mean())}
    rng = np.random.default_rng(config.SEED)
    random_score = rng.random(t_te.size)
    for outcome in outcomes:
        y = Y_te[outcome]
        test_rows, test_ipw, curves = {}, {}, {}
        for kind, (_, learner, _) in chosen[outcome]["learners"].items():
            s = learner.predict_uplift(X_te)
            test_rows[kind] = _metrics(y, t_te, s)
            test_ipw[kind] = _metrics(y, t_te, s, w_ipw)
            curves[kind] = _curve_points(y, t_te, s)
            if kind == chosen[outcome]["best"]:
                best_score = s
            if kind == "response_model":
                resp_score = s
        test_rows["random"] = _metrics(y, t_te, random_score)
        test_ipw["random"] = _metrics(y, t_te, random_score, w_ipw)
        curves["random"] = _curve_points(y, t_te, random_score)
        n_boot = 20 if quick else 100
        boot = {
            "n_boot": n_boot,
            "method": "Poisson(1) weights over test users",
            "best_qini_auc": poisson_bootstrap(qini_auc_score, y, t_te, best_score, n_boot,
                                               seed=config.SEED),
            "best_capture_at_30": poisson_bootstrap(
                lambda *a, **kw: incremental_capture_at_k(*a, k=0.3, **kw), y, t_te,
                best_score, n_boot, seed=config.SEED + 1),
            "response_capture_at_30": poisson_bootstrap(
                lambda *a, **kw: incremental_capture_at_k(*a, k=0.3, **kw), y, t_te,
                resp_score, n_boot, seed=config.SEED + 2),
        }
        ate = float(y[t_te == 1].mean() - y[t_te == 0].mean())
        result[outcome].update({"test": test_rows, "test_ipw_weighted": test_ipw,
                                "test_bootstrap": boot, "test_ate": ate,
                                "test_curves": curves})
        print(outcome, "test", {k: round(v["qini_auc"], 4) for k, v in test_rows.items()},
              flush=True)
    result["runtime_seconds"] = round(time.time() - t0, 1)
    mem = psutil.Process().memory_info()
    result["peak_ram_gb"] = round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2)
    config.write_json("uplift_quick.json" if quick else "uplift.json", result)
    return result
