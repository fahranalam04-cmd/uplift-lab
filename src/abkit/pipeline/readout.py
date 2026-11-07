"""Full-population experiment readout on Criteo (13,979,592 users).

1. SRM against the documented 85/15 design.
2. Covariate balance: SMD per feature (full data, SQL), treatment-predictability AUC, and
   the file-order block structure (the CSV is stored as contiguous single-arm blocks).
3. ITT (visit, conversion): Welch + exact binomial bootstrap from counts.
4. Covariate-adjusted ITT from one streamed pass with cross-fitted LightGBM nuisances:
   Lin adjustment on the 12 features, CUPAC, CUPAC + features, AIPW (doubly robust), and a
   post-stratified estimate; Hausman-style test of raw vs adjusted.
5. CI-width reduction and equivalent sample-size saving of each adjusted estimator.
6. CACE / effect on the exposed (one-sided non-compliance, Wald estimator).
"""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import roc_auc_score

from abkit.balance import smd_from_moments, treatment_predictability_auc
from abkit.doubly_robust import aipw_from_moments, aipw_scores, hausman_randomization_test
from abkit.inference import (
    ArmSummary,
    EffectEstimate,
    bootstrap_from_binary_counts,
    welch_from_summaries,
)
from abkit.power import minimum_detectable_effect
from abkit.srm import srm_detectable_share_deviation, srm_test
from abkit.variance_reduction import (
    MomentAccumulator,
    check_pretreatment,
    equivalent_sample_size_saving,
    lin_from_moments,
)

from . import config, data

TRAIN_ROWS_PER_FOLD = 2_000_000
LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "n_estimators": 2_000,
    "n_jobs": config.THREADS,
    "verbose": -1,
}
PROPENSITY_CLIP = 0.01
PRIMARY_EFFECT = "aipw"  # effect estimate of record (doubly robust)
PRIMARY_VR = "cupac_plus_lin"  # headline variance-reduction estimator


def peak_ram_gb() -> float:
    mem = psutil.Process().memory_info()
    return round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2)


# --------------------------------------------------------------------------- diagnostics
def srm_and_balance() -> dict[str, Any]:
    con = data.connect()
    p = config.PARQUET.as_posix()
    feats = config.FEATURES
    agg = ", ".join(f"avg({f}), var_samp({f})" for f in feats)
    c, t = con.execute(
        f"SELECT treatment, count(*), {agg} FROM read_parquet('{p}') GROUP BY treatment "
        "ORDER BY treatment"
    ).fetchall()
    blocks = con.execute(
        f"""WITH x AS (SELECT row_id, treatment, visit, conversion,
                   CASE WHEN treatment <> lag(treatment) OVER (ORDER BY row_id) THEN 1 ELSE 0
                   END AS flip FROM read_parquet('{p}')),
             r AS (SELECT *, sum(flip) OVER (ORDER BY row_id ROWS UNBOUNDED PRECEDING) AS blk
                   FROM x)
            SELECT blk, min(row_id), max(row_id), any_value(treatment), count(*), avg(visit),
                   avg(conversion)
            FROM r GROUP BY blk ORDER BY blk"""
    ).fetchall()
    con.close()
    n0, n1 = int(c[1]), int(t[1])
    srm = srm_test([n1, n0], [config.DESIGN_TREATMENT_SHARE, 1 - config.DESIGN_TREATMENT_SHARE])
    m1, v1 = np.array(t[2::2], float), np.array(t[3::2], float)
    m0, v0 = np.array(c[2::2], float), np.array(c[3::2], float)
    smd = smd_from_moments(m1, v1, m0, v0)
    smd_se = float(np.sqrt(1 / n1 + 1 / n0))  # SE of an SMD under randomization
    sample = data.load_buckets(range(10), [*feats, "treatment"], sample_frac=0.0716)
    X = np.column_stack([sample[f] for f in feats])
    auc = treatment_predictability_auc(X, sample["treatment"], n_splits=3, seed=config.SEED)
    del sample, X
    return {
        "srm": {
            **srm.to_dict(),
            "design_ratio_source": "Criteo v2.1 dataset card: treatment ratio .85",
            "detectable_share_deviation": srm_detectable_share_deviation(
                n0 + n1, config.DESIGN_TREATMENT_SHARE, srm.threshold),
        },
        "balance": {
            "smd": {f: float(s) for f, s in zip(feats, smd, strict=True)},
            "max_abs_smd": float(np.abs(smd).max()),
            "smd_se_under_randomization": smd_se,
            "max_abs_smd_z": float(np.abs(smd).max() / smd_se),
            "n_features_abs_smd_over_0_1": int((np.abs(smd) > 0.1).sum()),
            "treatment_predictability_auc": auc,
            "treatment_auc_sample": "seeded ~7.16% Bernoulli subsample of all rows, "
                                    "HistGradientBoosting, 3-fold CV",
        },
        "file_blocks": {
            "n_blocks": len(blocks),
            "note": "contiguous runs of a single arm in original CSV order",
            "blocks": [{"block": int(b), "first_row": int(r0), "last_row": int(r1),
                        "treatment": int(tr), "n": int(n), "visit_rate": float(v),
                        "conversion_rate": float(cv)}
                       for b, r0, r1, tr, n, v, cv in blocks],
        },
    }


def raw_itt() -> dict[str, Any]:
    con = data.connect()
    p = config.PARQUET.as_posix()
    rows = con.execute(
        f"""SELECT treatment, count(*), sum(visit), sum(conversion), sum(exposure),
                   sum(visit * exposure), sum(conversion * exposure)
            FROM read_parquet('{p}') GROUP BY treatment ORDER BY treatment"""
    ).fetchall()
    con.close()
    (_, n0, v0, c0, e0, _, _), (_, n1, v1, c1, e1, ve1, ce1) = rows
    out: dict[str, Any] = {"n_treatment": n1, "n_control": n0, "control_exposed": int(e0),
                           "exposure_rate_treated": e1 / n1}
    for name, x1, x0, xe1 in (("visit", v1, v0, ve1), ("conversion", c1, c0, ce1)):
        s1, s0 = ArmSummary.from_binary_counts(n1, x1), ArmSummary.from_binary_counts(n0, x0)
        welch = welch_from_summaries(s1, s0)
        boot = bootstrap_from_binary_counts(n1, x1, n0, x0, n_boot=10_000, seed=config.SEED)
        out[name] = {
            "welch": welch.to_dict(),
            "bootstrap": boot.to_dict(),
            "_cov_y_exposure_treated": (xe1 / n1 - (x1 / n1) * (e1 / n1)) * n1 / (n1 - 1),
            "mde_at_this_n": minimum_detectable_effect(n0 + n1, s0.mean,
                                                       treat_fraction=n1 / (n0 + n1)),
        }
    return out


def cace(itt: EffectEstimate, pi: float, n1: int, cov_y_d: float | None) -> dict[str, float]:
    """Wald / Bloom estimator ITT / P(exposed | treated) with delta-method SE.

    ``cov_y_d`` (treated-arm covariance of outcome and exposure) is available for the raw
    estimator; for adjusted ITTs the covariance term is omitted (approximation).
    """
    var_pi = pi * (1 - pi) / (n1 - 1)
    var = itt.abs_se**2 / pi**2 + itt.abs_effect**2 * var_pi / pi**4
    if cov_y_d is not None:
        var -= 2 * itt.abs_effect / pi**3 * cov_y_d / n1
    est, se, z = itt.abs_effect / pi, float(np.sqrt(var)), stats.norm.ppf(0.975)
    return {"estimate": est, "se": se, "ci_low": est - z * se, "ci_high": est + z * se,
            "relative_to_control": est / itt.control_mean,
            "covariance_term": "included" if cov_y_d is not None else "omitted"}


# --------------------------------------------------------------------------- nuisances
def _fold_sample(fold: int, columns: list[str]) -> dict[str, np.ndarray]:
    buckets = [b for b in range(10) if b % 2 == fold]
    frac = min(1.0, TRAIN_ROWS_PER_FOLD / (len(buckets) * 1_397_960))
    return data.load_buckets(buckets, columns, sample_frac=frac, seed=config.SEED + fold)


def _fit(X: np.ndarray, y: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    rng = np.random.default_rng(seed)
    es = rng.random(X.shape[0]) < 0.1  # internal early-stopping holdout
    m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    m.fit(X[~es], y[~es], eval_X=(X[es],), eval_y=(y[es],),
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def train_nuisances() -> tuple[dict[tuple[str, int], lgb.LGBMClassifier], dict[str, Any]]:
    """Per fold k (bucket parity), fit on a sample of fold k only:
    pooled outcome models (CUPAC), per-arm outcome models (AIPW) and a propensity model."""
    check_pretreatment(config.FEATURES)
    models: dict[tuple[str, int], lgb.LGBMClassifier] = {}
    info: dict[str, Any] = {}
    for fold in (0, 1):
        s = _fold_sample(fold, [*config.FEATURES, "treatment", *config.OUTCOMES])
        X = np.column_stack([s[f] for f in config.FEATURES])
        t = s["treatment"].astype(int)
        specs: dict[str, tuple[np.ndarray, np.ndarray]] = {"propensity": (X, t)}
        for o in config.OUTCOMES:
            y = s[o].astype(int)
            specs[f"pooled_{o}"] = (X, y)
            specs[f"arm1_{o}"] = (X[t == 1], y[t == 1])
            specs[f"arm0_{o}"] = (X[t == 0], y[t == 0])
        for i, (name, (Xs, ys)) in enumerate(specs.items()):
            m = _fit(Xs, ys, config.SEED + 10 * fold + i)
            models[(name, fold)] = m
            info[f"{name}_fold{fold}"] = {"rows": int(ys.size),
                                          "best_iteration": int(m.best_iteration_)}
            print(f"fold{fold} {name}: {ys.size:,} rows, {m.best_iteration_} trees", flush=True)
        del s, X, specs
    return models, info


class _Pooled:
    """Streaming pooled mean / covariance (shifted for stability)."""

    def __init__(self, k: int) -> None:
        self.n, self.s, self.ss, self.shift = 0, np.zeros(k), np.zeros((k, k)), None

    def update(self, Z: np.ndarray) -> None:
        if self.shift is None:
            self.shift = Z.mean(axis=0)
        Zc = Z - self.shift
        self.n += Z.shape[0]
        self.s += Zc.sum(axis=0)
        self.ss += Zc.T @ Zc

    def moments(self) -> tuple[int, np.ndarray, np.ndarray]:
        assert self.shift is not None
        m = self.s / self.n
        return self.n, m + self.shift, (self.ss - self.n * np.outer(m, m)) / (self.n - 1)


def stream(models: dict[tuple[str, int], lgb.LGBMClassifier]) -> dict[str, Any]:
    """One pass: out-of-fold nuisance predictions, Lin/CUPAC moments, AIPW score moments."""
    outs = config.OUTCOMES
    pred_cols = [f"pred_{o}" for o in outs]
    acc = MomentAccumulator([*config.FEATURES, *pred_cols, *outs])
    aipw_acc = {o: _Pooled(2) for o in outs}
    writer = pq.ParquetWriter(
        config.CUPAC_PARQUET,
        pa.schema([("row_id", pa.int64()), ("bucket", pa.int8()), ("treatment", pa.int8()),
                   *[(o, pa.int8()) for o in outs], *[(c, pa.float32()) for c in pred_cols],
                   ("e_hat", pa.float32())]),
        compression="zstd",
    )
    oof: dict[str, list[np.ndarray]] = {k: [] for k in [*outs, "propensity"]}
    lab: dict[str, list[np.ndarray]] = {k: [] for k in [*outs, "propensity"]}
    cols = ["row_id", "bucket", "treatment", *outs, *config.FEATURES]
    for batch in data.iter_batches(cols):
        X = np.column_stack([batch[f] for f in config.FEATURES]).astype(np.float64)
        fold = batch["bucket"] % 2
        t = batch["treatment"].astype(np.float64)

        def oof_pred(name: str, X: np.ndarray = X, fold: np.ndarray = fold) -> np.ndarray:
            p = np.empty(X.shape[0])
            for k in (0, 1):
                held = fold == k
                p[held] = models[(name, 1 - k)].predict_proba(X[held])[:, 1]
            return p

        e = oof_pred("propensity")
        preds = {}
        for o in outs:
            y = batch[o].astype(np.float64)
            preds[o] = oof_pred(f"pooled_{o}")
            psi1, psi0 = aipw_scores(y, t, oof_pred(f"arm1_{o}"), oof_pred(f"arm0_{o}"), e,
                                     PROPENSITY_CLIP)
            aipw_acc[o].update(np.column_stack([psi1, psi0]))
            oof[o].append(preds[o].astype(np.float32))
            lab[o].append(batch[o])
        oof["propensity"].append(e.astype(np.float32))
        lab["propensity"].append(batch["treatment"])
        Z = np.column_stack([X, *[preds[o] for o in outs],
                             *[batch[o].astype(np.float64) for o in outs]])
        acc.update(Z, batch["treatment"])
        writer.write_table(pa.table({
            "row_id": batch["row_id"], "bucket": batch["bucket"], "treatment": batch["treatment"],
            **{o: batch[o] for o in outs},
            **{f"pred_{o}": preds[o].astype(np.float32) for o in outs},
            "e_hat": e.astype(np.float32),
        }))
    writer.close()
    quality: dict[str, Any] = {}
    for k in oof:
        p, y = np.concatenate(oof[k]), np.concatenate(lab[k]).astype(float)
        quality[k] = {"oof_auc": float(roc_auc_score(y, p)),
                      "oof_r2": float(1 - np.mean((y - p) ** 2) / np.var(y)), "n": int(y.size)}
        if k == "propensity":
            quality[k]["quantiles"] = {str(q): float(np.quantile(p, q))
                                       for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)}
            quality[k]["share_clipped"] = float(np.mean((p < PROPENSITY_CLIP)
                                                        | (p > 1 - PROPENSITY_CLIP)))
        del p, y
    return {"acc": acc, "aipw": aipw_acc, "quality": quality}


def post_stratified(n_strata: int = 10) -> dict[str, Any]:
    """ATE post-stratified on deciles of the out-of-fold CUPAC prediction (SQL)."""
    con = data.connect()
    c = config.CUPAC_PARQUET.as_posix()
    out: dict[str, Any] = {}
    for o in config.OUTCOMES:
        rows = con.execute(
            f"""WITH d AS (SELECT treatment, {o} AS y,
                               ntile({n_strata}) OVER (ORDER BY pred_{o}, row_id) AS q
                           FROM read_parquet('{c}'))
                SELECT q, count(*), sum(treatment), sum(y * treatment), sum(y * (1 - treatment)),
                       avg(treatment)
                FROM d GROUP BY q ORDER BY q"""
        ).fetchall()
        N = sum(r[1] for r in rows)
        tau = var = 0.0
        strata = []
        for q, n, n1, x1, x0, share in rows:
            n0 = n - n1
            p1, p0 = x1 / n1, x0 / n0
            w = n / N
            tau += w * (p1 - p0)
            var += w**2 * (p1 * (1 - p1) / (n1 - 1) + p0 * (1 - p0) / (n0 - 1))
            strata.append({"stratum": int(q), "n": int(n), "treatment_share": float(share),
                           "treated_rate": p1, "control_rate": p0})
        se = float(np.sqrt(var))
        out[o] = {"estimate": tau, "se": se, "ci_low": tau - 1.959964 * se,
                  "ci_high": tau + 1.959964 * se, "n_strata": n_strata, "strata": strata}
    con.close()
    return out


def adjusted_estimates(streamed: dict[str, Any], itt: dict[str, Any]) -> dict[str, Any]:
    acc: MomentAccumulator = streamed["acc"]
    n1, n0 = itt["n_treatment"], itt["n_control"]
    out: dict[str, Any] = {}
    for o in config.OUTCOMES:
        specs = {
            "raw": [],
            "lin_12_features": list(config.FEATURES),
            "cupac": [f"pred_{o}"],
            "cupac_plus_lin": [f"pred_{o}", *config.FEATURES],
        }
        ests: dict[str, EffectEstimate] = {
            name: lin_from_moments(acc, o, covs, method=name) for name, covs in specs.items()
        }
        n, mean, cov = streamed["aipw"][o].moments()
        ests["aipw"] = aipw_from_moments(n, mean[0], mean[1], cov[0, 0], cov[1, 1], cov[0, 1],
                                         n1, n0)
        raw = ests["raw"]
        res: dict[str, Any] = {}
        for name, est in ests.items():
            res[name] = {**est.to_dict(), **equivalent_sample_size_saving(raw.abs_se, est.abs_se),
                         "covariates": specs.get(name, "cross-fitted mu1, mu0, e (LightGBM)")}
            if name != "raw":
                res[name]["hausman_vs_raw"] = hausman_randomization_test(raw, est)
                res[name]["shift_vs_raw_relative"] = est.abs_effect / raw.abs_effect - 1
        out[o] = res
    return out


def run() -> dict[str, Any]:
    t0 = time.time()
    result: dict[str, Any] = {"scope": "all 13,979,592 users (full population), CPU only, "
                                       "4 threads"}
    result.update(srm_and_balance())
    print("srm/balance done", flush=True)
    itt = raw_itt()
    models, info = train_nuisances()
    streamed = stream(models)
    print("stream done", flush=True)
    adj = adjusted_estimates(streamed, itt)
    pi = itt["exposure_rate_treated"]
    for o in config.OUTCOMES:
        raw = EffectEstimate(**{k: v for k, v in adj[o]["raw"].items()
                                if k in EffectEstimate.__dataclass_fields__})
        prim = EffectEstimate(**{k: v for k, v in adj[o][PRIMARY_EFFECT].items()
                                 if k in EffectEstimate.__dataclass_fields__})
        itt[o]["cace_raw"] = cace(raw, pi, itt["n_treatment"],
                                  itt[o].pop("_cov_y_exposure_treated"))
        itt[o]["cace_adjusted"] = cace(prim, pi, itt["n_treatment"], None)
    result["itt"] = itt
    result["nuisance_models"] = {
        "params": LGB_PARAMS, "folds": info, "oof_quality": streamed["quality"],
        "train_rows_per_fold_cap": TRAIN_ROWS_PER_FOLD, "propensity_clip": PROPENSITY_CLIP,
        "crossfit": "2 folds by bucket parity; every row is scored by models trained on a "
                    "sample of the other fold only",
    }
    result["adjusted"] = adj
    result["post_stratified"] = post_stratified()
    result["primary_effect_estimator"] = PRIMARY_EFFECT
    result["primary_effect_reason"] = (
        "assignment is only conditionally random (pooled tests), so the doubly-robust AIPW "
        "with a cross-fitted propensity is the effect estimate of record")
    result["primary_variance_reduction_estimator"] = PRIMARY_VR
    result["runtime_seconds"] = round(time.time() - t0, 1)
    result["peak_ram_gb"] = peak_ram_gb()
    config.write_json("readout.json", result)
    return result
