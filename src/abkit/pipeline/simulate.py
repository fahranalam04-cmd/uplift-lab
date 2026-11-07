"""Simulation studies that validate the methodology on real Criteo outcomes.

All simulations resample (with replacement) from the 2,096,937 *control-arm* users, so the
null of no effect holds by construction while outcome distributions (a 3.8% visit rate and a
0.2% conversion rate) are real. Fake assignments use the trial's 85/15 Bernoulli design.

1. A/A at a fixed horizon: false-positive rate of Welch, bootstrap and CUPAC-adjusted tests.
2. Peeking: a fixed-horizon test re-run at every look vs the mSPRT (always-valid p-values).
3. Power: analytic calculator vs empirical power with an injected effect, raw and CUPAC.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import stats

from abkit.doubly_robust import hausman_randomization_test
from abkit.inference import bootstrap_from_binary_counts, diff_in_means
from abkit.power import minimum_detectable_effect, power_at
from abkit.sequential import sequential_from_cumulative_sums
from abkit.variance_reduction import regression_adjusted

from . import config, data

ALPHA = 0.05
TREAT_SHARE = config.DESIGN_TREATMENT_SHARE


def _rate_ci(k: int, n: int) -> dict[str, float]:
    """Proportion with a Wilson 95% interval."""
    lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return {"rate": k / n, "ci_low": float(lo), "ci_high": float(hi), "k": k, "n_sims": n}


def load_bank() -> dict[str, np.ndarray]:
    if not config.CUPAC_PARQUET.exists():
        raise FileNotFoundError("run `uplift-lab readout` first (needs CUPAC predictions)")
    con = data.connect()
    tbl = con.sql(
        f"SELECT visit, conversion, pred_visit, pred_conversion "
        f"FROM read_parquet('{config.CUPAC_PARQUET.as_posix()}') WHERE treatment = 0 "
        "ORDER BY row_id"
    ).to_arrow_table()
    con.close()
    return {c: tbl.column(c).to_numpy() for c in tbl.column_names}


def aa_fixed_horizon(bank: dict[str, np.ndarray], n_sims: int, n_users: int,
                     seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    N = bank["visit"].size
    rejects: dict[str, int] = {}
    pvals: dict[str, list[float]] = {}
    se_ratio: dict[str, list[float]] = {o: [] for o in config.OUTCOMES}
    for _ in range(n_sims):
        idx = rng.integers(0, N, n_users)
        t = (rng.random(n_users) < TREAT_SHARE).astype(np.int8)
        for o in config.OUTCOMES:
            y = bank[o][idx].astype(float)
            raw = diff_in_means(y, t, ALPHA)
            adj = regression_adjusted(y, t, bank[f"pred_{o}"][idx], ALPHA)
            n1 = int(t.sum())
            boot = bootstrap_from_binary_counts(
                n1, int(y[t == 1].sum()), n_users - n1, int(y[t == 0].sum()),
                n_boot=1_000, seed=int(rng.integers(1 << 31)),
            )
            haus = hausman_randomization_test(raw, adj)
            res = {
                f"{o}_welch": raw.p_value < ALPHA,
                f"{o}_cupac": adj.p_value < ALPHA,
                f"{o}_bootstrap": not (boot.abs_ci_low <= 0 <= boot.abs_ci_high),
                f"{o}_hausman_raw_vs_cupac": bool(haus["p_value"] < ALPHA),
            }
            for k, v in res.items():
                rejects[k] = rejects.get(k, 0) + int(v)
            pvals.setdefault(f"{o}_welch", []).append(raw.p_value)
            pvals.setdefault(f"{o}_cupac", []).append(adj.p_value)
            se_ratio[o].append(adj.abs_se / raw.abs_se)
    out: dict[str, Any] = {
        "n_sims": n_sims, "n_users_per_sim": n_users, "alpha": ALPHA,
        "design": "resample control-arm users with replacement; Bernoulli(0.85) fake assignment",
        "false_positive_rate": {k: _rate_ci(v, n_sims) for k, v in rejects.items()},
        "p_value_uniformity_ks": {
            k: {"statistic": float(r.statistic), "p_value": float(r.pvalue)}
            for k, r in ((k, stats.kstest(v, "uniform")) for k, v in pvals.items())
        },
        "mean_se_ratio_cupac_vs_raw": {o: float(np.mean(v)) for o, v in se_ratio.items()},
    }
    return out


def _cumulative_looks(
    y: np.ndarray, t: np.ndarray, looks: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Cumulative per-arm counts and successes at each look (axis 1)."""
    yt, yc = y * t, y * (1 - t)
    n1 = np.cumsum(t, axis=1, dtype=np.int64)[:, looks]
    x1 = np.cumsum(yt, axis=1, dtype=np.int64)[:, looks]
    x0 = np.cumsum(yc, axis=1, dtype=np.int64)[:, looks]
    n0 = (looks + 1)[None, :] - n1
    return n1, x1, n0, x0


def _z_stats(n1: np.ndarray, x1: np.ndarray, n0: np.ndarray, x0: np.ndarray) -> np.ndarray:
    p1, p0 = x1 / n1, x0 / n0
    v = p1 * (1 - p1) / np.maximum(n1 - 1, 1) + p0 * (1 - p0) / np.maximum(n0 - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(v > 0, (p1 - p0) / np.sqrt(v), 0.0)
    return z


def peeking(bank: dict[str, np.ndarray], n_sims: int, n_users: int, n_looks: int, seed: int,
            batch: int = 25) -> dict[str, Any]:
    """Naive repeated significance testing vs mSPRT on the visit metric (A/A and A/B)."""
    rng = np.random.default_rng(seed)
    y_bank = bank["visit"].astype(np.int8)
    p0 = float(y_bank.mean())
    mde = minimum_detectable_effect(n_users, p0, treat_fraction=TREAT_SHARE)
    tau2 = mde**2
    looks = np.linspace(n_users // n_looks, n_users, n_looks).astype(int) - 1
    look_grid = [k for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if k <= n_looks
                 and n_looks % k == 0]
    zcrit = stats.norm.ppf(1 - ALPHA / 2)
    naive_rej = {k: 0 for k in look_grid}
    msprt_rej_aa = 0
    ab = {"fixed_final_reject": 0, "msprt_reject": 0, "msprt_stop_n": []}
    q = mde / (1 - p0)  # flip prob for treated zeros -> absolute lift of exactly mde in mean
    for start in range(0, n_sims, batch):
        b = min(batch, n_sims - start)
        idx = rng.integers(0, y_bank.size, (b, n_users))
        y = y_bank[idx]
        t = (rng.random((b, n_users)) < TREAT_SHARE).astype(np.int8)
        # ---- A/A ----
        n1, x1, n0, x0 = _cumulative_looks(y, t, looks)
        z = _z_stats(n1, x1, n0, x0)
        for k in look_grid:
            sub = np.arange(n_looks // k - 1, n_looks, n_looks // k)
            naive_rej[k] += int((np.abs(z[:, sub]) > zcrit).any(axis=1).sum())
        res, _, _ = sequential_from_cumulative_sums(n1, x1, x1, n0, x0, x0, tau2, ALPHA)
        msprt_rej_aa += int(res.rejected.sum())
        # ---- A/B with an injected lift equal to the fixed-horizon MDE ----
        lift = ((rng.random((b, n_users)) < q) & (t == 1)).astype(np.int8)
        y_ab = np.maximum(y, lift)
        n1, x1, n0, x0 = _cumulative_looks(y_ab, t, looks)
        z = _z_stats(n1, x1, n0, x0)
        ab["fixed_final_reject"] += int((np.abs(z[:, -1]) > zcrit).sum())
        res, _, _ = sequential_from_cumulative_sums(n1, x1, x1, n0, x0, x0, tau2, ALPHA)
        rej = np.asarray(res.rejected)
        ab["msprt_reject"] += int(rej.sum())
        stop = np.asarray(res.stop_index)
        ab["msprt_stop_n"].extend((looks[stop[rej]] + 1).tolist())
    stop_n = np.asarray(ab["msprt_stop_n"], dtype=float)
    return {
        "metric": "visit (control-arm base rate)",
        "base_rate": p0,
        "n_sims": n_sims,
        "n_users_per_sim": n_users,
        "max_looks": n_looks,
        "msprt_tau2": tau2,
        "msprt_tau2_rule": "tau = fixed-horizon MDE at n_users (80% power, alpha 0.05)",
        "aa_false_positive_rate_naive_by_looks": {
            str(k): _rate_ci(v, n_sims) for k, v in naive_rej.items()
        },
        "aa_false_positive_rate_msprt": _rate_ci(msprt_rej_aa, n_sims),
        "ab_injected_abs_lift": mde,
        "ab_injected_rel_lift": mde / p0,
        "ab_power_fixed_horizon_final_look": _rate_ci(ab["fixed_final_reject"], n_sims),
        "ab_power_msprt_any_look": _rate_ci(ab["msprt_reject"], n_sims),
        "ab_msprt_median_stop_n": float(np.median(stop_n)) if stop_n.size else None,
        "ab_msprt_mean_stop_fraction": float(stop_n.mean() / n_users) if stop_n.size else None,
    }


def power_validation(bank: dict[str, np.ndarray], n_grid: list[int], n_sims: int, rel_lift: float,
                     variance_reduction: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_bank, pred_bank = bank["visit"].astype(np.int8), bank["pred_visit"].astype(float)
    p0 = float(y_bank.mean())
    delta = rel_lift * p0
    q = delta / (1 - p0)
    rows = []
    for n in n_grid:
        rej_raw = rej_adj = 0
        for _ in range(n_sims):
            idx = rng.integers(0, y_bank.size, n)
            t = (rng.random(n) < TREAT_SHARE).astype(np.int8)
            y = np.maximum(y_bank[idx], ((rng.random(n) < q) & (t == 1))).astype(float)
            rej_raw += diff_in_means(y, t, ALPHA).p_value < ALPHA
            rej_adj += regression_adjusted(y, t, pred_bank[idx], ALPHA).p_value < ALPHA
        rows.append({
            "n_users": n,
            "analytic_power_raw": power_at(n, p0, delta, treat_fraction=TREAT_SHARE),
            "simulated_power_raw": _rate_ci(int(rej_raw), n_sims),
            "analytic_power_cupac": power_at(n, p0, delta, treat_fraction=TREAT_SHARE,
                                             variance_reduction=variance_reduction),
            "simulated_power_cupac": _rate_ci(int(rej_adj), n_sims),
        })
        print(f"power n={n}: raw {rows[-1]['simulated_power_raw']['rate']:.3f} vs "
              f"{rows[-1]['analytic_power_raw']:.3f}", flush=True)
    gaps = [abs(r["simulated_power_raw"]["rate"] - r["analytic_power_raw"]) for r in rows]
    gaps_adj = [abs(r["simulated_power_cupac"]["rate"] - r["analytic_power_cupac"]) for r in rows]
    return {
        "metric": "visit", "base_rate": p0, "rel_lift": rel_lift, "abs_lift": delta,
        "n_sims_per_point": n_sims, "variance_reduction_input": variance_reduction,
        "variance_reduction_source": "readout.json adjusted.visit.cupac.variance_reduction",
        "grid": rows,
        "max_abs_gap_raw": float(max(gaps)),
        "max_abs_gap_cupac": float(max(gaps_adj)),
    }


def run(quick: bool = False) -> dict[str, Any]:
    t0 = time.time()
    bank = load_bank()
    vr = config.read_json("readout.json")["adjusted"]["visit"]["cupac"]["variance_reduction"]
    out: dict[str, Any] = {"bank_size": int(bank["visit"].size), "seed": config.SEED,
                           "hardware": "CPU only (i7-11800H, 4 threads)"}
    out["aa"] = aa_fixed_horizon(bank, n_sims=500 if quick else 10_000, n_users=100_000,
                                 seed=config.SEED)
    print("aa done", flush=True)
    out["peeking"] = peeking(bank, n_sims=200 if quick else 5_000, n_users=200_000,
                             n_looks=1_000, seed=config.SEED + 1)
    print("peeking done", flush=True)
    out["power"] = power_validation(
        bank, n_grid=[50_000, 100_000, 200_000, 400_000], n_sims=200 if quick else 2_000,
        rel_lift=0.10, variance_reduction=float(vr), seed=config.SEED + 2,
    )
    out["runtime_seconds"] = round(time.time() - t0, 1)
    config.write_json("simulations_quick.json" if quick else "simulations.json", out)
    return out
