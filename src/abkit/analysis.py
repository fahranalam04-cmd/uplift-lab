"""One-call experiment readout for a tidy DataFrame (one row per randomization unit)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .balance import standardized_mean_differences
from .inference import bootstrap_diff_in_means, diff_in_means
from .power import minimum_detectable_effect
from .sequential import sequential_from_cumulative_sums
from .srm import srm_test
from .variance_reduction import (
    POST_TREATMENT_COLUMNS,
    check_pretreatment,
    equivalent_sample_size_saving,
    regression_adjusted,
)

__all__ = ["analyze_experiment"]

_MDE_Z = 1.959964 + 0.841621  # z_{0.975} + z_{0.80}: tau of the mSPRT mixture = 80%-power MDE


def _encode_treatment(col: pd.Series, treatment_value: Any | None) -> np.ndarray:
    values = pd.unique(col.dropna())
    if len(values) != 2:
        raise ValueError(f"treatment column must have exactly 2 distinct values, got {values!r}")
    if treatment_value is None:
        if set(values) <= {0, 1}:  # also covers True/False
            treatment_value = 1
        else:
            raise ValueError("treatment_value is required for non 0/1 treatment labels")
    if treatment_value not in set(values):
        raise ValueError(f"treatment_value {treatment_value!r} not found in column")
    return (col == treatment_value).to_numpy().astype(int)


def analyze_experiment(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_cols: Sequence[str],
    covariate_cols: Sequence[str] = (),
    *,
    treatment_value: Any | None = None,
    expected_treatment_share: float = 0.5,
    alpha: float = 0.05,
    n_boot: int = 2_000,
    order_col: str | None = None,
    n_looks: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """SRM, balance, Welch + bootstrap lift, optional Lin/CUPED adjustment and mSPRT path.

    ``covariate_cols`` must be measured before assignment; they are rejected if they overlap
    the outcomes, the treatment column or known post-treatment fields. If ``order_col`` is
    given (e.g. a timestamp), the always-valid mSPRT is evaluated at ``n_looks`` equally
    spaced looks in that order.
    """
    leaky = {treatment_col, *outcome_cols} | POST_TREATMENT_COLUMNS
    check_pretreatment(covariate_cols, leaky)
    needed = [treatment_col, *outcome_cols, *covariate_cols] + ([order_col] if order_col else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    data = df.dropna(subset=needed)
    if order_col:
        data = data.sort_values(order_col, kind="mergesort")
    t = _encode_treatment(data[treatment_col], treatment_value)
    n1, n0 = int(t.sum()), int(t.size - t.sum())
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_analyzed": int(t.size),
        "n_dropped_missing": int(len(df) - t.size),
        "srm": srm_test([n1, n0], [expected_treatment_share, 1 - expected_treatment_share])
        .to_dict(),
    }
    X = data[list(covariate_cols)].to_numpy(dtype=float) if covariate_cols else None
    if X is not None:
        out["balance"] = standardized_mean_differences(X, t, covariate_cols).to_dict("records")
    out["outcomes"] = {}
    for col in outcome_cols:
        y = data[col].to_numpy(dtype=float)
        raw = diff_in_means(y, t, alpha)
        res: dict[str, Any] = {
            "welch": raw.to_dict(),
            "bootstrap": bootstrap_diff_in_means(y, t, n_boot=n_boot, alpha=alpha, seed=seed)
            .to_dict(),
        }
        binary = bool(np.isin(y, (0.0, 1.0)).all())
        res["mde_at_this_n"] = minimum_detectable_effect(
            t.size, float(y[t == 0].mean()), sd=None if binary else float(y.std(ddof=1)),
            alpha=alpha, treat_fraction=n1 / t.size,
        ) if (not binary or 0 < y[t == 0].mean() < 1) else None
        if X is not None:
            adj = regression_adjusted(y, t, X, alpha=alpha, names=covariate_cols)
            res["adjusted"] = {**adj.to_dict(),
                               **equivalent_sample_size_saving(raw.abs_se, adj.abs_se)}
        if order_col:
            looks = np.unique(np.linspace(max(t.size // n_looks, 20), t.size, n_looks)
                              .astype(int)) - 1
            cum = [np.cumsum(a)[looks] for a in (t, y * t, y**2 * t, 1 - t, y * (1 - t),
                                                 y**2 * (1 - t))]
            seq, tau_hat, _ = sequential_from_cumulative_sums(
                *cum,
                tau2=max(_MDE_Z * raw.abs_se, 1e-12) ** 2, alpha=alpha,
            )
            res["sequential"] = {
                "looks_n": (looks + 1).tolist(),
                "estimate": tau_hat.tolist(),
                "always_valid_p": seq.p_values.tolist(),
                "ci_low": seq.ci_low.tolist(),
                "ci_high": seq.ci_high.tolist(),
                "rejected": bool(seq.rejected),
                "first_rejection_n": int(looks[seq.stop_index] + 1) if seq.rejected else None,
                "tau2_rule": "tau = fixed-horizon MDE at the analysed n (2.80 x SE)",
            }
        out["outcomes"][col] = res
    return out
