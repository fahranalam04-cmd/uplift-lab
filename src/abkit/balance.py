"""Covariate balance diagnostics for randomized experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

__all__ = [
    "auc_standard_error",
    "smd_from_moments",
    "standardized_mean_differences",
    "treatment_predictability_auc",
]


def smd_from_moments(
    mean1: ArrayLike, var1: ArrayLike, mean0: ArrayLike, var0: ArrayLike
) -> np.ndarray:
    """Standardized mean difference (mean1 - mean0) / sqrt((var1 + var0) / 2).

    Rule of thumb: |SMD| < 0.1 is considered balanced. Zero-variance features return 0.
    """
    m1, v1, m0, v0 = (np.asarray(a, dtype=float) for a in (mean1, var1, mean0, var0))
    pooled = np.sqrt((v1 + v0) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        smd = np.where(pooled > 0, (m1 - m0) / pooled, 0.0)
    return smd


def standardized_mean_differences(
    X: ArrayLike, t: ArrayLike, names: Sequence[str] | None = None
) -> pd.DataFrame:
    """Per-feature balance table: arm means and SMD."""
    X_arr, t_arr = np.asarray(X, dtype=float), np.asarray(t)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    x1, x0 = X_arr[t_arr == 1], X_arr[t_arr == 0]
    smd = smd_from_moments(x1.mean(0), x1.var(0, ddof=1), x0.mean(0), x0.var(0, ddof=1))
    cols = list(names) if names is not None else [f"x{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(
        {"feature": cols, "mean_treatment": x1.mean(0), "mean_control": x0.mean(0), "smd": smd}
    )


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of a ROC AUC."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (
        n_pos * n_neg
    )
    return float(np.sqrt(var))


def treatment_predictability_auc(
    X: ArrayLike, t: ArrayLike, n_splits: int = 3, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Cross-validated AUC of a gradient-boosted classifier predicting treatment from features.

    Under successful randomization, pre-treatment features carry no information about
    assignment and the out-of-fold AUC should be ~0.5. A CI that excludes 0.5 by a material
    margin points to broken randomization, leakage or post-treatment features.
    """
    X_arr, t_arr = np.asarray(X, dtype=float), np.asarray(t).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X_arr, t_arr, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(t_arr, proba))
    se = auc_standard_error(auc, int(t_arr.sum()), int((1 - t_arr).sum()))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"auc": auc, "se": se, "ci_low": auc - z * se, "ci_high": auc + z * se,
            "n": int(t_arr.size)}
