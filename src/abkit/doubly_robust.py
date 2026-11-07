"""Doubly-robust (AIPW) ATE and a Hausman-style randomization check.

AIPW (Robins et al. 1994; cross-fitted as in Chernozhukov et al. 2018) uses per-unit
influence scores

    psi1 = mu1(x) + t (y - mu1(x)) / e(x)
    psi0 = mu0(x) + (1 - t) (y - mu0(x)) / (1 - e(x))

with ATE = mean(psi1 - psi0). It is consistent if either the outcome models or the
propensity model are correct, which matters when assignment is only *conditionally*
random (e.g. several tests with different treatment ratios pooled into one file).
The scores reduce to n, means and a 2x2 covariance, so they stream like everything else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.model_selection import KFold

from .inference import EffectEstimate, effect_from_components

__all__ = [
    "aipw",
    "aipw_from_moments",
    "aipw_scores",
    "hausman_randomization_test",
]


def aipw_scores(
    y: ArrayLike, t: ArrayLike, mu1: ArrayLike, mu0: ArrayLike, e: ArrayLike, clip: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Influence scores (psi1, psi0); propensities are clipped to [clip, 1 - clip]."""
    y_a, t_a = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    m1, m0 = np.asarray(mu1, dtype=float), np.asarray(mu0, dtype=float)
    e_a = np.clip(np.asarray(e, dtype=float), clip, 1 - clip)
    psi1 = m1 + t_a * (y_a - m1) / e_a
    psi0 = m0 + (1 - t_a) * (y_a - m0) / (1 - e_a)
    return psi1, psi0


def aipw_from_moments(
    n: int,
    mean_psi1: float,
    mean_psi0: float,
    var_psi1: float,
    var_psi0: float,
    cov_psi10: float,
    n_treatment: int,
    n_control: int,
    alpha: float = 0.05,
) -> EffectEstimate:
    """AIPW estimate from the mean / covariance of the influence scores (ddof=1)."""
    var_tau = (var_psi1 + var_psi0 - 2 * cov_psi10) / n
    cov_tau_mu0 = (cov_psi10 - var_psi0) / n
    return effect_from_components(
        method="aipw", n_treatment=n_treatment, n_control=n_control, mu1=mean_psi1,
        mu0=mean_psi0, var_tau=var_tau, var_mu0=var_psi0 / n, cov_tau_mu0=cov_tau_mu0,
        alpha=alpha,
    )


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def aipw(
    y: ArrayLike,
    t: ArrayLike,
    X: ArrayLike,
    make_outcome_model: Callable[[], Any],
    make_propensity_model: Callable[[], Any] | None = None,
    n_folds: int = 2,
    seed: int = 0,
    clip: float = 0.01,
    alpha: float = 0.05,
) -> tuple[EffectEstimate, dict[str, np.ndarray]]:
    """Cross-fitted AIPW. Without a propensity model the constant mean(t) is used, which is
    the right choice for a single, cleanly randomized experiment."""
    X_a, t_a, y_a = np.asarray(X, dtype=float), np.asarray(t).astype(int), np.asarray(y, float)
    if X_a.ndim == 1:
        X_a = X_a[:, None]
    n = y_a.size
    mu1, mu0, e = np.empty(n), np.empty(n), np.full(n, t_a.mean())
    for train, held in KFold(n_folds, shuffle=True, random_state=seed).split(X_a):
        Xt, tt, yt = X_a[train], t_a[train], y_a[train]
        mu1[held] = _predict(make_outcome_model().fit(Xt[tt == 1], yt[tt == 1]), X_a[held])
        mu0[held] = _predict(make_outcome_model().fit(Xt[tt == 0], yt[tt == 0]), X_a[held])
        if make_propensity_model is not None:
            e[held] = _predict(make_propensity_model().fit(Xt, tt), X_a[held])
    psi1, psi0 = aipw_scores(y_a, t_a, mu1, mu0, e, clip)
    c = np.cov(psi1, psi0, ddof=1)
    est = aipw_from_moments(n, psi1.mean(), psi0.mean(), c[0, 0], c[1, 1], c[0, 1],
                            int(t_a.sum()), int(n - t_a.sum()), alpha)
    return est, {"mu1": mu1, "mu0": mu0, "e": e}


def hausman_randomization_test(raw: EffectEstimate, adjusted: EffectEstimate) -> dict[str, float]:
    """Is the raw vs covariate-adjusted gap larger than chance allows under randomization?

    Under clean randomization both estimators are consistent for the same ATE and the
    adjusted one is (asymptotically) efficient, so Var(raw - adj) ~ Var(raw) - Var(adj)
    (Hausman 1978). A large |z| means assignment is related to the covariates.
    """
    diff = raw.abs_effect - adjusted.abs_effect
    var = raw.abs_se**2 - adjusted.abs_se**2
    if var <= 0:
        return {"difference": diff, "z": float("nan"), "p_value": float("nan")}
    z = diff / np.sqrt(var)
    return {"difference": float(diff), "se_difference": float(np.sqrt(var)), "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}
