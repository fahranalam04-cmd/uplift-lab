"""Always-valid inference under continuous monitoring: the mixture SPRT (mSPRT).

Following Johari, Pekelis & Walsh (2017) "Always valid inference", the effect estimate at
look n is treated as tau_hat_n ~ N(theta, v_n) (CLT, plug-in variance). Mixing the
likelihood ratio over theta ~ N(theta0, tau2) gives the closed form

    Lambda_n = sqrt(v / (v + tau2)) * exp(tau2 (tau_hat - theta0)^2 / (2 v (v + tau2)))

which is a non-negative martingale under H0, so by Ville's inequality
P(sup_n Lambda_n >= 1/alpha) <= alpha. The always-valid p-value is
p_n = min(1, min_{m <= n} 1 / Lambda_m) and the matching confidence sequence is the running
intersection of {theta : Lambda_n(theta) < 1/alpha}. Users may peek after every observation
and stop whenever p_n < alpha without inflating the type-I error.

``tau2`` (mixing variance) tunes where power is concentrated; a common choice is the square
of the effect size you most want to detect quickly (e.g. the planned MDE).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "SequentialResult",
    "always_valid_ci",
    "always_valid_p_values",
    "log_mixture_likelihood_ratio",
    "msprt_test",
    "sequential_from_cumulative_sums",
]


def log_mixture_likelihood_ratio(
    tau_hat: ArrayLike, var: ArrayLike, tau2: float, theta0: float = 0.0
) -> np.ndarray:
    """log Lambda for a normal mixing distribution N(theta0, tau2) over the effect."""
    if tau2 <= 0:
        raise ValueError("tau2 must be positive")
    th, v = np.asarray(tau_hat, dtype=float), np.asarray(var, dtype=float)
    return 0.5 * np.log(v / (v + tau2)) + tau2 * (th - theta0) ** 2 / (2 * v * (v + tau2))


def always_valid_p_values(
    tau_hat: ArrayLike, var: ArrayLike, tau2: float, theta0: float = 0.0
) -> np.ndarray:
    """Running-minimum always-valid p-values along the last axis (looks)."""
    log_lr = log_mixture_likelihood_ratio(tau_hat, var, tau2, theta0)
    p = np.minimum(1.0, np.exp(-log_lr))
    return np.minimum.accumulate(p, axis=-1)


def always_valid_ci(
    tau_hat: ArrayLike, var: ArrayLike, tau2: float, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence sequence: running intersection of {theta : Lambda_n(theta) < 1/alpha}."""
    th, v = np.asarray(tau_hat, dtype=float), np.asarray(var, dtype=float)
    half = np.sqrt(v * (v + tau2) / tau2 * (2 * np.log(1 / alpha) + np.log((v + tau2) / v)))
    lower = np.maximum.accumulate(th - half, axis=-1)
    upper = np.minimum.accumulate(th + half, axis=-1)
    return lower, upper


@dataclass(frozen=True)
class SequentialResult:
    p_values: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    alpha: float

    @property
    def rejected(self) -> bool | np.ndarray:
        out = (self.p_values < self.alpha).any(axis=-1)
        return bool(out) if np.ndim(out) == 0 else out

    @property
    def stop_index(self) -> int | np.ndarray:
        """First look at which H0 is rejected (-1 if never)."""
        hit = self.p_values < self.alpha
        idx = np.where(hit.any(axis=-1), hit.argmax(axis=-1), -1)
        return int(idx) if np.ndim(idx) == 0 else idx


def msprt_test(
    tau_hat: ArrayLike, var: ArrayLike, tau2: float, alpha: float = 0.05, theta0: float = 0.0
) -> SequentialResult:
    """Run the mSPRT over a sequence of looks (last axis); supports batched simulations."""
    p = always_valid_p_values(tau_hat, var, tau2, theta0)
    lo, hi = always_valid_ci(tau_hat, var, tau2, alpha)
    return SequentialResult(p_values=p, ci_low=lo, ci_high=hi, alpha=alpha)


def sequential_from_cumulative_sums(
    n1: ArrayLike,
    sum1: ArrayLike,
    sumsq1: ArrayLike,
    n0: ArrayLike,
    sum0: ArrayLike,
    sumsq0: ArrayLike,
    tau2: float,
    alpha: float = 0.05,
) -> tuple[SequentialResult, np.ndarray, np.ndarray]:
    """mSPRT from cumulative per-arm counts, sums and sums of squares at each look.

    Returns the result plus the per-look effect estimates and their variances.
    """
    n1a, n0a = np.asarray(n1, dtype=float), np.asarray(n0, dtype=float)
    m1, m0 = np.asarray(sum1) / n1a, np.asarray(sum0) / n0a
    var1 = (np.asarray(sumsq1) - n1a * m1**2) / (n1a - 1)
    var0 = (np.asarray(sumsq0) - n0a * m0**2) / (n0a - 1)
    # Guard against zero variance at very early looks with rare binary outcomes.
    v = np.maximum(var1, 1e-12) / n1a + np.maximum(var0, 1e-12) / n0a
    tau_hat = m1 - m0
    return msprt_test(tau_hat, v, tau2, alpha), tau_hat, v
