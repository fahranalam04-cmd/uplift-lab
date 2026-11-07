"""Difference-in-means inference for two-arm experiments.

Everything can be computed either from raw arrays or from per-arm sufficient statistics
(:class:`ArmSummary`), so the same code path serves a 1,000-row CSV and a 14M-row trial
aggregated in SQL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

__all__ = [
    "ArmSummary",
    "BootstrapResult",
    "EffectEstimate",
    "bootstrap_diff_in_means",
    "bootstrap_from_binary_counts",
    "diff_in_means",
    "effect_from_components",
    "welch_from_summaries",
]


@dataclass(frozen=True)
class ArmSummary:
    """Sufficient statistics of one arm: size, mean and unbiased (ddof=1) variance."""

    n: int
    mean: float
    var: float

    @classmethod
    def from_array(cls, y: ArrayLike) -> ArmSummary:
        arr = np.asarray(y, dtype=float)
        if arr.size < 2:
            raise ValueError("each arm needs at least 2 observations")
        return cls(n=int(arr.size), mean=float(arr.mean()), var=float(arr.var(ddof=1)))

    @classmethod
    def from_binary_counts(cls, n: int, successes: int) -> ArmSummary:
        if n < 2 or not 0 <= successes <= n:
            raise ValueError("need n >= 2 and 0 <= successes <= n")
        p = successes / n
        return cls(n=int(n), mean=p, var=p * (1.0 - p) * n / (n - 1))

    @classmethod
    def from_sums(cls, n: int, total: float, total_sq: float) -> ArmSummary:
        """Build from n, sum(y) and sum(y^2) (e.g. aggregated in SQL)."""
        mean = total / n
        var = (total_sq - n * mean * mean) / (n - 1)
        return cls(n=int(n), mean=float(mean), var=float(max(var, 0.0)))

    @property
    def se(self) -> float:
        return float(np.sqrt(self.var / self.n))


@dataclass(frozen=True)
class EffectEstimate:
    """Absolute and relative treatment effect with confidence intervals."""

    method: str
    n_treatment: int
    n_control: int
    control_mean: float
    treatment_mean: float
    abs_effect: float
    abs_se: float
    abs_ci_low: float
    abs_ci_high: float
    p_value: float
    rel_effect: float
    rel_se: float
    rel_ci_low: float
    rel_ci_high: float
    alpha: float
    df: float | None = None

    @property
    def abs_ci_width(self) -> float:
        return self.abs_ci_high - self.abs_ci_low

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict[str, float | int | str | None]:
        out = asdict(self)
        out["abs_ci_width"] = self.abs_ci_width
        return out


def _rel_se(
    tau: float, mu0: float, var_tau: float, var_mu0: float, cov_tau_mu0: float
) -> float:
    """Delta-method SE of tau / mu0 (relative lift)."""
    grad_tau, grad_mu0 = 1.0 / mu0, -tau / mu0**2
    var = grad_tau**2 * var_tau + grad_mu0**2 * var_mu0 + 2 * grad_tau * grad_mu0 * cov_tau_mu0
    return float(np.sqrt(max(var, 0.0)))


def effect_from_components(
    *,
    method: str,
    n_treatment: int,
    n_control: int,
    mu1: float,
    mu0: float,
    var_tau: float,
    var_mu0: float,
    cov_tau_mu0: float,
    alpha: float = 0.05,
    df: float | None = None,
) -> EffectEstimate:
    """Assemble an :class:`EffectEstimate` from (possibly covariate-adjusted) arm means.

    ``var_tau`` is Var(mu1_hat - mu0_hat); ``var_mu0`` and ``cov_tau_mu0`` feed the delta
    method for the relative lift ``tau / mu0``. Uses a t reference distribution when ``df``
    is given, otherwise the normal.
    """
    tau = mu1 - mu0
    se = float(np.sqrt(var_tau))
    crit = stats.t.ppf(1 - alpha / 2, df) if df is not None else stats.norm.ppf(1 - alpha / 2)
    stat = tau / se if se > 0 else np.inf
    p = 2 * (stats.t.sf(abs(stat), df) if df is not None else stats.norm.sf(abs(stat)))
    rel = tau / mu0 if mu0 != 0 else np.nan
    rel_se = _rel_se(tau, mu0, var_tau, var_mu0, cov_tau_mu0) if mu0 != 0 else np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    return EffectEstimate(
        method=method,
        n_treatment=int(n_treatment),
        n_control=int(n_control),
        control_mean=float(mu0),
        treatment_mean=float(mu1),
        abs_effect=float(tau),
        abs_se=se,
        abs_ci_low=float(tau - crit * se),
        abs_ci_high=float(tau + crit * se),
        p_value=float(p),
        rel_effect=float(rel),
        rel_se=float(rel_se),
        rel_ci_low=float(rel - z * rel_se),
        rel_ci_high=float(rel + z * rel_se),
        alpha=alpha,
        df=None if df is None else float(df),
    )


def welch_from_summaries(
    treatment: ArmSummary, control: ArmSummary, alpha: float = 0.05
) -> EffectEstimate:
    """Welch two-sample t-test (unequal variances) with delta-method relative lift."""
    v1, v0 = treatment.var / treatment.n, control.var / control.n
    df = (v1 + v0) ** 2 / (v1**2 / (treatment.n - 1) + v0**2 / (control.n - 1))
    return effect_from_components(
        method="welch",
        n_treatment=treatment.n,
        n_control=control.n,
        mu1=treatment.mean,
        mu0=control.mean,
        var_tau=v1 + v0,
        var_mu0=v0,
        cov_tau_mu0=-v0,
        alpha=alpha,
        df=df,
    )


def _split(y: ArrayLike, t: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y_arr, t_arr = np.asarray(y, dtype=float), np.asarray(t)
    if y_arr.shape != t_arr.shape:
        raise ValueError("y and t must have the same shape")
    if not set(np.unique(t_arr)).issubset({0, 1}):
        raise ValueError("t must be binary 0/1")
    return y_arr[t_arr == 1], y_arr[t_arr == 0]


def diff_in_means(y: ArrayLike, t: ArrayLike, alpha: float = 0.05) -> EffectEstimate:
    """Welch difference in means from raw outcome and 0/1 treatment arrays."""
    y1, y0 = _split(y, t)
    return welch_from_summaries(ArmSummary.from_array(y1), ArmSummary.from_array(y0), alpha)


@dataclass(frozen=True)
class BootstrapResult:
    n_boot: int
    abs_effect: float
    abs_se: float
    abs_ci_low: float
    abs_ci_high: float
    rel_effect: float
    rel_ci_low: float
    rel_ci_high: float
    alpha: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _bootstrap_summary(
    m1: np.ndarray, m0: np.ndarray, point_abs: float, point_rel: float, alpha: float
) -> BootstrapResult:
    diffs, rels = m1 - m0, m1 / m0 - 1.0
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return BootstrapResult(
        n_boot=int(diffs.size),
        abs_effect=float(point_abs),
        abs_se=float(diffs.std(ddof=1)),
        abs_ci_low=float(np.percentile(diffs, lo)),
        abs_ci_high=float(np.percentile(diffs, hi)),
        rel_effect=float(point_rel),
        rel_ci_low=float(np.percentile(rels, lo)),
        rel_ci_high=float(np.percentile(rels, hi)),
        alpha=alpha,
    )


def bootstrap_from_binary_counts(
    n1: int,
    x1: int,
    n0: int,
    x0: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Stratified nonparametric bootstrap for binary outcomes, computed exactly from counts.

    Resampling n Bernoulli rows with replacement yields a success count ~ Binomial(n, p_hat),
    so the bootstrap distribution can be drawn without materialising the rows. This is
    identical in distribution to the row-level bootstrap and works for 14M users in ms.
    """
    rng = np.random.default_rng(seed)
    m1 = rng.binomial(n1, x1 / n1, size=n_boot) / n1
    m0 = rng.binomial(n0, x0 / n0, size=n_boot) / n0
    return _bootstrap_summary(m1, m0, x1 / n1 - x0 / n0, (x1 / n1) / (x0 / n0) - 1, alpha)


def bootstrap_diff_in_means(
    y: ArrayLike,
    t: ArrayLike,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
    batch: int = 100,
) -> BootstrapResult:
    """Stratified percentile bootstrap (resample within each arm) for abs and relative lift.

    Binary outcomes take the exact binomial shortcut; other outcomes are resampled row-wise
    in batches to bound memory.
    """
    y1, y0 = _split(y, t)
    if np.isin(y1, (0.0, 1.0)).all() and np.isin(y0, (0.0, 1.0)).all():
        return bootstrap_from_binary_counts(
            y1.size, int(y1.sum()), y0.size, int(y0.sum()), n_boot, alpha, seed
        )
    rng = np.random.default_rng(seed)
    m1, m0 = np.empty(n_boot), np.empty(n_boot)
    for start in range(0, n_boot, batch):
        b = min(batch, n_boot - start)
        m1[start : start + b] = y1[rng.integers(0, y1.size, size=(b, y1.size))].mean(axis=1)
        m0[start : start + b] = y0[rng.integers(0, y0.size, size=(b, y0.size))].mean(axis=1)
    return _bootstrap_summary(m1, m0, y1.mean() - y0.mean(), y1.mean() / y0.mean() - 1, alpha)
