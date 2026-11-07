"""Power, minimum detectable effect (MDE) and sample-size calculations.

Normal-approximation (z-test) design for a difference in means between a treatment arm with
share ``treat_fraction`` and a control arm. For binary metrics the per-arm variances are
``p0 (1 - p0)`` and ``p1 (1 - p1)`` (unpooled). ``variance_reduction`` (R^2 of a CUPED /
regression-adjustment covariate set) shrinks the variance by ``1 - R^2``.

The two-sided power includes both rejection tails, which matches
``statsmodels.stats.power.NormalIndPower`` exactly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy import optimize, stats

__all__ = [
    "SampleSize",
    "minimum_detectable_effect",
    "power_at",
    "required_sample_size",
    "variance_of_effect",
]


@dataclass(frozen=True)
class SampleSize:
    n_total: int
    n_treatment: int
    n_control: int
    n_total_exact: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _check(alpha: float, treat_fraction: float, variance_reduction: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < treat_fraction < 1:
        raise ValueError("treat_fraction must be in (0, 1)")
    if not 0 <= variance_reduction < 1:
        raise ValueError("variance_reduction must be in [0, 1)")


def _arm_variances(baseline: float, effect: float, sd: float | None) -> tuple[float, float]:
    if sd is not None:
        return sd**2, sd**2
    p0, p1 = baseline, baseline + effect
    if not (0 < p0 < 1 and 0 < p1 < 1):
        raise ValueError("binary metric needs baseline and baseline + effect in (0, 1)")
    return p1 * (1 - p1), p0 * (1 - p0)


def variance_of_effect(
    n_total: float,
    baseline: float,
    effect: float,
    sd: float | None = None,
    treat_fraction: float = 0.5,
    variance_reduction: float = 0.0,
) -> float:
    """Var(tau_hat) = (1 - R^2) * (var1 / n1 + var0 / n0)."""
    v1, v0 = _arm_variances(baseline, effect, sd)
    n1, n0 = n_total * treat_fraction, n_total * (1 - treat_fraction)
    return (1 - variance_reduction) * (v1 / n1 + v0 / n0)


def power_at(
    n_total: float,
    baseline: float,
    effect: float,
    *,
    sd: float | None = None,
    alpha: float = 0.05,
    treat_fraction: float = 0.5,
    two_sided: bool = True,
    variance_reduction: float = 0.0,
) -> float:
    """Power of the z-test to detect an absolute ``effect`` with ``n_total`` users."""
    _check(alpha, treat_fraction, variance_reduction)
    se = math.sqrt(
        variance_of_effect(n_total, baseline, effect, sd, treat_fraction, variance_reduction)
    )
    shift = effect / se
    if two_sided:
        z = stats.norm.ppf(1 - alpha / 2)
        return float(stats.norm.sf(z - shift) + stats.norm.cdf(-z - shift))
    z = stats.norm.ppf(1 - alpha)
    return float(stats.norm.sf(z - shift))


def required_sample_size(
    baseline: float,
    mde: float,
    *,
    relative: bool = False,
    sd: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    treat_fraction: float = 0.5,
    two_sided: bool = True,
    variance_reduction: float = 0.0,
) -> SampleSize:
    """Total users needed to detect ``mde`` (absolute, or relative to ``baseline``)."""
    _check(alpha, treat_fraction, variance_reduction)
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    effect = mde * baseline if relative else mde
    if effect == 0:
        raise ValueError("mde must be non-zero")
    v1, v0 = _arm_variances(baseline, effect, sd)
    z_a = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    # Closed form (ignores the far rejection tail) as the bracket for the exact solve.
    approx = (z_a + z_b) ** 2 * (1 - variance_reduction) * (
        v1 / treat_fraction + v0 / (1 - treat_fraction)
    ) / effect**2

    def gap(n: float) -> float:
        return (
            power_at(
                n,
                baseline,
                effect,
                sd=sd,
                alpha=alpha,
                treat_fraction=treat_fraction,
                two_sided=two_sided,
                variance_reduction=variance_reduction,
            )
            - power
        )

    n_exact = optimize.brentq(gap, approx * 0.5, approx * 1.01 + 10, xtol=1e-6)
    n_treat = math.ceil(n_exact * treat_fraction)
    n_ctrl = math.ceil(n_exact * (1 - treat_fraction))
    n_total = n_treat + n_ctrl
    return SampleSize(n_total=n_total, n_treatment=n_treat, n_control=n_ctrl,
                      n_total_exact=float(n_exact))


def minimum_detectable_effect(
    n_total: float,
    baseline: float,
    *,
    sd: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    treat_fraction: float = 0.5,
    two_sided: bool = True,
    variance_reduction: float = 0.0,
    relative: bool = False,
) -> float:
    """Smallest positive effect detectable with the given power (absolute unless ``relative``)."""
    _check(alpha, treat_fraction, variance_reduction)

    def gap(effect: float) -> float:
        return (
            power_at(
                n_total,
                baseline,
                effect,
                sd=sd,
                alpha=alpha,
                treat_fraction=treat_fraction,
                two_sided=two_sided,
                variance_reduction=variance_reduction,
            )
            - power
        )

    upper = (1 - baseline) * (1 - 1e-9) if sd is None else 100 * (sd or 1.0)
    if gap(upper) < 0:
        raise ValueError("no detectable effect within the metric's range at this sample size")
    eff = optimize.brentq(gap, 1e-12 * max(abs(baseline), 1.0), upper, xtol=1e-14)
    return float(eff / baseline) if relative else float(eff)

