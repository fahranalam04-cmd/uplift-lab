"""Sample-ratio-mismatch (SRM) check: chi-square goodness of fit against the design ratio."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

__all__ = ["SRMResult", "srm_detectable_share_deviation", "srm_test"]


@dataclass(frozen=True)
class SRMResult:
    counts: tuple[int, ...]
    expected_ratios: tuple[float, ...]
    observed_ratios: tuple[float, ...]
    chi2: float
    dof: int
    p_value: float
    threshold: float
    mismatch: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def srm_test(
    counts: Sequence[int], expected_ratios: Sequence[float], threshold: float = 0.001
) -> SRMResult:
    """Chi-square test of observed arm counts against the *intended* allocation.

    The conventional SRM alarm threshold is p < 0.001 (a strict level, because SRM checks run
    on every experiment and a false alarm halts analysis). ``expected_ratios`` is normalised,
    so ``[85, 15]`` and ``[0.85, 0.15]`` are equivalent.
    """
    obs = np.asarray(counts, dtype=float)
    ratios = np.asarray(expected_ratios, dtype=float)
    if obs.shape != ratios.shape or obs.size < 2:
        raise ValueError("counts and expected_ratios must have the same length >= 2")
    if (ratios <= 0).any() or (obs < 0).any():
        raise ValueError("ratios must be positive and counts non-negative")
    ratios = ratios / ratios.sum()
    expected = ratios * obs.sum()
    chi2, p = stats.chisquare(obs, expected)
    return SRMResult(
        counts=tuple(int(c) for c in obs),
        expected_ratios=tuple(float(r) for r in ratios),
        observed_ratios=tuple(float(o) for o in obs / obs.sum()),
        chi2=float(chi2),
        dof=int(obs.size - 1),
        p_value=float(p),
        threshold=threshold,
        mismatch=bool(p < threshold),
    )


def srm_detectable_share_deviation(
    n_total: int, expected_share: float, threshold: float = 0.001
) -> float:
    """Smallest |observed - expected| share of one arm (2 arms) that trips the SRM alarm.

    For two arms the chi-square statistic equals z^2 with
    z = (p_obs - p) / sqrt(p (1 - p) / n), so the alarm fires once the share deviates by
    z_{1 - threshold/2} * sqrt(p (1 - p) / n).
    """
    z = stats.norm.ppf(1 - threshold / 2)
    return float(z * np.sqrt(expected_share * (1 - expected_share) / n_total))
