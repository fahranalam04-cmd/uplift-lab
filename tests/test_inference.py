import numpy as np
import pytest
from scipy import stats

from abkit.inference import (
    ArmSummary,
    bootstrap_diff_in_means,
    bootstrap_from_binary_counts,
    diff_in_means,
    welch_from_summaries,
)


@pytest.fixture
def continuous():
    rng = np.random.default_rng(42)
    y1 = rng.lognormal(1.0, 0.8, 1_500)
    y0 = rng.lognormal(0.95, 0.6, 900)
    y = np.r_[y1, y0]
    t = np.r_[np.ones(y1.size, int), np.zeros(y0.size, int)]
    return y, t, y1, y0


def test_welch_matches_scipy(continuous):
    y, t, y1, y0 = continuous
    est = diff_in_means(y, t)
    ref = stats.ttest_ind(y1, y0, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert est.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert est.abs_effect / est.abs_se == pytest.approx(ref.statistic, rel=1e-12)
    assert est.df == pytest.approx(ref.df, rel=1e-12)
    assert (est.abs_ci_low, est.abs_ci_high) == pytest.approx((ci.low, ci.high), rel=1e-10)


def test_summary_constructors_agree():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, 10_000).astype(float)
    a, b = ArmSummary.from_array(y), ArmSummary.from_binary_counts(y.size, int(y.sum()))
    c = ArmSummary.from_sums(y.size, y.sum(), (y**2).sum())
    for other in (b, c):
        assert other.mean == pytest.approx(a.mean)
        assert other.var == pytest.approx(a.var, rel=1e-10)


def test_relative_lift_delta_method_matches_bootstrap():
    rng = np.random.default_rng(7)
    n1, n0 = 200_000, 40_000
    x1, x0 = rng.binomial(n1, 0.048), rng.binomial(n0, 0.040)
    est = welch_from_summaries(ArmSummary.from_binary_counts(n1, x1),
                               ArmSummary.from_binary_counts(n0, x0))
    boot = bootstrap_from_binary_counts(n1, x1, n0, x0, n_boot=20_000, seed=1)
    boot_rel_se = (boot.rel_ci_high - boot.rel_ci_low) / (2 * 1.959964)
    assert est.rel_se == pytest.approx(boot_rel_se, rel=0.05)
    assert est.abs_se == pytest.approx(boot.abs_se, rel=0.03)


def test_binary_bootstrap_fast_path_matches_row_bootstrap():
    rng = np.random.default_rng(3)
    t = rng.binomial(1, 0.5, 6_000)
    y = rng.binomial(1, np.where(t == 1, 0.12, 0.10)).astype(float)
    fast = bootstrap_diff_in_means(y, t, n_boot=4_000, seed=0)
    # Force the generic row-resampling path by adding a tiny non-binary jitter of zero mean.
    rows = bootstrap_diff_in_means(y + 1e-9 * (np.arange(y.size) % 2 - 0.5), t, n_boot=1_000,
                                   seed=0)
    assert fast.abs_se == pytest.approx(rows.abs_se, rel=0.08)
    welch = diff_in_means(y, t)
    assert fast.abs_ci_low == pytest.approx(welch.abs_ci_low, abs=0.15 * welch.abs_se)
    assert fast.abs_ci_high == pytest.approx(welch.abs_ci_high, abs=0.15 * welch.abs_se)


def test_continuous_bootstrap_close_to_welch(continuous):
    y, t, *_ = continuous
    boot = bootstrap_diff_in_means(y, t, n_boot=2_000, seed=0)
    welch = diff_in_means(y, t)
    assert boot.abs_se == pytest.approx(welch.abs_se, rel=0.08)


def test_input_validation():
    with pytest.raises(ValueError):
        diff_in_means([1.0, 2.0, 3.0], [0, 1, 2])
    with pytest.raises(ValueError):
        ArmSummary.from_array([1.0])
