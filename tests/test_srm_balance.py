import numpy as np
import pytest
from scipy import stats

from abkit.balance import (
    auc_standard_error,
    standardized_mean_differences,
    treatment_predictability_auc,
)
from abkit.srm import srm_detectable_share_deviation, srm_test


def test_srm_matches_scipy_chisquare():
    counts, ratios = [10_250, 1_750], [0.85, 0.15]
    res = srm_test(counts, ratios)
    ref = stats.chisquare(counts, np.array(ratios) * sum(counts))
    assert res.chi2 == pytest.approx(ref.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-12)
    assert res.dof == 1


def test_srm_two_arm_chi2_equals_z_squared():
    n, p, n1 = 100_000, 0.85, 85_400
    z = (n1 / n - p) / np.sqrt(p * (1 - p) / n)
    assert srm_test([n1, n - n1], [p, 1 - p]).chi2 == pytest.approx(z**2, rel=1e-10)


def test_srm_flags_and_passes():
    assert not srm_test([8_500, 1_500], [85, 15]).mismatch  # exact design, ratios normalised
    assert srm_test([8_000, 2_000], [0.85, 0.15]).mismatch


def test_srm_detectable_deviation_is_the_alarm_boundary():
    n, p, thr = 1_000_000, 0.85, 0.001
    dev = srm_detectable_share_deviation(n, p, thr)
    n1 = (p + dev) * n
    assert srm_test([n1, n - n1], [p, 1 - p], thr).p_value == pytest.approx(thr, rel=1e-6)


def test_srm_validates_input():
    with pytest.raises(ValueError):
        srm_test([1, 2, 3], [0.5, 0.5])


def test_smd_closed_form():
    X = np.array([[0.0], [2.0], [1.0], [3.0]])
    t = np.array([0, 0, 1, 1])
    # means 1 vs 2, both variances 2 -> smd = 1 / sqrt(2)
    smd = standardized_mean_differences(X, t, ["a"])["smd"].iloc[0]
    assert smd == pytest.approx(1 / np.sqrt(2))


def test_treatment_auc_near_half_under_randomization():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4_000, 3))
    t = rng.binomial(1, 0.85, 4_000)
    res = treatment_predictability_auc(X, t, n_splits=3)
    assert res["ci_low"] < 0.5 < res["ci_high"] or abs(res["auc"] - 0.5) < 0.04


def test_treatment_auc_detects_broken_randomization():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4_000, 3))
    t = (X[:, 0] + rng.normal(size=4_000) > 0).astype(int)  # assignment depends on a feature
    assert treatment_predictability_auc(X, t)["auc"] > 0.7


def test_hanley_mcneil_se_at_half():
    # At AUC = 0.5 the HM variance reduces to (n_pos + n_neg + 1) / (12 n_pos n_neg).
    n1, n0 = 300, 700
    expected = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
    assert auc_standard_error(0.5, n1, n0) == pytest.approx(expected, rel=1e-3)
