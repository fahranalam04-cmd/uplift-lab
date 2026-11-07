import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from abkit.doubly_robust import aipw, aipw_scores, hausman_randomization_test
from abkit.inference import diff_in_means
from abkit.variance_reduction import regression_adjusted


def _pooled_tests(n=8_000, seed=0):
    """Two 'tests' pooled: test B has a higher base rate AND a higher treatment share,
    so the naive difference in means is confounded (Simpson-style)."""
    rng = np.random.default_rng(seed)
    g = rng.binomial(1, 0.5, n)  # test id, observable through x
    x = g + rng.normal(0, 0.3, n)
    e = np.where(g == 1, 0.9, 0.6)
    t = rng.binomial(1, e)
    y = 1.0 + 2.0 * g + 0.5 * t + rng.normal(0, 1, n)  # true ATE = 0.5
    return x, t, y, e, g


def test_aipw_removes_pooling_confounding_when_test_id_is_observed():
    x, t, y, _, g = _pooled_tests()
    raw = diff_in_means(y, t)
    est, _ = aipw(y, t, g[:, None].astype(float), LinearRegression,
                  lambda: LogisticRegression(), seed=0)
    assert raw.abs_ci_low > 0.5  # naive estimate is biased upwards
    assert est.abs_ci_low < 0.5 < est.abs_ci_high


def test_noisy_proxy_leaves_residual_confounding():
    """Adjusting for a noisy proxy of the test id only removes part of the bias."""
    x, t, y, _, _ = _pooled_tests(n=40_000)
    raw = diff_in_means(y, t).abs_effect
    est, _ = aipw(y, t, x, LinearRegression, lambda: LogisticRegression(), seed=0)
    assert 0.5 < est.abs_effect < raw


def test_aipw_scores_with_true_nuisances_are_unbiased():
    x, t, y, e, _ = _pooled_tests(n=200_000, seed=1)
    g_hat = (x > 0.5).astype(float)  # imperfect outcome model: double robustness via e
    psi1, psi0 = aipw_scores(y, t, 1.5 + 2 * g_hat, 1.0 + 2 * g_hat, e)
    assert (psi1 - psi0).mean() == pytest.approx(0.5, abs=0.02)


def test_aipw_without_propensity_matches_lin_in_clean_rct():
    rng = np.random.default_rng(3)
    n = 20_000
    x = rng.normal(size=n)
    t = rng.binomial(1, 0.5, n)
    y = x + 0.2 * t + rng.normal(size=n)
    est, _ = aipw(y, t, x, LinearRegression, None, seed=0)
    lin = regression_adjusted(y, t, x)
    assert est.abs_effect == pytest.approx(lin.abs_effect, abs=0.3 * lin.abs_se)
    assert est.abs_se == pytest.approx(lin.abs_se, rel=0.05)


def test_hausman_flags_confounding_but_not_clean_randomization():
    x, t, y, _, _ = _pooled_tests()
    bad = hausman_randomization_test(diff_in_means(y, t), regression_adjusted(y, t, x))
    assert abs(bad["z"]) > 4
    rng = np.random.default_rng(4)
    rejections = 0
    for _ in range(200):
        xc = rng.normal(size=1_000)
        tc = rng.binomial(1, 0.5, 1_000)
        yc = xc + 0.3 * tc + rng.normal(size=1_000)
        h = hausman_randomization_test(diff_in_means(yc, tc), regression_adjusted(yc, tc, xc))
        rejections += h["p_value"] < 0.05
    assert rejections / 200 < 0.10
