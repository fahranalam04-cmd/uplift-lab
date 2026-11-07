import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from abkit.inference import diff_in_means
from abkit.variance_reduction import (
    MomentAccumulator,
    check_pretreatment,
    crossfit_predictions,
    cupac,
    cuped,
    equivalent_sample_size_saving,
    lin_from_moments,
    regression_adjusted,
)


def _data(n=3_000, seed=0, het=0.0, p_treat=0.6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) + np.array([5.0, -2.0, 0.0])  # non-zero means
    t = rng.binomial(1, p_treat, n)
    beta = np.array([1.0, 0.5, -0.8])
    y = 2.0 + X @ beta + t * (0.3 + het * X[:, 0]) + rng.normal(size=n)
    return X, t, y


def _lin_ols(X, t, y):
    Xc = X - X.mean(axis=0)
    design = sm.add_constant(np.column_stack([t, Xc, t[:, None] * Xc]))
    return sm.OLS(y, design)


def test_lin_point_estimate_equals_interacted_ols():
    X, t, y = _data(het=0.4)
    est = regression_adjusted(y, t, X)
    ols = _lin_ols(X, t, y).fit()
    assert est.abs_effect == pytest.approx(ols.params[1], rel=1e-9)


def test_lin_se_matches_hc2_without_slope_heterogeneity():
    X, t, y = _data(het=0.0)
    est = regression_adjusted(y, t, X)
    hc2 = _lin_ols(X, t, y).fit(cov_type="HC2").bse[1]
    assert est.abs_se == pytest.approx(hc2, rel=0.02)


def test_no_covariates_reduces_to_welch():
    X, t, y = _data()
    acc = MomentAccumulator(["x0", "y"]).update(np.column_stack([X[:, 0], y]), t)
    est, welch = lin_from_moments(acc, "y", ()), diff_in_means(y, t)
    assert est.abs_effect == pytest.approx(welch.abs_effect, rel=1e-10)
    assert est.abs_se == pytest.approx(welch.abs_se, rel=1e-10)
    assert est.rel_se == pytest.approx(welch.rel_se, rel=1e-10)


def test_streaming_accumulation_equals_single_batch():
    X, t, y = _data(n=5_000)
    Z = np.column_stack([X, y])
    whole = MomentAccumulator(["a", "b", "c", "y"]).update(Z, t)
    chunked = MomentAccumulator(["a", "b", "c", "y"])
    for idx in np.array_split(np.arange(Z.shape[0]), 7):
        chunked.update(Z[idx], t[idx])
    e1 = lin_from_moments(whole, "y", ["a", "b", "c"])
    e2 = lin_from_moments(chunked, "y", ["a", "b", "c"])
    assert e1.abs_effect == pytest.approx(e2.abs_effect, rel=1e-10)
    assert e1.abs_se == pytest.approx(e2.abs_se, rel=1e-8)


def test_adjusted_ci_coverage_and_shrinkage():
    """Over 300 simulated experiments with heterogeneous slopes, both raw and Lin 95% CIs
    cover the true ATE ~95% of the time, and adjustment shrinks the SE."""
    rng = np.random.default_rng(11)
    cover_raw = cover_adj = 0
    ratio = []
    for _ in range(300):
        X = rng.normal(size=(400, 2))
        t = rng.binomial(1, 0.5, 400)
        y = X @ [1.0, 1.0] + t * (0.5 + 0.5 * X[:, 0]) + rng.normal(size=400)
        raw, adj = diff_in_means(y, t), regression_adjusted(y, t, X)
        cover_raw += raw.abs_ci_low <= 0.5 <= raw.abs_ci_high
        cover_adj += adj.abs_ci_low <= 0.5 <= adj.abs_ci_high
        ratio.append(adj.abs_se / raw.abs_se)
    assert 0.92 <= cover_raw / 300 <= 0.98
    assert 0.92 <= cover_adj / 300 <= 0.98
    assert np.mean(ratio) < 0.75


def test_cuped_variance_reduction_is_rho_squared():
    rng = np.random.default_rng(5)
    n = 20_000
    x = rng.normal(10, 2, n)
    t = rng.binomial(1, 0.5, n)
    y = 0.8 * x + 0.2 * t + rng.normal(0, 1.5, n)
    res = cuped(y, t, x)
    rho = np.corrcoef(x, y)[0, 1]
    assert res.variance_reduction == pytest.approx(rho**2, abs=0.01)
    assert res.estimate.abs_ci_low < 0.2 < res.estimate.abs_ci_high
    assert res.estimate.abs_se < diff_in_means(y, t).abs_se


def test_leakage_guard_rejects_post_treatment_columns():
    with pytest.raises(ValueError, match="exposure"):
        check_pretreatment(["f0", "exposure"])
    acc = MomentAccumulator(["exposure", "visit"]).update(np.ones((4, 2)) * [[0], [1], [0], [1]],
                                                          [0, 0, 1, 1])
    with pytest.raises(ValueError):
        lin_from_moments(acc, "visit", ["exposure"])


def test_crossfit_prediction_never_sees_own_outcome():
    """Perturbing unit i's outcome must not change unit i's out-of-fold prediction."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)
    folds = np.arange(200) % 2
    base = crossfit_predictions(X, y, LinearRegression, folds=folds)
    y2 = y.copy()
    y2[0] += 1_000.0
    moved = crossfit_predictions(X, y2, LinearRegression, folds=folds)
    assert moved[0] == pytest.approx(base[0], rel=1e-12)
    assert not np.allclose(moved[1], base[1])  # other-fold units do see it


def test_cupac_beats_linear_adjustment_on_nonlinear_signal():
    rng = np.random.default_rng(2)
    n = 6_000
    X = rng.uniform(-2, 2, size=(n, 2))
    t = rng.binomial(1, 0.5, n)
    y = np.sin(2 * X[:, 0]) * 3 + X[:, 1] ** 2 + 0.1 * t + rng.normal(0, 0.5, n)
    raw = diff_in_means(y, t)
    lin = regression_adjusted(y, t, X)
    est, _ = cupac(y, t, X, lambda: HistGradientBoostingRegressor(max_iter=100), seed=0)
    assert est.abs_se < lin.abs_se < raw.abs_se
    assert est.abs_ci_low < 0.1 < est.abs_ci_high


def test_equivalent_sample_size_saving():
    out = equivalent_sample_size_saving(1.0, 0.9)
    assert out["ci_width_reduction"] == pytest.approx(0.1)
    assert out["sample_size_saving"] == pytest.approx(0.19)
    assert out["effective_sample_multiplier"] == pytest.approx(1 / 0.81)
