import warnings

import numpy as np
import pytest
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from abkit.uplift import (
    SLearner,
    TLearner,
    TransformedOutcome,
    XLearner,
    incremental_capture_at_k,
    poisson_bootstrap,
    qini_auc_score,
    qini_curve,
    uplift_at_k,
    uplift_auc_score,
    uplift_curve,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import sklift.metrics as skm


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    n = 3_000
    x = rng.normal(size=n)
    t = rng.binomial(1, 0.7, n)
    y = rng.binomial(1, 0.1 + 0.08 * t * (x > 0))
    score = np.round(x + rng.normal(0, 0.5, n), 1)  # rounding creates ties
    return y, t, score


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_curves_and_scores_match_scikit_uplift(scored):
    y, t, s = scored
    for ours, ref in ((qini_curve, skm.qini_curve), (uplift_curve, skm.uplift_curve)):
        xo, vo = ours(y, t, s)
        xr, vr = ref(y, s, t)
        np.testing.assert_allclose(xo, xr)
        np.testing.assert_allclose(vo, vr, rtol=1e-10, atol=1e-10)
    for neg in (True, False):
        assert qini_auc_score(y, t, s, negative_effect=neg) == pytest.approx(
            skm.qini_auc_score(y, s, t, negative_effect=neg), rel=1e-10
        )
    assert uplift_auc_score(y, t, s) == pytest.approx(skm.uplift_auc_score(y, s, t), rel=1e-10)
    for k in (0.1, 0.3):
        assert uplift_at_k(y, t, s, k) == pytest.approx(
            skm.uplift_at_k(y, s, t, strategy="overall", k=k), rel=1e-10
        )


def test_integer_weights_equal_row_duplication(scored):
    y, t, s = scored
    w = np.random.default_rng(1).integers(0, 3, y.size)
    rep = np.repeat(np.arange(y.size), w)
    # jitter-free duplication keeps ties identical, so weighted == duplicated exactly
    assert qini_auc_score(y, t, s, sample_weight=w) == pytest.approx(
        qini_auc_score(y[rep], t[rep], s[rep]), rel=1e-10
    )
    assert uplift_auc_score(y, t, s, sample_weight=w) == pytest.approx(
        uplift_auc_score(y[rep], t[rep], s[rep]), rel=1e-10
    )


def test_incremental_capture_bounds(scored):
    y, t, s = scored
    rng = np.random.default_rng(2)
    rand = np.mean([incremental_capture_at_k(y, t, rng.random(y.size), 0.3) for _ in range(30)])
    assert rand == pytest.approx(0.3, abs=0.15)
    assert incremental_capture_at_k(y, t, s, 0.5) > 0.7  # effect lives in x > 0 half


def test_poisson_bootstrap_interval(scored):
    y, t, s = scored
    out = poisson_bootstrap(qini_auc_score, y, t, s, n_boot=50, seed=0)
    assert out["ci_low"] < out["point"] < out["ci_high"]


def _hte_data(n=6_000, seed=0, p=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.binomial(1, p, n)
    tau = 1.0 * X[:, 0] + 0.5
    y = X[:, 1] + X[:, 2] ** 2 + t * tau + rng.normal(0, 1, n)
    return X, t, y, tau


@pytest.mark.parametrize("p", [0.5, 0.85])
def test_meta_learners_recover_effect_ranking(p):
    X, t, y, tau = _hte_data(p=p)
    gbm = HistGradientBoostingRegressor(max_iter=80, max_depth=3)
    learners = [SLearner(gbm), TLearner(gbm), XLearner(gbm, gbm), TransformedOutcome(gbm)]
    for learner in learners:
        pred = learner.fit(X, t, y).predict_uplift(X)
        rho = stats.spearmanr(pred, tau).statistic
        floor = 0.3 if learner.name == "transformed_outcome" else 0.7
        assert rho > floor, (learner.name, rho)


def test_learner_input_validation():
    with pytest.raises(ValueError):
        TLearner(HistGradientBoostingRegressor()).fit(np.zeros((4, 1)), [0, 1, 2, 1], [0, 1, 0, 1])
