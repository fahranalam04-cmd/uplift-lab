import numpy as np
import pytest
from scipy import integrate, stats

from abkit.sequential import (
    always_valid_ci,
    log_mixture_likelihood_ratio,
    msprt_test,
    sequential_from_cumulative_sums,
)


@pytest.mark.parametrize(("tau_hat", "v", "tau2"), [(0.3, 0.04, 0.09), (-0.01, 1e-4, 4e-4)])
def test_closed_form_matches_numerical_mixture(tau_hat, v, tau2):
    sd = np.sqrt(v)
    num, _ = integrate.quad(
        lambda th: stats.norm.pdf(tau_hat, th, sd) * stats.norm.pdf(th, 0, np.sqrt(tau2)),
        -50 * np.sqrt(tau2), 50 * np.sqrt(tau2), points=[tau_hat], limit=200,
    )
    expected = np.log(num / stats.norm.pdf(tau_hat, 0, sd))
    assert log_mixture_likelihood_ratio(tau_hat, v, tau2) == pytest.approx(expected, rel=1e-6)


def test_ci_boundary_has_lambda_one_over_alpha():
    tau_hat, v, tau2, alpha = 0.2, 0.01, 0.04, 0.05
    lo, hi = always_valid_ci(np.array([tau_hat]), np.array([v]), tau2, alpha)
    for b in (lo[0], hi[0]):
        assert log_mixture_likelihood_ratio(tau_hat, v, tau2, theta0=b) == pytest.approx(
            np.log(1 / alpha), rel=1e-9
        )


def _cum_normal(rng, n_sims, n, looks, effect=0.0):
    t = rng.binomial(1, 0.5, (n_sims, n))
    y = rng.normal(size=(n_sims, n)) + effect * t
    idx = np.linspace(n // looks, n, looks).astype(int) - 1
    c = lambda a: np.cumsum(a, axis=1)[:, idx]  # noqa: E731
    return c(t), c(y * t), c(y**2 * t), c(1 - t), c(y * (1 - t)), c(y**2 * (1 - t))


def test_aa_type1_error_controlled_under_continuous_peeking():
    rng = np.random.default_rng(0)
    n1, s1, q1, n0, s0, q0 = _cum_normal(rng, 400, 4_000, 200)
    res, tau_hat, v = sequential_from_cumulative_sums(n1, s1, q1, n0, s0, q0, tau2=0.01)
    fpr = res.rejected.mean()
    assert fpr <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)
    # The naive fixed-horizon z-test applied at every look inflates the error badly.
    naive = (np.abs(tau_hat / np.sqrt(v)) > 1.959964).any(axis=1).mean()
    assert naive > 0.2


def test_duality_between_p_values_and_confidence_sequence():
    rng = np.random.default_rng(1)
    n1, s1, q1, n0, s0, q0 = _cum_normal(rng, 50, 2_000, 50, effect=0.15)
    res, _, _ = sequential_from_cumulative_sums(n1, s1, q1, n0, s0, q0, tau2=0.01)
    excludes_zero = (res.ci_low > 0) | (res.ci_high < 0)
    np.testing.assert_array_equal(res.p_values < 0.05, excludes_zero)


def test_detects_real_effect_and_p_values_monotone():
    rng = np.random.default_rng(2)
    n1, s1, q1, n0, s0, q0 = _cum_normal(rng, 100, 4_000, 100, effect=0.2)
    res, _, _ = sequential_from_cumulative_sums(n1, s1, q1, n0, s0, q0, tau2=0.04)
    assert res.rejected.mean() > 0.9
    assert (np.diff(res.p_values, axis=1) <= 0).all()
    assert (res.stop_index[res.rejected] >= 0).all()


def test_single_sequence_scalar_outputs():
    res = msprt_test(np.array([0.0, 0.5, 0.5]), np.array([1.0, 0.01, 0.005]), tau2=0.25)
    assert isinstance(res.rejected, bool) and res.rejected
    assert res.stop_index == 1
