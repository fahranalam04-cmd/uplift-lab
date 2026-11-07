import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower

from abkit.power import minimum_detectable_effect, power_at, required_sample_size


@pytest.mark.parametrize("frac", [0.5, 0.85])
def test_sample_size_matches_statsmodels_for_means(frac):
    sd, delta = 2.0, 0.1
    n1_ref = NormalIndPower().solve_power(
        effect_size=delta / sd, alpha=0.05, power=0.8, ratio=(1 - frac) / frac
    )
    ss = required_sample_size(10.0, delta, sd=sd, treat_fraction=frac)
    assert ss.n_total_exact * frac == pytest.approx(n1_ref, rel=1e-6)
    assert ss.n_treatment == int(np.ceil(ss.n_total_exact * frac))


def test_power_matches_statsmodels():
    sd, delta, n1, n0 = 1.0, 0.05, 6_000, 4_000
    ref = NormalIndPower().power(effect_size=delta / sd, nobs1=n1, alpha=0.05, ratio=n0 / n1)
    ours = power_at(n1 + n0, 0.0, delta, sd=sd, treat_fraction=n1 / (n1 + n0))
    assert ours == pytest.approx(ref, rel=1e-10)


def test_binary_sample_size_closed_form():
    # Unpooled normal approximation, 50/50: n per arm = (z_a + z_b)^2 (p0 q0 + p1 q1) / d^2
    p0, p1 = 0.10, 0.12
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    n_arm = z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2
    ss = required_sample_size(p0, p1 - p0)
    # Exact two-tailed solve differs from the one-tail closed form by a negligible amount.
    assert ss.n_treatment == pytest.approx(n_arm, rel=1e-3)


def test_mde_inverts_power():
    n, base = 500_000, 0.047
    mde = minimum_detectable_effect(n, base, treat_fraction=0.85)
    assert power_at(n, base, mde, treat_fraction=0.85) == pytest.approx(0.8, abs=1e-8)
    rel = minimum_detectable_effect(n, base, treat_fraction=0.85, relative=True)
    assert rel == pytest.approx(mde / base)


def test_sample_size_and_mde_roundtrip():
    ss = required_sample_size(0.05, 0.10, relative=True, treat_fraction=0.7)
    mde = minimum_detectable_effect(ss.n_total_exact, 0.05, treat_fraction=0.7, relative=True)
    assert mde == pytest.approx(0.10, rel=1e-6)


def test_variance_reduction_scales_sample_size():
    base = required_sample_size(10.0, 0.1, sd=2.0)
    adj = required_sample_size(10.0, 0.1, sd=2.0, variance_reduction=0.3)
    assert adj.n_total_exact == pytest.approx(0.7 * base.n_total_exact, rel=1e-6)


def test_one_sided_needs_fewer_users():
    two = required_sample_size(0.1, 0.01)
    one = required_sample_size(0.1, 0.01, two_sided=False)
    assert one.n_total < two.n_total


def test_validation():
    with pytest.raises(ValueError):
        required_sample_size(0.1, 0.0)
    with pytest.raises(ValueError):
        power_at(1_000, 0.1, 0.01, treat_fraction=1.0)
