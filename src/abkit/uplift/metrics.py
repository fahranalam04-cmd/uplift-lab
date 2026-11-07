"""Uplift evaluation: uplift / Qini curves, normalized AUCs, uplift@k and incremental capture.

Definitions follow scikit-uplift 0.5 (validated in tests) and additionally accept
``sample_weight`` so that metrics can be bootstrapped with Poisson weights on large test
sets without re-sorting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "incremental_capture_at_k",
    "perfect_qini_curve",
    "perfect_uplift_curve",
    "poisson_bootstrap",
    "qini_auc_score",
    "qini_curve",
    "uplift_at_k",
    "uplift_auc_score",
    "uplift_curve",
]


def _prep(
    y: ArrayLike, t: ArrayLike, score: ArrayLike, w: ArrayLike | None
) -> tuple[np.ndarray, ...]:
    y_a, t_a, s_a = (np.asarray(a, dtype=float) for a in (y, t, score))
    if not (y_a.shape == t_a.shape == s_a.shape) or y_a.ndim != 1:
        raise ValueError("y, t and score must be 1-D arrays of equal length")
    w_a = np.ones_like(y_a) if w is None else np.asarray(w, dtype=float)
    return y_a, t_a, s_a, w_a


def _cumulatives(
    y: ArrayLike, t: ArrayLike, score: ArrayLike, w: ArrayLike | None = None
) -> tuple[np.ndarray, ...]:
    y_a, t_a, s_a, w_a = _prep(y, t, score, w)
    order = np.argsort(s_a, kind="mergesort")[::-1]
    y_a, t_a, s_a, w_a = y_a[order], t_a[order], s_a[order], w_a[order]
    thr = np.r_[np.where(np.diff(s_a))[0], s_a.size - 1]
    n_all = np.cumsum(w_a)[thr]
    n_t = np.cumsum(w_a * t_a)[thr]
    y_t = np.cumsum(w_a * y_a * t_a)[thr]
    n_c = n_all - n_t
    y_c = np.cumsum(w_a * y_a * (1 - t_a))[thr]
    return n_all, n_t, y_t, n_c, y_c


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def _prepend_origin(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or x[0] != 0 or v[0] != 0:
        x, v = np.r_[0.0, x], np.r_[0.0, v]
    return x, v


def uplift_curve(
    y: ArrayLike, t: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental outcomes when targeting the top-n users by score:
    (mean_T(top n) - mean_C(top n)) * n."""
    n_all, n_t, y_t, n_c, y_c = _cumulatives(y, t, score, sample_weight)
    return _prepend_origin(n_all, (_div(y_t, n_t) - _div(y_c, n_c)) * n_all)


def qini_curve(
    y: ArrayLike, t: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve: Y_T(n) - Y_C(n) * N_T(n) / N_C(n) (Radcliffe 2007)."""
    n_all, n_t, y_t, n_c, y_c = _cumulatives(y, t, score, sample_weight)
    return _prepend_origin(n_all, y_t - y_c * _div(n_t, n_c))


def perfect_uplift_curve(
    y: ArrayLike, t: ArrayLike, sample_weight: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    y_a, t_a = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    w = np.ones_like(y_a) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    control_responders = np.sum(w * ((y_a == 1) & (t_a == 0)))
    treated_nonresponders = np.sum(w * ((y_a == 0) & (t_a == 1)))
    summand = y_a if control_responders > treated_nonresponders else t_a
    perfect = 2 * (y_a == t_a) + summand
    return uplift_curve(y_a, t_a, perfect, sample_weight)


def perfect_qini_curve(
    y: ArrayLike,
    t: ArrayLike,
    negative_effect: bool = True,
    sample_weight: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_a, t_a = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    if negative_effect:
        return qini_curve(y_a, t_a, y_a * t_a - y_a * (1 - t_a), sample_weight)
    w = np.ones_like(y_a) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    n = w.sum()
    yt, nt = np.sum(w * y_a * t_a), np.sum(w * t_a)
    yc, nc = np.sum(w * y_a * (1 - t_a)), np.sum(w * (1 - t_a))
    total = yt - nt * yc / nc
    return np.array([0.0, total, n]), np.array([0.0, total, total])


def _normalized_area(
    curve: tuple[np.ndarray, np.ndarray], perfect: tuple[np.ndarray, np.ndarray]
) -> float:
    x_p, y_p = perfect
    baseline = np.trapezoid([0.0, y_p[-1]], [0.0, x_p[-1]])
    return float(
        (np.trapezoid(curve[1], curve[0]) - baseline) / (np.trapezoid(y_p, x_p) - baseline)
    )


def qini_auc_score(
    y: ArrayLike,
    t: ArrayLike,
    score: ArrayLike,
    negative_effect: bool = True,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Normalized Qini coefficient: area between model and random Qini curves divided by the
    same area for the perfect curve. 0 = random targeting, 1 = perfect ranking."""
    return _normalized_area(
        qini_curve(y, t, score, sample_weight),
        perfect_qini_curve(y, t, negative_effect, sample_weight),
    )


def uplift_auc_score(
    y: ArrayLike, t: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None
) -> float:
    """Normalized area under the uplift curve (0 = random, 1 = perfect)."""
    return _normalized_area(
        uplift_curve(y, t, score, sample_weight), perfect_uplift_curve(y, t, sample_weight)
    )


def _top_k(score: np.ndarray, k: float) -> np.ndarray:
    if not 0 < k < 1:
        raise ValueError("k must be a fraction in (0, 1)")
    order = np.argsort(score, kind="mergesort")[::-1]
    return order[: int(score.size * k)]


def uplift_at_k(
    y: ArrayLike,
    t: ArrayLike,
    score: ArrayLike,
    k: float = 0.3,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Observed uplift (treated rate - control rate) among the top-k fraction by score
    ('overall' strategy in scikit-uplift)."""
    y_a, t_a, s_a, w_a = _prep(y, t, score, sample_weight)
    top = _top_k(s_a, k)
    yt, tt, wt = y_a[top], t_a[top], w_a[top]
    rate_t = np.sum(wt * yt * tt) / np.sum(wt * tt)
    rate_c = np.sum(wt * yt * (1 - tt)) / np.sum(wt * (1 - tt))
    return float(rate_t - rate_c)


def incremental_capture_at_k(
    y: ArrayLike,
    t: ArrayLike,
    score: ArrayLike,
    k: float = 0.3,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Share of all incremental outcomes captured by treating only the top-k fraction.

    = [uplift@k * n_k] / [ATE * N]. Random targeting captures k in expectation.
    """
    y_a, t_a, s_a, w_a = _prep(y, t, score, sample_weight)
    top = _top_k(s_a, k)
    n_k, n = np.sum(w_a[top]), np.sum(w_a)
    ate = np.sum(w_a * y_a * t_a) / np.sum(w_a * t_a) - np.sum(w_a * y_a * (1 - t_a)) / np.sum(
        w_a * (1 - t_a)
    )
    return float(uplift_at_k(y_a, t_a, s_a, k, w_a) * n_k / (ate * n))


def poisson_bootstrap(
    metric: Callable[..., float],
    y: ArrayLike,
    t: ArrayLike,
    score: ArrayLike,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Poisson(1)-weight bootstrap of ``metric(y, t, score, sample_weight=w)``.

    Approximates the multinomial bootstrap for large n while keeping arrays in place.
    """
    rng = np.random.default_rng(seed)
    n = np.asarray(y).size
    vals = np.array(
        [metric(y, t, score, sample_weight=rng.poisson(1.0, n)) for _ in range(n_boot)]
    )
    return {
        "point": float(metric(y, t, score)),
        "se": float(vals.std(ddof=1)),
        "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
        "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_boot": n_boot,
    }
