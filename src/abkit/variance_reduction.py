"""Variance reduction: CUPED, Lin (2013) regression adjustment and CUPAC with cross-fitting.

All estimators reduce to per-arm first and second moments, accumulated by
:class:`MomentAccumulator`. That makes the 14M-row Criteo readout a single streaming pass
over Parquet while the identical code is unit-tested against statsmodels on small arrays.

Variance formula (super-population, influence-function / delta method)::

    Var(tau_hat) = s2_e1 / n1 + s2_e0 / n0 + (b1 - b0)' S_x (b1 - b0) / N

where ``s2_e`` are per-arm residual variances of y on the covariates, ``b`` the per-arm
slopes and ``S_x`` the pooled covariate covariance. With no covariates this is exactly the
Welch variance, so raw and adjusted CIs are directly comparable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold

from .inference import EffectEstimate, effect_from_components

__all__ = [
    "POST_TREATMENT_COLUMNS",
    "MomentAccumulator",
    "check_pretreatment",
    "crossfit_predictions",
    "cuped",
    "cupac",
    "equivalent_sample_size_saving",
    "lin_from_moments",
    "regression_adjusted",
]

#: Columns that are measured after (or are) the assignment and must never be covariates.
POST_TREATMENT_COLUMNS = frozenset({"treatment", "exposure", "visit", "conversion"})


def check_pretreatment(
    covariates: Iterable[str], post_treatment: Iterable[str] = POST_TREATMENT_COLUMNS
) -> None:
    """Leakage guard: raise if any covariate is a known post-treatment variable."""
    bad = sorted(set(covariates) & set(post_treatment))
    if bad:
        raise ValueError(f"post-treatment columns cannot be used for adjustment: {bad}")


@dataclass
class _Arm:
    n: int = 0
    s: np.ndarray | None = None
    ss: np.ndarray | None = None


@dataclass
class MomentAccumulator:
    """Streaming per-arm moments of a fixed set of named columns.

    Columns are shifted by the first batch's mean before accumulating sums of squares, which
    keeps the one-pass covariance numerically stable for features with large means.
    """

    columns: Sequence[str]
    arms: dict[int, _Arm] = field(default_factory=lambda: {0: _Arm(), 1: _Arm()})
    shift: np.ndarray | None = None

    def update(self, Z: ArrayLike, t: ArrayLike) -> MomentAccumulator:
        Z_arr = np.asarray(Z, dtype=np.float64)
        t_arr = np.asarray(t)
        if Z_arr.ndim != 2 or Z_arr.shape[1] != len(self.columns):
            raise ValueError(f"Z must be (n, {len(self.columns)})")
        if self.shift is None:
            self.shift = Z_arr.mean(axis=0)
        Zc = Z_arr - self.shift
        for a in (0, 1):
            block = Zc[t_arr == a]
            arm = self.arms[a]
            if arm.s is None:
                k = len(self.columns)
                arm.s, arm.ss = np.zeros(k), np.zeros((k, k))
            arm.n += block.shape[0]
            arm.s += block.sum(axis=0)
            arm.ss += block.T @ block
        return self

    def arm_moments(self, arm: int) -> tuple[int, np.ndarray, np.ndarray]:
        """(n, mean vector, ddof=1 covariance matrix) for one arm."""
        a = self.arms[arm]
        if a.s is None or a.n < 2 or self.shift is None:
            raise ValueError(f"arm {arm} has fewer than 2 observations")
        mean_c = a.s / a.n
        cov = (a.ss - a.n * np.outer(mean_c, mean_c)) / (a.n - 1)
        return a.n, mean_c + self.shift, cov

    def index(self, names: Sequence[str]) -> list[int]:
        cols = list(self.columns)
        return [cols.index(n) for n in names]


def lin_from_moments(
    acc: MomentAccumulator,
    outcome: str,
    covariates: Sequence[str] = (),
    alpha: float = 0.05,
    method: str | None = None,
) -> EffectEstimate:
    """Lin (2013) fully-interacted regression adjustment from accumulated moments.

    Equivalent to the coefficient on T in OLS ``y ~ T + Xc + T:Xc`` with covariates centered
    at the pooled mean. With ``covariates=()`` this is the plain difference in means.
    """
    check_pretreatment(covariates)
    yi = acc.index([outcome])[0]
    xi = acc.index(covariates)
    n1, mean1, cov1 = acc.arm_moments(1)
    n0, mean0, cov0 = acc.arm_moments(0)
    N = n1 + n0
    p = len(xi)
    xbar = (n1 * mean1[xi] + n0 * mean0[xi]) / N if p else np.zeros(0)

    def fit(n: int, mean: np.ndarray, cov: np.ndarray) -> tuple[float, np.ndarray, float]:
        if p == 0:
            return float(mean[yi]), np.zeros(0), float(cov[yi, yi])
        sxx, sxy = cov[np.ix_(xi, xi)], cov[xi, yi]
        beta = np.linalg.solve(sxx, sxy)
        resid_var = (cov[yi, yi] - sxy @ beta) * (n - 1) / (n - 1 - p)
        mu = float(mean[yi] - beta @ (mean[xi] - xbar))
        return mu, beta, float(max(resid_var, 0.0))

    mu1, b1, s2e1 = fit(n1, mean1, cov1)
    mu0, b0, s2e0 = fit(n0, mean0, cov0)
    if p:
        # Pooled covariate covariance (total, not within-arm).
        d1, d0 = mean1[xi] - xbar, mean0[xi] - xbar
        sx = ((n1 - 1) * cov1[np.ix_(xi, xi)] + (n0 - 1) * cov0[np.ix_(xi, xi)]
              + n1 * np.outer(d1, d1) + n0 * np.outer(d0, d0)) / (N - 1)
        db = b1 - b0
        het = float(db @ sx @ db) / N
        var_mu0 = s2e0 / n0 + float(b0 @ sx @ b0) / N
        cov_tau_mu0 = float(db @ sx @ b0) / N - s2e0 / n0
    else:
        het, var_mu0, cov_tau_mu0 = 0.0, s2e0 / n0, -s2e0 / n0
    return effect_from_components(
        method=method or ("diff_in_means" if p == 0 else f"lin_{p}cov"),
        n_treatment=n1,
        n_control=n0,
        mu1=mu1,
        mu0=mu0,
        var_tau=s2e1 / n1 + s2e0 / n0 + het,
        var_mu0=var_mu0,
        cov_tau_mu0=cov_tau_mu0,
        alpha=alpha,
    )


def _as_2d(X: ArrayLike) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    return arr[:, None] if arr.ndim == 1 else arr


def regression_adjusted(
    y: ArrayLike,
    t: ArrayLike,
    X: ArrayLike,
    alpha: float = 0.05,
    names: Sequence[str] | None = None,
) -> EffectEstimate:
    """Lin regression adjustment on in-memory arrays."""
    X_arr = _as_2d(X)
    cols = list(names) if names is not None else [f"x{i}" for i in range(X_arr.shape[1])]
    acc = MomentAccumulator([*cols, "__y__"])
    acc.update(np.column_stack([X_arr, np.asarray(y, dtype=float)]), t)
    return lin_from_moments(acc, "__y__", cols, alpha=alpha)


@dataclass(frozen=True)
class CupedResult:
    estimate: EffectEstimate
    theta: float
    variance_reduction: float


def cuped(y: ArrayLike, t: ArrayLike, x: ArrayLike, alpha: float = 0.05) -> CupedResult:
    """Classic CUPED (Deng et al. 2013): y_cv = y - theta (x - mean(x)), theta = cov(y,x)/var(x).

    theta is estimated on the pooled sample ignoring assignment (valid because x is
    pre-treatment and independent of T). The point estimate is then a Welch difference in
    means of y_cv. Returns the realised variance reduction 1 - var(y_cv) / var(y).
    """
    from .inference import diff_in_means

    y_arr, x_arr = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    c = np.cov(y_arr, x_arr, ddof=1)
    theta = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0
    y_cv = y_arr - theta * (x_arr - x_arr.mean())
    est = diff_in_means(y_cv, t, alpha)
    est = replace(est, method="cuped")
    return CupedResult(est, float(theta), float(1 - y_cv.var(ddof=1) / y_arr.var(ddof=1)))


class _Model(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def crossfit_predictions(
    X: ArrayLike,
    y: ArrayLike,
    make_model: Callable[[], _Model],
    n_folds: int = 2,
    seed: int = 0,
    folds: ArrayLike | None = None,
) -> np.ndarray:
    """Out-of-fold predictions of E[y | X] for use as a CUPAC covariate.

    Each unit's prediction comes from a model that never saw that unit's outcome, so the
    covariate is a fixed function of pre-treatment X with respect to that unit and cannot
    leak its own (post-treatment) outcome into the adjustment. Treatment is deliberately not
    an input: the model predicts the outcome from pre-treatment features only.
    """
    X_arr, y_arr = _as_2d(X), np.asarray(y)
    if folds is None:
        fold_ids = np.empty(X_arr.shape[0], dtype=int)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (_, idx) in enumerate(kf.split(X_arr)):
            fold_ids[idx] = k
    else:
        fold_ids = np.asarray(folds)
    pred = np.empty(X_arr.shape[0], dtype=float)
    for k in np.unique(fold_ids):
        held = fold_ids == k
        model = make_model()
        model.fit(X_arr[~held], y_arr[~held])
        pred[held] = _predict(model, X_arr[held])
    return pred


def cupac(
    y: ArrayLike,
    t: ArrayLike,
    X: ArrayLike,
    make_model: Callable[[], _Model],
    n_folds: int = 2,
    seed: int = 0,
    include_linear: bool = False,
    alpha: float = 0.05,
) -> tuple[EffectEstimate, np.ndarray]:
    """CUPAC (Tang et al. 2020): regression adjustment on a cross-fitted ML prediction of y.

    ``include_linear`` also adjusts linearly for the raw features. Returns the estimate and
    the out-of-fold predictions.
    """
    X_arr = _as_2d(X)
    pred = crossfit_predictions(X_arr, y, make_model, n_folds=n_folds, seed=seed)
    cov = np.column_stack([pred, X_arr]) if include_linear else pred[:, None]
    names = ["cupac_pred"] + ([f"x{i}" for i in range(X_arr.shape[1])] if include_linear else [])
    est = regression_adjusted(y, t, cov, alpha=alpha, names=names)
    return replace(est, method="cupac"), pred


def equivalent_sample_size_saving(se_raw: float, se_adjusted: float) -> dict[str, float]:
    """Translate an SE reduction into CI-width and sample-size terms.

    Variance scales as 1/n, so matching the adjusted precision without adjustment needs
    (se_raw / se_adj)^2 times as many users; equivalently adjustment saves
    1 - (se_adj / se_raw)^2 of the sample at equal precision.
    """
    ratio = se_adjusted / se_raw
    return {
        "se_ratio": float(ratio),
        "ci_width_reduction": float(1 - ratio),
        "variance_reduction": float(1 - ratio**2),
        "sample_size_saving": float(1 - ratio**2),
        "effective_sample_multiplier": float(1 / ratio**2),
    }
