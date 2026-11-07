"""Meta-learners for conditional average treatment effects (CATE / uplift).

Base learners are any scikit-learn compatible estimators (cloned internally), so the same
classes run with LightGBM in the Criteo pipeline and with tiny sklearn models in tests.
Outcome models may be classifiers (``predict_proba`` is used) or regressors.

* S-learner: one model mu(x, t); uplift = mu(x, 1) - mu(x, 0).
* T-learner: separate mu_1(x), mu_0(x) per arm.
* X-learner (Kunzel et al. 2019): impute individual effects with the other arm's model,
  regress them on x per arm, blend with the propensity e: tau = e * tau_0 + (1 - e) * tau_1.
* Transformed outcome (Athey & Imbens 2016): regress z = y (t - e) / (e (1 - e)) on x; valid
  for any known propensity (the class-transformation trick generalised beyond 50/50).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone

__all__ = ["SLearner", "TLearner", "TransformedOutcome", "XLearner"]


def _mean_pred(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def _arrays(X: ArrayLike, t: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_a, t_a, y_a = np.asarray(X), np.asarray(t).astype(int), np.asarray(y)
    if X_a.ndim != 2 or not (X_a.shape[0] == t_a.size == y_a.size):
        raise ValueError("X must be 2-D with one row per t / y entry")
    if not set(np.unique(t_a)).issubset({0, 1}):
        raise ValueError("t must be binary 0/1")
    return X_a, t_a, y_a


class SLearner:
    name = "s_learner"

    def __init__(self, outcome_model: Any) -> None:
        self.outcome_model = outcome_model

    def fit(self, X: ArrayLike, t: ArrayLike, y: ArrayLike) -> SLearner:
        X_a, t_a, y_a = _arrays(X, t, y)
        self.model_ = clone(self.outcome_model).fit(np.column_stack([X_a, t_a]), y_a)
        return self

    def predict_uplift(self, X: ArrayLike) -> np.ndarray:
        X_a = np.asarray(X)
        ones, zeros = np.ones(X_a.shape[0]), np.zeros(X_a.shape[0])
        return _mean_pred(self.model_, np.column_stack([X_a, ones])) - _mean_pred(
            self.model_, np.column_stack([X_a, zeros])
        )


class TLearner:
    name = "t_learner"

    def __init__(self, outcome_model: Any, control_model: Any | None = None) -> None:
        self.outcome_model = outcome_model
        self.control_model = control_model

    def fit(self, X: ArrayLike, t: ArrayLike, y: ArrayLike) -> TLearner:
        X_a, t_a, y_a = _arrays(X, t, y)
        ctrl = self.control_model if self.control_model is not None else self.outcome_model
        self.model_t_ = clone(self.outcome_model).fit(X_a[t_a == 1], y_a[t_a == 1])
        self.model_c_ = clone(ctrl).fit(X_a[t_a == 0], y_a[t_a == 0])
        return self

    def predict_uplift(self, X: ArrayLike) -> np.ndarray:
        X_a = np.asarray(X)
        return _mean_pred(self.model_t_, X_a) - _mean_pred(self.model_c_, X_a)


class XLearner:
    name = "x_learner"

    def __init__(
        self,
        outcome_model: Any,
        effect_model: Any,
        propensity: float | None = None,
        control_model: Any | None = None,
    ) -> None:
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.propensity = propensity
        self.control_model = control_model

    def fit(self, X: ArrayLike, t: ArrayLike, y: ArrayLike) -> XLearner:
        X_a, t_a, y_a = _arrays(X, t, y)
        self.e_ = float(t_a.mean()) if self.propensity is None else float(self.propensity)
        stage1 = TLearner(self.outcome_model, self.control_model).fit(X_a, t_a, y_a)
        X1, X0 = X_a[t_a == 1], X_a[t_a == 0]
        d1 = y_a[t_a == 1] - _mean_pred(stage1.model_c_, X1)
        d0 = _mean_pred(stage1.model_t_, X0) - y_a[t_a == 0]
        self.tau1_ = clone(self.effect_model).fit(X1, d1)
        self.tau0_ = clone(self.effect_model).fit(X0, d0)
        self.stage1_ = stage1
        return self

    def predict_uplift(self, X: ArrayLike) -> np.ndarray:
        X_a = np.asarray(X)
        return self.e_ * self.tau0_.predict(X_a) + (1 - self.e_) * self.tau1_.predict(X_a)


class TransformedOutcome:
    name = "transformed_outcome"

    def __init__(self, effect_model: Any, propensity: float | None = None) -> None:
        self.effect_model = effect_model
        self.propensity = propensity

    def fit(self, X: ArrayLike, t: ArrayLike, y: ArrayLike) -> TransformedOutcome:
        X_a, t_a, y_a = _arrays(X, t, y)
        e = float(t_a.mean()) if self.propensity is None else float(self.propensity)
        z = y_a * (t_a - e) / (e * (1 - e))
        self.e_ = e
        self.model_ = clone(self.effect_model).fit(X_a, z)
        return self

    def predict_uplift(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(X)), dtype=float)
