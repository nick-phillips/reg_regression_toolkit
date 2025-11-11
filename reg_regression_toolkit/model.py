"""Model fitting and cross-validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldResult:
    """Results captured for a single cross-validation fold."""

    fold_index: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    X_test: np.ndarray
    test_indices: np.ndarray
    log_loss_value: float
    confusion: np.ndarray
    ids: Optional[np.ndarray] = None


@dataclass
class CrossValidationResult:
    """Aggregate object storing cross-validated outputs."""

    folds: List[FoldResult]
    models: List[ClassifierMixin]
    scalers: List[BaseEstimator]
    classes_: np.ndarray
    feature_names: List[str]

    def _concat(self, attr: str) -> np.ndarray:
        arrays = [getattr(fold, attr) for fold in self.folds if getattr(fold, attr) is not None]
        if not arrays:
            return np.array([])
        return np.concatenate(arrays, axis=0)

    @property
    def y_true(self) -> np.ndarray:
        return self._concat("y_true")

    @property
    def y_pred(self) -> np.ndarray:
        return self._concat("y_pred")

    @property
    def y_proba(self) -> np.ndarray:
        return self._concat("y_proba")

    @property
    def ids(self) -> Optional[np.ndarray]:
        arrays = [fold.ids for fold in self.folds if fold.ids is not None]
        if not arrays:
            return None
        return np.concatenate(arrays, axis=0)

    @property
    def X_tests(self) -> List[np.ndarray]:
        return [fold.X_test for fold in self.folds]

    @property
    def mean_log_loss(self) -> float:
        losses = [fold.log_loss_value for fold in self.folds]
        return float(np.mean(losses))


def _resolve_logistic_kwargs(
    logistic_kwargs: Optional[Dict[str, Any]],
    y_train: np.ndarray,
) -> Dict[str, Any]:
    kwargs = dict(logistic_kwargs or {})
    kwargs.setdefault("Cs", np.logspace(-4, 4, 20))
    kwargs.setdefault("cv", 5)
    kwargs.setdefault("scoring", "neg_log_loss")
    kwargs.setdefault("max_iter", 5000)
    kwargs.setdefault("class_weight", "balanced")
    kwargs.setdefault("solver", "saga")
    kwargs.setdefault("n_jobs", None)
    kwargs.setdefault("random_state", 42)
    kwargs.setdefault("penalty", "l1")

    penalty = kwargs.get("penalty")
    if penalty == "elasticnet":
        kwargs.setdefault("l1_ratios", [0.5])
    else:
        kwargs.pop("l1_ratios", None)

    return kwargs


def fit_logistic_regression_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    logistic_kwargs: Optional[Dict[str, Any]] = None,
) -> LogisticRegressionCV:
    """Fit an L1-regularized logistic regression model with cross-validation."""

    kwargs = _resolve_logistic_kwargs(logistic_kwargs, y_train)
    model = LogisticRegressionCV(**kwargs)
    model.fit(X_train, y_train)
    return model


def cross_validate_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ids: Optional[np.ndarray] = None,
    cv_splits: int = 5,
    scaler: Optional[BaseEstimator] = None,
    logistic_kwargs: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    shuffle: bool = True,
    feature_names: Optional[Iterable[str]] = None,
) -> CrossValidationResult:
    """Perform stratified cross-validation for logistic regression."""

    if scaler is None:
        scaler = StandardScaler()

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=shuffle, random_state=random_state)

    folds: List[FoldResult] = []
    models: List[ClassifierMixin] = []
    scalers: List[BaseEstimator] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        scaler_fold = clone(scaler)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler_fold.fit_transform(X_train)
        X_test_scaled = scaler_fold.transform(X_test)

        model = fit_logistic_regression_cv(
            X_train_scaled,
            y_train,
            logistic_kwargs=logistic_kwargs,
        )

        y_proba = model.predict_proba(X_test_scaled)
        y_pred = model.predict(X_test_scaled)

        fold_loss = log_loss(y_test, y_proba, labels=model.classes_)
        fold_confusion = confusion_matrix(y_test, y_pred, labels=model.classes_)

        fold_ids = ids[test_idx] if ids is not None else None

        folds.append(
            FoldResult(
                fold_index=fold_idx,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                X_test=X_test_scaled,
                test_indices=test_idx,
                log_loss_value=fold_loss,
                confusion=fold_confusion,
                ids=fold_ids,
            )
        )
        models.append(model)
        scalers.append(scaler_fold)

    classes_ = models[0].classes_ if models else np.unique(y)
    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(X.shape[1])]
    return CrossValidationResult(
        folds=folds,
        models=models,
        scalers=scalers,
        classes_=classes_,
        feature_names=list(feature_names),
    )
