"""Evaluation helpers for cross-validated workflows."""

from __future__ import annotations

from typing import Optional

import numpy as np

import pandas as pd
from sklearn.metrics import auc, classification_report, roc_curve

from .model import CrossValidationResult


def aggregate_confusion_matrix(result: CrossValidationResult) -> np.ndarray:
    """Sum the confusion matrices across folds."""

    matrices = [fold.confusion for fold in result.folds]
    return np.sum(matrices, axis=0)


def predictions_dataframe(result: CrossValidationResult) -> pd.DataFrame:
    """Return a tidy DataFrame mapping ids to predicted probabilities."""

    data = {
        "y_true": result.y_true,
        "y_pred": result.y_pred,
    }

    for idx, cls in enumerate(result.classes_):
        data[f"proba_{cls}"] = result.y_proba[:, idx]

    if result.ids is not None:
        data["id"] = result.ids

    return pd.DataFrame(data)


def fold_metrics_dataframe(result: CrossValidationResult) -> pd.DataFrame:
    """Collect per-fold metrics into a DataFrame."""

    rows = []
    for fold in result.folds:
        rows.append(
            {
                "fold": fold.fold_index,
                "log_loss": fold.log_loss_value,
            }
        )
    return pd.DataFrame(rows)


def roc_curve_from_result(
    result: CrossValidationResult,
    *,
    positive_label: Optional[int] = None,
):
    """Compute ROC curve data for binary classification results."""

    classes_ = result.classes_
    if classes_.shape[0] != 2:
        raise ValueError("ROC computation requires exactly two classes.")

    if positive_label is None:
        positive_index = 1
    else:
        if positive_label not in classes_:
            raise ValueError(f"Positive label {positive_label} not found in classes {classes_}.")
        positive_index = int(np.where(classes_ == positive_label)[0][0])

    y_true_binary = (result.y_true == classes_[positive_index]).astype(int)
    y_scores = result.y_proba[:, positive_index]
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": roc_auc,
    }


def sklearn_classification_report(result: CrossValidationResult) -> str:
    """Return the precision/recall/F1 report."""

    return classification_report(result.y_true, result.y_pred, target_names=[str(c) for c in result.classes_])
