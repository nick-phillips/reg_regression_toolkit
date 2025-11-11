"""Coefficient aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

import pandas as pd

from .model import CrossValidationResult


@dataclass
class CoefficientRecord:
    """Record capturing coefficient statistics for a feature/class pair."""

    feature: str
    class_label: str
    num_models_non_zero: int
    mean_coef: float
    var_coef: float
    abs_mean_coef: float


def _collect_coefficients(
    result: CrossValidationResult,
) -> dict[tuple[str, str], List[float]]:
    storage: dict[tuple[str, str], List[float]] = {}

    for model in result.models:
        coefs = np.atleast_2d(model.coef_)
        classes = model.classes_

        if coefs.shape[0] == 1 and classes.size == 2:
            class_iterable = [(classes[1], coefs[0])]
        else:
            class_iterable = [
                (classes[class_idx], coefs[class_idx])
                for class_idx in range(coefs.shape[0])
            ]

        for class_label, class_coefs in class_iterable:
            for feature_idx, coef in enumerate(class_coefs):
                if np.isclose(coef, 0.0):
                    continue
                key = (result.feature_names[feature_idx], str(class_label))
                storage.setdefault(key, []).append(float(coef))

    return storage


def summarize_coefficients(result: CrossValidationResult) -> pd.DataFrame:
    """Summarize non-zero coefficients across folds.

    Returns a DataFrame sorted by the number of folds with a non-zero
    coefficient followed by the absolute mean coefficient magnitude.
    """

    aggregated = _collect_coefficients(result)
    records: List[CoefficientRecord] = []

    for (feature, class_label), values in aggregated.items():
        values_array = np.asarray(values, dtype=float)
        record = CoefficientRecord(
            feature=feature,
            class_label=class_label,
            num_models_non_zero=len(values),
            mean_coef=float(np.mean(values_array)),
            var_coef=float(np.var(values_array)),
            abs_mean_coef=float(np.abs(np.mean(values_array))),
        )
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "feature",
                "class_label",
                "num_models_non_zero",
                "mean_coef",
                "var_coef",
                "abs_mean_coef",
            ]
        )

    df = pd.DataFrame(records)
    df.sort_values(
        ["num_models_non_zero", "abs_mean_coef"],
        ascending=[False, False],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df
