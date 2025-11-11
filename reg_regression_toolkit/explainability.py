"""Explainability helpers built on top of SHAP."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import shap

from .model import CrossValidationResult


def _prepare_background(
    background: Optional[np.ndarray],
    fallback: np.ndarray,
    *,
    background_fraction: float,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    data = fallback if background is None else background
    if not 0 < background_fraction <= 1:
        raise ValueError("background_fraction must be in (0, 1].")

    sample_size = max(1, int(len(data) * background_fraction))
    if sample_size >= len(data):
        return data
    indices = rng.choice(len(data), size=sample_size, replace=False)
    return data[indices]


def compute_linear_shap(
    result: CrossValidationResult,
    *,
    background: Optional[np.ndarray] = None,
    background_fraction: float = 1.0,
    random_state: int = 42,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Compute SHAP values for each fold and concatenate the results.

    Returns either a single array (binary case) or a list of arrays (multi-class case)
    where each array corresponds to a class.
    """

    shap_outputs: List[Union[np.ndarray, List[np.ndarray]]] = []

    for model, X_test in zip(result.models, result.X_tests):
        bg = _prepare_background(background, X_test, background_fraction=background_fraction, random_state=random_state)
        explainer = shap.LinearExplainer(model, bg)
        shap_values = explainer.shap_values(X_test)
        shap_outputs.append(shap_values)

    first_output = shap_outputs[0]
    if isinstance(first_output, list):
        concatenated: List[np.ndarray] = []
        num_classes = len(first_output)
        for class_idx in range(num_classes):
            concatenated.append(
                np.concatenate(
                    [fold_values[class_idx] for fold_values in shap_outputs],
                    axis=0,
                )
            )
        return concatenated

    if isinstance(first_output, np.ndarray) and first_output.ndim == 3:
        stacked = np.concatenate(shap_outputs, axis=0)
        num_classes = stacked.shape[2]
        return [stacked[:, :, idx] for idx in range(num_classes)]

    return np.concatenate(shap_outputs, axis=0)
