"""High-level orchestration for the regularized regression workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd
from sklearn.base import BaseEstimator

from . import data as data_utils
from . import evaluation, filters, importance, model


@dataclass
class FilterConfig:
    """Configuration for sequential feature filtering."""

    remove_feature_lists: Optional[Sequence[Iterable[str]]] = None
    keep_feature_lists: Optional[Sequence[Iterable[str]]] = None
    add_back_columns: Sequence[str] = field(default_factory=tuple)


@dataclass
class WorkflowArtifacts:
    """Outputs produced by the high-level workflow."""

    result: model.CrossValidationResult
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    coefficient_summary: pd.DataFrame
    label_mapping: Dict[str, int]


def _normalize_feature_lists(
    feature_lists: Optional[Sequence[Iterable[str] | str]],
) -> Optional[Sequence[Iterable[str]]]:
    if not feature_lists:
        return None

    if isinstance(feature_lists, (list, tuple)) and feature_lists and all(isinstance(entry, str) for entry in feature_lists):
        return (list(feature_lists),)

    normalized = []
    for features in feature_lists:
        if isinstance(features, str):
            normalized.append([features])
        else:
            normalized.append(list(features))
    return tuple(normalized)


def _apply_filters(
    df: pd.DataFrame,
    *,
    config: Optional[FilterConfig],
    id_column: str,
) -> pd.DataFrame:
    if config is None:
        return df

    backup = df.copy()
    return filters.sequential_feature_filters(
        df,
        remove_feature_lists=_normalize_feature_lists(config.remove_feature_lists),
        keep_feature_lists=_normalize_feature_lists(config.keep_feature_lists),
        add_back_columns=list(config.add_back_columns),
        backup_df=backup,
        id_column=id_column,
    )


def _resolve_filter_config(
    filter_config: Optional[FilterConfig],
    *,
    remove_features: Optional[Sequence[Iterable[str] | str]] = None,
    keep_features: Optional[Sequence[Iterable[str] | str]] = None,
    add_back_features: Optional[Sequence[str]] = None,
) -> Optional[FilterConfig]:
    if filter_config is not None:
        return filter_config

    has_manual_filters = any(
        value for value in (remove_features, keep_features, add_back_features) if value
    )
    if not has_manual_filters:
        return None

    return FilterConfig(
        remove_feature_lists=_normalize_feature_lists(remove_features),
        keep_feature_lists=_normalize_feature_lists(keep_features),
        add_back_columns=tuple(add_back_features or ()),
    )


def run_workflow(
    features_df: pd.DataFrame,
    *,
    metadata_df: Optional[pd.DataFrame] = None,
    label_column: str,
    id_column: str = "sample_id",
    filter_config: Optional[FilterConfig] = None,
    remove_features: Optional[Sequence[Iterable[str] | str]] = None,
    keep_features: Optional[Sequence[Iterable[str] | str]] = None,
    add_back_features: Optional[Sequence[str]] = None,
    cv_splits: int = 10,
    scaler: Optional[BaseEstimator] = None,
    logistic_kwargs: Optional[Dict[str, object]] = None,
    random_state: int = 42,
    shuffle: bool = True,
) -> WorkflowArtifacts:
    """Execute the end-to-end workflow on the provided features and labels."""

    resolved_config = _resolve_filter_config(
        filter_config,
        remove_features=remove_features,
        keep_features=keep_features,
        add_back_features=add_back_features,
    )

    filtered = _apply_filters(features_df, config=resolved_config, id_column=id_column)

    if metadata_df is not None:
        filtered = data_utils.merge_with_metadata(filtered, metadata_df, id_column=id_column)

    if label_column not in filtered.columns:
        raise ValueError(f"Label column '{label_column}' not found after merging metadata.")

    encoded = data_utils.encode_labels(filtered, label_column=label_column)
    filtered = filtered.copy()
    filtered[label_column] = encoded.values

    ordered = data_utils.enforce_column_order(filtered, id_column=id_column, label_column=label_column)
    X_df, y_series, ids_series = data_utils.split_features_and_target(
        ordered,
        label_column=label_column,
        id_column=id_column,
    )

    result = model.cross_validate_logistic_regression(
        X_df.to_numpy(),
        y_series.to_numpy(),
        ids=None if ids_series is None else ids_series.to_numpy(),
        cv_splits=cv_splits,
        scaler=scaler,
        logistic_kwargs=logistic_kwargs,
        random_state=random_state,
        shuffle=shuffle,
        feature_names=list(X_df.columns),
    )

    predictions = evaluation.predictions_dataframe(result)
    fold_metrics = evaluation.fold_metrics_dataframe(result)
    coefficient_summary = importance.summarize_coefficients(result)

    return WorkflowArtifacts(
        result=result,
        predictions=predictions,
        fold_metrics=fold_metrics,
        coefficient_summary=coefficient_summary,
        label_mapping={label: idx for label, idx in encoded.mapping.items()},
    )


