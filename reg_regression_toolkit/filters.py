"""Feature selection utilities used throughout the workflow."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def load_feature_list(path: str) -> list[str]:
    """Load a list of feature names from disk.

    The file should contain one feature name per line. Blank lines and lines
    that begin with ``#`` are ignored. No header row is required.
    """

    features: list[str] = []
    first_value = True
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            if first_value and entry.lower() in {"gene_symbol", "feature", "features", "feature_name"}:
                first_value = False
                continue
            first_value = False
            features.append(entry)
    return features


def drop_features(
    df: pd.DataFrame,
    features: Iterable[str],
    *,
    id_column: str = "sample_id",
) -> pd.DataFrame:
    """Drop provided feature names from the feature matrix."""

    selected = [feature for feature in features if feature in df.columns and feature != id_column]
    if not selected:
        return df
    return df.drop(columns=selected)


def keep_features(
    df: pd.DataFrame,
    features: Iterable[str],
    *,
    id_column: str = "sample_id",
) -> pd.DataFrame:
    """Keep only the specified feature names along with the identifier column."""

    selected = [feature for feature in features if feature in df.columns]
    columns: list[str] = []
    if id_column in df.columns:
        columns.append(id_column)
    columns.extend(selected)
    if not columns:
        return df.iloc[:, 0:0]
    return df.loc[:, columns]


def merge_with_backup_columns(
    df: pd.DataFrame,
    backup_df: pd.DataFrame,
    columns_to_add: Iterable[str],
    *,
    id_column: str = "sample_id",
) -> pd.DataFrame:
    """Add columns from a backup DataFrame via left merge."""

    columns_to_add = [
        col
        for col in columns_to_add
        if col in backup_df.columns and col != id_column and col not in df.columns
    ]
    if not columns_to_add:
        return df

    add_back = backup_df.loc[:, [id_column] + columns_to_add]
    merged = df.merge(add_back, on=id_column, how="left")
    return merged


def sequential_feature_filters(
    df: pd.DataFrame,
    *,
    remove_feature_lists: Optional[Iterable[Iterable[str]]] = None,
    keep_feature_lists: Optional[Iterable[Iterable[str]]] = None,
    add_back_columns: Optional[Iterable[str]] = None,
    backup_df: Optional[pd.DataFrame] = None,
    id_column: str = "sample_id",
) -> pd.DataFrame:
    """Apply a chain of drop/keep operations to the input DataFrame."""

    filtered = df.copy()

    if remove_feature_lists:
        for features in remove_feature_lists:
            filtered = drop_features(filtered, features, id_column=id_column)

    if keep_feature_lists:
        ordered_columns: list[str] = []
        seen: set[str] = set()
        for features in keep_feature_lists:
            for feature in features:
                if feature not in seen:
                    ordered_columns.append(feature)
                    seen.add(feature)
        filtered = keep_features(filtered, ordered_columns, id_column=id_column)

    if add_back_columns and backup_df is not None:
        filtered = merge_with_backup_columns(
            filtered,
            backup_df,
            add_back_columns,
            id_column=id_column,
        )

    return filtered
