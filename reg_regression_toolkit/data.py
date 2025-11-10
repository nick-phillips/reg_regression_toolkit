"""Data loading and preprocessing helpers for regularized regression workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class EncodedLabels:
    """Container for encoded label information."""

    values: pd.Series
    mapping: Dict[str, int]


def load_table(path: str, sep: str = ",", **kwargs) -> pd.DataFrame:
    """Load a delimited table into a DataFrame."""

    return pd.read_csv(path, sep=sep, **kwargs)


def restructure_transposed_expression(
    raw_df: pd.DataFrame,
    *,
    sample_id_column: str = "sample_id",
    metadata_rows: int = 2,
    gene_symbol_column: str = "gene_symbol",
    drop_leading_columns: int = 0,
) -> pd.DataFrame:
    """Reshape an expression matrix that stores metadata in the first rows.

    Many internal expression exports store metadata rows that need to be removed
    after transposition. This helper mirrors the transformation performed in the
    original BLCA notebook but exposes the magic numbers as parameters.
    """

    transposed = raw_df.T.copy()
    transposed.columns = transposed.iloc[0]
    transposed = transposed.drop(transposed.index[0]).reset_index(drop=False)
    transposed = transposed.rename(columns={"index": sample_id_column})

    if metadata_rows > 0:
        transposed.columns = transposed.iloc[0]
        for _ in range(metadata_rows):
            transposed = transposed.drop(transposed.index[0])
        transposed = transposed.reset_index(drop=True)

    if gene_symbol_column in transposed.columns:
        transposed = transposed.rename(columns={gene_symbol_column: sample_id_column})

    if drop_leading_columns > 0:
        transposed = transposed.iloc[:, drop_leading_columns:]

    return transposed


def merge_with_metadata(
    features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    id_column: str = "sample_id",
    how: str = "inner",
) -> pd.DataFrame:
    """Join feature matrix with sample-level metadata."""

    return features_df.merge(metadata_df, on=id_column, how=how)


def encode_labels(
    df: pd.DataFrame,
    *,
    label_column: str,
    mapping: Optional[Dict[str, int]] = None,
) -> EncodedLabels:
    """Encode categorical labels and capture the mapping."""

    if mapping is None:
        unique_labels = sorted(df[label_column].dropna().unique())
        mapping = {label: idx for idx, label in enumerate(unique_labels)}

    encoded = df[label_column].map(mapping)

    if encoded.isnull().any():
        missing_labels = df.loc[encoded.isnull(), label_column].unique()
        raise ValueError(f"Missing mapping for labels: {missing_labels}")

    return EncodedLabels(values=encoded.astype(int), mapping=mapping)


def split_features_and_target(
    df: pd.DataFrame,
    *,
    label_column: str,
    id_column: str = "sample_id",
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Split DataFrame into features, target, and optional identifiers."""

    feature_columns = [c for c in df.columns if c not in {label_column, id_column}]
    features = df[feature_columns]
    target = df[label_column]
    identifiers = df[id_column] if id_column in df.columns else None
    return features, target, identifiers


def enforce_column_order(
    df: pd.DataFrame,
    *,
    id_column: str = "sample_id",
    label_column: Optional[str] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Reorder columns to `[id, features..., label]` if available."""

    columns: list[str] = []
    if id_column in df.columns:
        columns.append(id_column)

    if feature_order is not None:
        columns.extend([col for col in feature_order if col in df.columns])
    else:
        inferred_features = [c for c in df.columns if c not in {id_column, label_column}]
        columns.extend(inferred_features)

    if label_column and label_column in df.columns:
        columns.append(label_column)

    return df.loc[:, columns]


def apply_label_mapping(
    df: pd.DataFrame,
    *,
    label_column: str,
    mapping: Dict[str, int],
    new_column: str = "label_id",
) -> pd.DataFrame:
    """Append an encoded label column based on a mapping."""

    encoded, _ = encode_labels(df, label_column=label_column, mapping=mapping)
    df = df.copy()
    df[new_column] = encoded.values
    return df
