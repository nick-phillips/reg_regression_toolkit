#!/usr/bin/env python3
"""Command-line driver for the demo workflow used in the examples notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from reg_regression_toolkit import evaluation
from reg_regression_toolkit.filters import load_feature_list
from reg_regression_toolkit.reporting import (
    ensure_output_dir,
    export_shap_summaries,
    generate_curve_plots,
    save_predictions_table,
    write_config_json,
)
from reg_regression_toolkit.workflow import run_workflow


def _default_logistic_kwargs() -> dict:
    return {
        "Cs": [0.001, 0.01, 0.1, 1.0, 10.0],
        "cv": 5,
        "penalty": "l1",
        "solver": "saga",
        "scoring": "neg_log_loss",
        "class_weight": "balanced",
        "max_iter": 5000,
        "n_jobs": None,
        "random_state": 42,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the demo workflow end-to-end.")
    parser.add_argument("--expression", default="data/dummy_expression.csv", help="Path to the expression matrix.")
    parser.add_argument("--metadata", default="data/dummy_metadata.csv", help="Path to the metadata table.")
    parser.add_argument(
        "--remove-features",
        help="Optional feature list file containing names to drop.",
    )
    parser.add_argument(
        "--keep-features",
        help="Optional feature list file containing the whitelist features.",
    )
    parser.add_argument(
        "--label-column",
        default="Sample_type",
        help="Label column present in the metadata table.",
    )
    parser.add_argument("--output-dir", default="outputs/workflow_cli", help="Where to store generated artifacts.")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of outer cross-validation folds.")
    parser.add_argument(
        "--binary-labels",
        nargs=2,
        metavar=("POS", "NEG"),
        help="Optional pair of labels to use for an additional binary ROC evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed forwarded to cross-validation and label encoding.",
    )
    return parser.parse_args()


def _subset_binary(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    label_column: str,
    allowed_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = metadata[label_column].isin(allowed_labels)
    binary_metadata = metadata.loc[mask].reset_index(drop=True)
    binary_expression = expression[expression["sample_id"].isin(binary_metadata["sample_id"])].reset_index(drop=True)
    return binary_expression, binary_metadata


def main() -> None:
    args = parse_args()

    expression = pd.read_csv(args.expression)
    metadata = pd.read_csv(args.metadata)
    remove_features = load_feature_list(args.remove_features) if args.remove_features else None
    keep_features = load_feature_list(args.keep_features) if args.keep_features else None

    logistic_kwargs = _default_logistic_kwargs()
    output_dir = ensure_output_dir(args.output_dir)

    artifacts = run_workflow(
        expression,
        metadata_df=metadata,
        label_column=args.label_column,
        remove_features=remove_features,
        keep_features=keep_features,
        add_back_features=("feature_signal_partner",),
        cv_splits=args.cv_splits,
        logistic_kwargs=logistic_kwargs,
        random_state=args.random_state,
    )

    metrics_summary = generate_curve_plots(artifacts, output_dir)
    predictions = save_predictions_table(artifacts, output_dir)
    shap_table = export_shap_summaries(artifacts, output_dir)

    inputs_section = {
        "expression_path": str(Path(args.expression).resolve()),
        "metadata_path": str(Path(args.metadata).resolve()),
    }
    if args.remove_features:
        inputs_section["remove_features_path"] = str(Path(args.remove_features).resolve())
    if args.keep_features:
        inputs_section["keep_features_path"] = str(Path(args.keep_features).resolve())

    filters_section = {
        "add_back_features": ["feature_signal_partner"],
    }
    if remove_features is not None:
        filters_section["remove_features"] = remove_features
    if keep_features is not None:
        filters_section["keep_features"] = keep_features

    config_data = {
        "inputs": inputs_section,
        "filters": filters_section,
        "cross_validation": {
            "splits": args.cv_splits,
            "shuffle": True,
            "random_state": args.random_state,
        },
        "model": {
            "type": "LogisticRegressionCV",
            "parameters": logistic_kwargs,
        },
        "output_dir": str(output_dir.resolve()),
    }
    write_config_json(config_data, output_dir)

    print(f"Wrote predictions to {output_dir/'predictions.csv'} ({len(predictions)} rows)")
    print(f"Generated ROC/PR plots in {output_dir}")
    print(f"Saved SHAP summaries and table with {len(shap_table)} rows")

    if args.binary_labels:
        binary_expression, binary_metadata = _subset_binary(
            expression,
            metadata,
            args.label_column,
            args.binary_labels,
        )
        binary_artifacts = run_workflow(
            binary_expression,
            metadata_df=binary_metadata,
            label_column=args.label_column,
            remove_features=remove_features,
            keep_features=keep_features,
            add_back_features=("feature_signal_partner",),
            cv_splits=args.cv_splits,
            logistic_kwargs=logistic_kwargs,
            random_state=args.random_state,
        )
        roc = evaluation.roc_curve_from_result(binary_artifacts.result, positive_label=binary_artifacts.result.classes_[1])
        print(
            f"Binary ROC AUC for labels {args.binary_labels[0]} vs {args.binary_labels[1]}: "
            f"{roc['auc']:.3f}"
        )

    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
