"""Utilities for exporting workflow artifacts to files and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .explainability import compute_shap
from .workflow import WorkflowArtifacts


PathLike = Union[str, Path]


def ensure_output_dir(output_dir: PathLike) -> Path:
    """Create and return the output directory."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rename_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    if "id" in renamed.columns:
        renamed = renamed.rename(columns={"id": "example_id"})
    if "y_true" in renamed.columns:
        renamed = renamed.rename(columns={"y_true": "true_label"})
    if "y_pred" in renamed.columns:
        renamed = renamed.rename(columns={"y_pred": "predicted_label"})
    return renamed


def save_predictions_table(
    artifacts: WorkflowArtifacts,
    output_dir: PathLike,
    filename: str = "predictions.csv",
) -> pd.DataFrame:
    """Write the per-example predictions table and return it."""

    output_path = ensure_output_dir(output_dir) / filename
    predictions = _rename_prediction_columns(artifacts.predictions)
    predictions.to_csv(output_path, index=False)
    return predictions


def _plot_binary_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    positive_class_name: str,
    negative_class_name: str,
    output_dir: Path,
) -> Dict[str, float]:
    positive_scores = y_scores[:, 1] if y_scores.shape[1] > 1 else y_scores[:, 0]
    fpr, tpr, _ = roc_curve(y_true, positive_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve\nPositive: {positive_class_name} | Negative: {negative_class_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, positive_scores)
    average_precision = average_precision_score(y_true, positive_scores)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP = {average_precision:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve\nPositive: {positive_class_name} | Negative: {negative_class_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", bbox_inches="tight")
    plt.close()

    return {
        "class": positive_class_name,
        "auc": float(roc_auc),
        "average_precision": float(average_precision),
    }


def _plot_multiclass_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    classes: Sequence[Union[str, int]],
    class_names: Sequence[str],
    output_dir: Path,
) -> Dict[str, List[Dict[str, float]]]:
    y_true_bin = label_binarize(y_true, classes=classes)

    roc_entries: List[Dict[str, float]] = []
    plt.figure(figsize=(6, 4))
    for idx, (cls, name) in enumerate(zip(classes, class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_scores[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        roc_entries.append({"class": name, "auc": float(roc_auc)})
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()

    pr_entries: List[Dict[str, float]] = []
    plt.figure(figsize=(6, 4))
    for idx, (cls, name) in enumerate(zip(classes, class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_scores[:, idx])
        average_precision = average_precision_score(y_true_bin[:, idx], y_scores[:, idx])
        plt.plot(recall, precision, label=f"{name} (AP = {average_precision:.2f})")
        pr_entries.append({"class": name, "average_precision": float(average_precision)})
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", bbox_inches="tight")
    plt.close()

    return {"roc": roc_entries, "precision_recall": pr_entries}


def generate_curve_plots(
    artifacts: WorkflowArtifacts,
    output_dir: PathLike,
) -> Dict[str, List[Dict[str, float]]]:
    """Generate ROC and PR curves and return the computed metrics."""

    output_path = ensure_output_dir(output_dir)
    classes = artifacts.result.classes_
    y_true = artifacts.result.y_true
    y_scores = artifacts.result.y_proba
    
    # Create reverse mapping from numeric class to label name
    reverse_label_mapping = {v: k for k, v in artifacts.label_mapping.items()}
    class_names = [reverse_label_mapping.get(cls, str(cls)) for cls in classes]

    if classes.size == 2:
        positive_class_name = class_names[1]
        negative_class_name = class_names[0]
        metrics = _plot_binary_curves(
            (y_true == classes[1]).astype(int),
            y_scores,
            positive_class_name=positive_class_name,
            negative_class_name=negative_class_name,
            output_dir=output_path,
        )
        return {
            "roc": [{"class": positive_class_name, "auc": metrics["auc"]}],
            "precision_recall": [
                {"class": positive_class_name, "average_precision": metrics["average_precision"]}
            ],
        }

    return _plot_multiclass_curves(
        y_true,
        y_scores,
        classes=classes,
        class_names=class_names,
        output_dir=output_path,
    )


def export_shap_summaries(
    artifacts: WorkflowArtifacts,
    output_dir: PathLike,
    max_display: int = 20,
) -> pd.DataFrame:
    """Create SHAP summary plots and a feature importance table."""

    output_path = ensure_output_dir(output_dir)

    shap_values = compute_shap(artifacts.result)
    X_concat = np.vstack(artifacts.result.X_tests)
    feature_names = artifacts.result.feature_names
    
    # Create reverse mapping from numeric class to label name
    reverse_label_mapping = {v: k for k, v in artifacts.label_mapping.items()}

    if isinstance(shap_values, list):
        aggregated_abs = np.mean([np.abs(values) for values in shap_values], axis=0)

        for idx, cls in enumerate(artifacts.result.classes_):
            class_name = reverse_label_mapping.get(cls, str(cls))
            plt.figure(figsize=(7, 5))
            shap.summary_plot(
                shap_values[idx],
                X_concat,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
            )
            plt.title(f"SHAP Summary: {class_name}")
            plt.tight_layout()
            plt.savefig(output_path / f"shap_summary_{class_name}.png", bbox_inches="tight")
            plt.close()

        mean_abs_importance = np.asarray(aggregated_abs.mean(axis=0), dtype=float).reshape(-1)
        
        # Overall bar plot using mean absolute SHAP values (consistent with bar chart)
        # Note: Beeswarm of averaged raw SHAP across classes is problematic because
        # opposite-signed values cancel out. Bar plot of mean|SHAP| is more meaningful.
        plt.figure(figsize=(7, 5))
        shap.summary_plot(
            aggregated_abs,
            X_concat,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )
        plt.title("Mean |SHAP| Across Classes")
        plt.tight_layout()
        plt.savefig(output_path / "shap_summary_overall.png", bbox_inches="tight")
        plt.close()
    else:
        aggregated_abs = np.abs(shap_values)
        mean_abs_importance = np.asarray(aggregated_abs.mean(axis=0), dtype=float).reshape(-1)

        plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_values,
            X_concat,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path / "shap_summary.png", bbox_inches="tight")
        plt.close()

    importance_table = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Plot top N feature importance
    top_n = min(max_display, len(importance_table))
    plt.figure(figsize=(7, 5))
    top_features = importance_table.head(top_n).sort_values("mean_abs_shap")
    top_features.plot(
        kind="barh",
        x="feature",
        y="mean_abs_shap",
        legend=False,
        ax=plt.gca(),
    )
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path / "shap_feature_importance.png", bbox_inches="tight")
    plt.close()

    importance_table.to_csv(output_path / "shap_feature_importance.csv", index=False)
    return importance_table


def write_config_json(
    config: Dict[str, object],
    output_dir: PathLike,
    filename: str = "config.json",
) -> Path:
    """Serialize the supplied configuration to JSON."""

    output_path = ensure_output_dir(output_dir) / filename
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return output_path


__all__ = [
    "ensure_output_dir",
    "save_predictions_table",
    "generate_curve_plots",
    "export_shap_summaries",
    "write_config_json",
]
