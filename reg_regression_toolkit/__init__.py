"""Public API surface for reg_regression_toolkit."""

from .data import (
    EncodedLabels,
    encode_labels,
    enforce_column_order,
    load_table,
    merge_with_metadata,
    restructure_transposed_expression,
    split_features_and_target,
)
from .evaluation import (
    aggregate_confusion_matrix,
    fold_metrics_dataframe,
    predictions_dataframe,
    roc_curve_from_result,
    sklearn_classification_report,
)
from .filters import load_feature_list, sequential_feature_filters
from .importance import summarize_coefficients
from .model import (
    CrossValidationResult,
    FoldResult,
    cross_validate_logistic_regression,
    fit_logistic_regression_cv,
)
from .reporting import (
    ensure_output_dir,
    export_shap_summaries,
    generate_curve_plots,
    save_predictions_table,
    write_config_json,
)
from .workflow import FilterConfig, WorkflowArtifacts, run_workflow

__all__ = [
    "aggregate_confusion_matrix",
    "CrossValidationResult",
    "EncodedLabels",
    "FilterConfig",
    "FoldResult",
    "WorkflowArtifacts",
    "cross_validate_logistic_regression",
    "encode_labels",
    "enforce_column_order",
    "fit_logistic_regression_cv",
    "ensure_output_dir",
    "export_shap_summaries",
    "generate_curve_plots",
    "load_table",
    "load_feature_list",
    "merge_with_metadata",
    "predictions_dataframe",
    "restructure_transposed_expression",
    "roc_curve_from_result",
    "sequential_feature_filters",
    "sklearn_classification_report",
    "split_features_and_target",
    "summarize_coefficients",
    "save_predictions_table",
    "write_config_json",
    "run_workflow",
]

