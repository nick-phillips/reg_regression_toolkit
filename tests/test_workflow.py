import pandas as pd

from reg_regression_toolkit import evaluation, workflow
from reg_regression_toolkit.filters import load_feature_list, sequential_feature_filters


def _load_dummy_frames():
    expression = pd.read_csv("data/dummy_expression.csv")
    metadata = pd.read_csv("data/dummy_metadata.csv")
    return expression, metadata


def _filter_lists():
    remove_features = load_feature_list("data/dummy_remove_features.txt")
    keep_features = load_feature_list("data/dummy_keep_features.txt")
    return remove_features, keep_features


def test_sequential_filters_roundtrip():
    expression, _ = _load_dummy_frames()
    remove_features, keep_features = _filter_lists()

    filtered = sequential_feature_filters(
        expression,
        remove_feature_lists=[remove_features],
        keep_feature_lists=[keep_features],
        add_back_columns=["feature_signal_partner"],
        backup_df=expression.copy(),
    )

    assert all(
        feature in filtered.columns
        for feature in ["sample_id", "feature_signal_primary", "feature_signal_partner", "feature_signal_secondary"]
    )
    for noise_feature in remove_features:
        assert noise_feature not in filtered.columns


def test_workflow_without_filter_config():
    expression, metadata = _load_dummy_frames()
    artifacts = workflow.run_workflow(
        expression,
        metadata_df=metadata,
        label_column="Sample_type",
        filter_config=None,
        cv_splits=3,
        logistic_kwargs={
            "Cs": [0.1, 1.0],
            "l1_ratios": [0.5, 0.9],
            "cv": 3,
            "max_iter": 500,
            "n_jobs": None,
        },
    )

    assert set(artifacts.result.classes_) == {0, 1, 2}
    assert len(artifacts.predictions) == len(expression)


def test_filter_config_still_supported():
    expression, metadata = _load_dummy_frames()
    remove_features, keep_features = _filter_lists()
    config = workflow.FilterConfig(
        remove_feature_lists=[remove_features],
        keep_feature_lists=[keep_features],
        add_back_columns=("feature_signal_partner",),
    )

    artifacts = workflow.run_workflow(
        expression,
        metadata_df=metadata,
        label_column="Sample_type",
        filter_config=config,
        cv_splits=3,
        logistic_kwargs={
            "Cs": [0.1, 1.0],
            "l1_ratios": [0.5, 0.9],
            "cv": 3,
            "max_iter": 500,
            "n_jobs": None,
        },
    )

    assert set(artifacts.result.classes_) == {0, 1, 2}
    assert len(artifacts.predictions) == len(expression)


def test_signal_features_dominate_coefficients():
    expression, metadata = _load_dummy_frames()
    remove_features, keep_features = _filter_lists()
    artifacts = workflow.run_workflow(
        expression,
        metadata_df=metadata,
        label_column="Sample_type",
        remove_features=remove_features,
        keep_features=keep_features,
        add_back_features=("feature_signal_partner",),
        cv_splits=3,
        logistic_kwargs={
            "Cs": [0.1, 1.0],
            "l1_ratios": [0.5, 0.9],
            "cv": 3,
            "max_iter": 500,
            "n_jobs": None,
        },
    )

    summary = artifacts.coefficient_summary
    assert not summary.empty

    signal_features = {"feature_signal_primary", "feature_signal_partner", "feature_signal_secondary"}
    noise_features = {"feature_noise_a", "feature_noise_b", "feature_noise_c"}

    assert signal_features.issubset(set(summary["feature"]))
    assert noise_features.isdisjoint(set(summary["feature"]))


def test_binary_workflow_and_roc_curve():
    expression, metadata = _load_dummy_frames()
    binary_metadata = metadata[metadata["Sample_type"].isin(["Control", "BLCA"])].reset_index(drop=True)
    binary_expression = expression[expression["sample_id"].isin(binary_metadata["sample_id"])].reset_index(drop=True)

    remove_features, keep_features = _filter_lists()
    artifacts = workflow.run_workflow(
        binary_expression,
        metadata_df=binary_metadata,
        label_column="Sample_type",
        remove_features=remove_features,
        keep_features=keep_features,
        add_back_features=("feature_signal_partner",),
        cv_splits=3,
        logistic_kwargs={
            "Cs": [0.1, 1.0],
            "l1_ratios": [0.5, 0.9],
            "cv": 3,
            "max_iter": 500,
            "n_jobs": None,
        },
    )

    assert set(artifacts.result.classes_) == {0, 1}
    roc_info = evaluation.roc_curve_from_result(artifacts.result, positive_label=1)
    assert 0.0 <= roc_info["auc"] <= 1.0
    # Perfect separation is expected on the dummy dataset.
    assert roc_info["auc"] > 0.95
