from pathlib import Path

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer

from traffic_forecasting.config import (
    BASELINE_METRICS_PATH,
    ENSEMBLE_METRICS_PATH,
    MODEL_COMPARISON_PATH,
)
from traffic_forecasting.data_loader import load_raw_data
from traffic_forecasting.evaluation import calculate_regression_metrics
from traffic_forecasting.features import build_feature_dataset
from traffic_forecasting.models import (
    get_baseline_model_registry,
    get_core_ensemble_model_registry,
)
from traffic_forecasting.preprocessing import (
    prepare_model_inputs,
    split_chronologically,
    transform_model_inputs,
)


def _fit_and_evaluate_model_registry(
    model_registry: dict[str, RegressorMixin],
    X_train_transformed: object,
    X_validation_transformed: object,
    X_test_transformed: object,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    *,
    include_test_metrics: bool,
) -> tuple[pd.DataFrame, dict[str, RegressorMixin]]:
    fitted_models: dict[str, RegressorMixin] = {}
    metric_rows: list[dict[str, str | bool | float]] = []

    for model_name, model in model_registry.items():
        model.fit(X_train_transformed, y_train)
        fitted_models[model_name] = model

        validation_predictions = model.predict(X_validation_transformed)
        metric_rows.append(
            {
                "model": model_name,
                "split": "validation",
                "used_for_model_comparison": True,
                **calculate_regression_metrics(y_validation, validation_predictions),
            }
        )

        if include_test_metrics:
            test_predictions = model.predict(X_test_transformed)
            metric_rows.append(
                {
                    "model": model_name,
                    "split": "test",
                    "used_for_model_comparison": False,
                    **calculate_regression_metrics(y_test, test_predictions),
                }
            )

    return pd.DataFrame(metric_rows), fitted_models


def train_and_evaluate_baselines(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    *,
    include_test_metrics: bool = False,
) -> tuple[pd.DataFrame, dict[str, RegressorMixin], ColumnTransformer]:
    """Train baselines, compare validation metrics, and optionally evaluate locked test data."""
    (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        preprocessor,
    ) = transform_model_inputs(X_train, X_validation, X_test)

    metrics, fitted_models = _fit_and_evaluate_model_registry(
        get_baseline_model_registry(),
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    return metrics, fitted_models, preprocessor


def train_and_evaluate_ensembles(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    *,
    include_test_metrics: bool = False,
) -> tuple[pd.DataFrame, dict[str, RegressorMixin], ColumnTransformer]:
    """Train core ensembles and optionally evaluate the locked test split."""
    (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        preprocessor,
    ) = transform_model_inputs(X_train, X_validation, X_test)

    metrics, fitted_models = _fit_and_evaluate_model_registry(
        get_core_ensemble_model_registry(),
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    return metrics, fitted_models, preprocessor


def save_baseline_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path = BASELINE_METRICS_PATH,
) -> Path:
    """Save the reproducible baseline metric table."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination, index=False)
    return destination


def save_ensemble_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path = ENSEMBLE_METRICS_PATH,
) -> Path:
    """Save the reproducible core ensemble metric table."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination, index=False)
    return destination


def build_model_comparison(
    baseline_metrics: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine validation metrics while preserving model-family labels."""
    baseline_results = baseline_metrics.copy()
    baseline_results.insert(0, "model_group", "baseline")

    ensemble_results = ensemble_metrics.copy()
    ensemble_results.insert(0, "model_group", "core_ensemble")

    return (
        pd.concat([baseline_results, ensemble_results], ignore_index=True)
        .sort_values(["split", "rmse"])
        .reset_index(drop=True)
    )


def save_model_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path = MODEL_COMPARISON_PATH,
) -> Path:
    """Save the baseline and core ensemble comparison table."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(destination, index=False)
    return destination


def run_baseline_pipeline(
    *,
    include_test_metrics: bool = False,
    metrics_path: str | Path = BASELINE_METRICS_PATH,
) -> pd.DataFrame:
    """Run validation comparison with optional locked-reference test evaluation."""
    feature_data = build_feature_dataset(load_raw_data())
    train_data, validation_data, test_data = split_chronologically(feature_data)
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        _,
        _,
        _,
    ) = prepare_model_inputs(train_data, validation_data, test_data)

    metrics, _, _ = train_and_evaluate_baselines(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    save_baseline_metrics(metrics, metrics_path)
    return metrics


def run_core_ensemble_pipeline(
    *,
    include_test_metrics: bool = False,
    ensemble_metrics_path: str | Path = ENSEMBLE_METRICS_PATH,
    comparison_path: str | Path = MODEL_COMPARISON_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run core ensemble evaluation and compare validation results with baselines."""
    feature_data = build_feature_dataset(load_raw_data())
    train_data, validation_data, test_data = split_chronologically(feature_data)
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        _,
        _,
        _,
    ) = prepare_model_inputs(train_data, validation_data, test_data)

    (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        _,
    ) = transform_model_inputs(X_train, X_validation, X_test)
    baseline_metrics, _ = _fit_and_evaluate_model_registry(
        get_baseline_model_registry(),
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    ensemble_metrics, _ = _fit_and_evaluate_model_registry(
        get_core_ensemble_model_registry(),
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    comparison = build_model_comparison(baseline_metrics, ensemble_metrics)

    save_ensemble_metrics(ensemble_metrics, ensemble_metrics_path)
    save_model_comparison(comparison, comparison_path)
    return ensemble_metrics, comparison


def run_pipeline() -> None:
    """Run the current baseline modeling pipeline."""
    metrics = run_baseline_pipeline()
    print(metrics.to_string(index=False))
