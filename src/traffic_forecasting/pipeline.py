from pathlib import Path

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer

from traffic_forecasting.config import BASELINE_METRICS_PATH
from traffic_forecasting.data_loader import load_raw_data
from traffic_forecasting.evaluation import calculate_regression_metrics
from traffic_forecasting.features import build_feature_dataset
from traffic_forecasting.models import get_baseline_model_registry
from traffic_forecasting.preprocessing import (
    prepare_model_inputs,
    split_chronologically,
    transform_model_inputs,
)


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

    fitted_models: dict[str, RegressorMixin] = {}
    metric_rows: list[dict[str, str | bool | float]] = []

    for model_name, model in get_baseline_model_registry().items():
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

    metrics = pd.DataFrame(metric_rows)
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


def run_pipeline() -> None:
    """Run the current baseline modeling pipeline."""
    metrics = run_baseline_pipeline()
    print(metrics.to_string(index=False))
