import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
from traffic_forecasting.pipeline import (
    save_baseline_metrics,
    train_and_evaluate_baselines,
)
from traffic_forecasting.preprocessing import (
    get_model_feature_groups,
    prepare_model_inputs,
    split_chronologically,
)


def _make_synthetic_feature_data(periods: int = 60) -> pd.DataFrame:
    feature_groups = get_model_feature_groups()
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: pd.date_range("2026-01-01", periods=periods, freq="h"),
            TARGET_COLUMN: np.linspace(1_000, 2_000, periods),
            "holiday": [pd.NA] * periods,
        }
    )

    for offset, column in enumerate(feature_groups["continuous_numeric"]):
        data[column] = np.arange(periods, dtype=float) + offset

    data["weather_main"] = ["Clear"] * periods
    data["weather_description"] = ["clear sky"] * periods
    data["is_weekend"] = (data[DATETIME_COLUMN].dt.dayofweek >= 5).astype("int8")
    data["is_holiday"] = np.zeros(periods, dtype="int8")

    for offset, column in enumerate(feature_groups["cyclical"]):
        data[column] = np.sin(np.arange(periods) + offset)

    return data


def _prepare_inputs() -> tuple[pd.DataFrame | pd.Series, ...]:
    splits = split_chronologically(_make_synthetic_feature_data())
    return prepare_model_inputs(*splits)[:6]


def test_train_and_evaluate_baselines_returns_validation_metrics_by_default() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = _prepare_inputs()

    metrics, fitted_models, preprocessor = train_and_evaluate_baselines(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

    assert len(metrics) == len(fitted_models)
    assert set(metrics["split"]) == {"validation"}
    assert metrics["used_for_model_comparison"].all()
    assert np.isfinite(metrics[["mae", "rmse", "mape", "r2"]].to_numpy()).all()

    check_is_fitted(preprocessor)
    for model in fitted_models.values():
        check_is_fitted(model)


def test_train_and_evaluate_baselines_calculates_locked_test_metrics_explicitly() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = _prepare_inputs()

    metrics, fitted_models, _ = train_and_evaluate_baselines(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=True,
    )

    assert len(metrics) == len(fitted_models) * 2
    assert set(metrics["split"]) == {"validation", "test"}
    assert not metrics.loc[
        metrics["split"] == "test",
        "used_for_model_comparison",
    ].any()


def test_preprocessing_statistics_are_fitted_from_train_only() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = _prepare_inputs()
    X_validation = X_validation.copy()
    X_test = X_test.copy()
    X_validation["temp"] = 100_000
    X_test["temp"] = 200_000

    _, _, preprocessor = train_and_evaluate_baselines(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

    continuous_columns = get_model_feature_groups()["continuous_numeric"]
    temp_index = continuous_columns.index("temp")
    scaler = preprocessor.named_transformers_["continuous"].named_steps["scaler"]

    assert scaler.mean_[temp_index] == pytest.approx(X_train["temp"].mean())
    assert scaler.mean_[temp_index] != pytest.approx(X_validation["temp"].mean())


def test_save_baseline_metrics_writes_csv(tmp_path) -> None:
    metrics = pd.DataFrame(
        {
            "model": ["dummy_mean"],
            "split": ["validation"],
            "used_for_model_comparison": [True],
            "mae": [100.0],
            "rmse": [120.0],
            "mape": [5.0],
            "r2": [0.5],
        }
    )
    output_path = tmp_path / "metrics" / "baseline.csv"

    saved_path = save_baseline_metrics(metrics, output_path)

    assert saved_path == output_path
    pd.testing.assert_frame_equal(pd.read_csv(saved_path), metrics)
