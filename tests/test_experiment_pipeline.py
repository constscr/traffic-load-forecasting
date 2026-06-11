import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
from traffic_forecasting.pipeline import (
    build_experiment_comparison,
    run_experiment_pipeline,
    save_experiment_predictions,
    save_prepared_datasets,
    save_trained_model_pipelines,
    train_and_evaluate_experiment_models,
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


def _prepare_experiment_inputs() -> tuple[pd.DataFrame | pd.Series, ...]:
    splits = split_chronologically(_make_synthetic_feature_data())
    return prepare_model_inputs(*splits)


def test_save_prepared_datasets_writes_interim_and_processed_csv(tmp_path) -> None:
    hourly_data = pd.DataFrame({DATETIME_COLUMN: pd.date_range("2026-01-01", periods=2)})
    feature_data = hourly_data.assign(feature=[1.0, 2.0])
    paths = (
        tmp_path / "data" / "interim" / "hourly.csv",
        tmp_path / "data" / "processed" / "features.csv",
    )

    saved_paths = save_prepared_datasets(
        hourly_data,
        feature_data,
        interim_path=paths[0],
        processed_path=paths[1],
    )

    assert saved_paths == paths
    assert pd.read_csv(paths[0]).shape == hourly_data.shape
    assert pd.read_csv(paths[1]).shape == feature_data.shape


def test_experiment_models_create_validation_outputs_without_test_by_default() -> None:
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        _,
        validation_timestamps,
        test_timestamps,
    ) = _prepare_experiment_inputs()

    metrics, predictions, fitted_pipelines = train_and_evaluate_experiment_models(
        {"ridge": Ridge()},
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        validation_timestamps,
        test_timestamps,
    )

    assert set(metrics["split"]) == {"validation"}
    assert set(predictions["split"]) == {"validation"}
    assert metrics["used_for_model_comparison"].all()
    assert tuple(fitted_pipelines) == ("ridge",)
    check_is_fitted(fitted_pipelines["ridge"])
    continuous_columns = get_model_feature_groups()["continuous_numeric"]
    temp_index = continuous_columns.index("temp")
    scaler = (
        fitted_pipelines["ridge"]
        .named_steps["preprocessing"]
        .named_transformers_["continuous"]
        .named_steps["scaler"]
    )
    assert scaler.mean_[temp_index] == pytest.approx(X_train["temp"].mean())


def test_experiment_test_outputs_require_explicit_opt_in() -> None:
    inputs = _prepare_experiment_inputs()

    metrics, predictions, _ = train_and_evaluate_experiment_models(
        {"dummy_mean": DummyRegressor()},
        *inputs[:6],
        inputs[7],
        inputs[8],
        include_test_outputs=True,
    )

    assert set(metrics["split"]) == {"validation", "test"}
    assert set(predictions["split"]) == {"validation", "test"}
    assert not metrics.loc[
        metrics["split"] == "test",
        "used_for_model_comparison",
    ].any()


def test_comparison_uses_validation_metrics_only() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["ridge", "ridge", "dummy_mean"],
            "split": ["validation", "test", "validation"],
            "used_for_model_comparison": [True, False, True],
            "mae": [10.0, 1.0, 20.0],
            "rmse": [12.0, 1.0, 25.0],
            "mape": [2.0, 0.1, 4.0],
            "r2": [0.9, 0.99, 0.7],
        }
    )

    comparison = build_experiment_comparison(metrics)

    assert comparison["model"].tolist() == ["ridge", "dummy_mean"]
    assert comparison["validation_rank"].tolist() == [1, 2]
    assert set(comparison["split"]) == {"validation"}


def test_prediction_and_model_persistence_are_explicit(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "model": ["ridge"],
            "split": ["validation"],
            DATETIME_COLUMN: [pd.Timestamp("2026-01-01")],
            f"actual_{TARGET_COLUMN}": [1_000.0],
            f"predicted_{TARGET_COLUMN}": [990.0],
        }
    )
    fitted_pipeline = Pipeline([("model", Ridge())]).fit([[0.0], [1.0]], [0.0, 1.0])
    validation_path = tmp_path / "predictions" / "validation.csv"
    test_path = tmp_path / "predictions" / "test.csv"

    prediction_paths = save_experiment_predictions(
        predictions,
        validation_path=validation_path,
        test_path=test_path,
    )
    model_paths = save_trained_model_pipelines(
        {"ridge": fitted_pipeline},
        ("ridge",),
        models_dir=tmp_path / "models",
    )

    assert prediction_paths == {"validation": validation_path}
    assert not test_path.exists()
    assert tuple(model_paths) == ("ridge",)
    loaded_pipeline = joblib.load(model_paths["ridge"])
    check_is_fitted(loaded_pipeline)


def test_run_experiment_pipeline_generates_reproducible_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    feature_data = _make_synthetic_feature_data()
    hourly_data = feature_data.loc[:, [DATETIME_COLUMN, TARGET_COLUMN]].copy()
    monkeypatch.setattr(
        "traffic_forecasting.pipeline.prepare_and_save_datasets",
        lambda **kwargs: (
            hourly_data,
            feature_data,
            save_prepared_datasets(
                hourly_data,
                feature_data,
                interim_path=kwargs["interim_path"],
                processed_path=kwargs["processed_path"],
            ),
        ),
    )
    monkeypatch.setattr(
        "traffic_forecasting.pipeline.get_experiment_model_registry",
        lambda: {"ridge": Ridge()},
    )
    paths = {
        "interim_path": tmp_path / "data" / "interim.csv",
        "processed_path": tmp_path / "data" / "processed.csv",
        "metrics_path": tmp_path / "reports" / "metrics.csv",
        "comparison_path": tmp_path / "reports" / "comparison.csv",
        "validation_predictions_path": tmp_path / "artifacts" / "validation.csv",
        "test_predictions_path": tmp_path / "artifacts" / "test.csv",
        "models_dir": tmp_path / "artifacts" / "models",
    }

    metrics, comparison, predictions, output_paths = run_experiment_pipeline(
        model_names=("ridge",),
        persisted_model_names=("ridge",),
        **paths,
    )

    assert len(metrics) == 1
    assert comparison["model"].tolist() == ["ridge"]
    assert set(predictions["split"]) == {"validation"}
    assert not paths["test_predictions_path"].exists()
    assert all(path.exists() for path in output_paths.values())
