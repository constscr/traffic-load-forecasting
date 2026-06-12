import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
from traffic_forecasting.feature_sets import get_feature_set_columns
from traffic_forecasting.features import get_feature_groups
from traffic_forecasting.pipeline import (
    build_feature_set_comparison,
    run_feature_set_experiment_pipeline,
    save_feature_set_experiment_results,
    train_and_evaluate_feature_sets,
)
from traffic_forecasting.preprocessing import (
    build_preprocessor,
    get_model_feature_groups,
    split_chronologically,
)


def _make_feature_data(periods: int = 60) -> pd.DataFrame:
    feature_groups = get_feature_groups()
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: pd.date_range("2026-01-01", periods=periods, freq="h"),
            TARGET_COLUMN: np.linspace(1_000, 2_000, periods),
        }
    )
    for offset, column in enumerate(feature_groups["weather_numeric"]):
        data[column] = np.arange(periods, dtype=float) + offset
    for offset, column in enumerate(feature_groups["lag"]):
        data[column] = np.arange(periods, dtype=float) + offset + 10
    for offset, column in enumerate(feature_groups["rolling"]):
        data[column] = np.arange(periods, dtype=float) + offset + 20

    data["weather_main"] = ["Clear"] * periods
    data["weather_description"] = ["clear sky"] * periods
    data["is_weekend"] = np.zeros(periods, dtype="int8")
    data["is_holiday"] = np.zeros(periods, dtype="int8")
    for offset, column in enumerate(feature_groups["cyclical"]):
        data[column] = np.sin(np.arange(periods) + offset)
    return data


def test_scenario_preprocessor_contains_only_selected_transformers() -> None:
    feature_columns = get_feature_set_columns("temporal_calendar")

    preprocessor = build_preprocessor(feature_columns)

    assert [name for name, _, _ in preprocessor.transformers] == ["binary", "cyclical"]


def test_weather_free_scenario_fits_without_weather_columns() -> None:
    feature_columns = get_feature_set_columns("temporal_calendar_lag")
    data = _make_feature_data().loc[:, feature_columns].copy()

    transformed = build_preprocessor(feature_columns).fit_transform(data)

    assert transformed.shape[0] == len(data)
    assert "weather_main" not in data.columns
    assert "weather_description" not in data.columns


def test_feature_set_experiments_share_validation_rows_and_use_train_only() -> None:
    train, validation, _ = split_chronologically(_make_feature_data())
    scenarios = ("temporal_calendar", "temporal_calendar_weather")

    metrics, predictions, fitted = train_and_evaluate_feature_sets(
        train,
        validation,
        scenario_names=scenarios,
        model_registry={"ridge": Ridge()},
    )

    assert len(metrics) == len(scenarios)
    assert set(metrics["split"]) == {"validation"}
    assert set(predictions["split"]) == {"validation"}
    assert predictions.groupby(["feature_set", "model"]).size().nunique() == 1
    assert predictions.groupby(["feature_set", "model"]).size().iloc[0] == len(validation)

    weather_pipeline = fitted[("temporal_calendar_weather", "ridge")]
    check_is_fitted(weather_pipeline)
    continuous_columns = get_model_feature_groups()["continuous_numeric"]
    scenario_columns = get_feature_set_columns("temporal_calendar_weather")
    scenario_continuous = [column for column in continuous_columns if column in scenario_columns]
    temp_index = scenario_continuous.index("temp")
    scaler = (
        weather_pipeline.named_steps["preprocessing"]
        .named_transformers_["continuous"]
        .named_steps["scaler"]
    )
    assert scaler.mean_[temp_index] == pytest.approx(train["temp"].mean())
    assert scaler.mean_[temp_index] != pytest.approx(validation["temp"].mean())


def test_feature_set_experiments_require_datetime_and_target_columns() -> None:
    train, validation, _ = split_chronologically(_make_feature_data())
    validation = validation.drop(columns=[DATETIME_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns in validation data"):
        train_and_evaluate_feature_sets(
            train,
            validation,
            scenario_names=("temporal_calendar",),
            model_registry={"ridge": Ridge()},
        )


def test_feature_set_comparison_ranks_within_each_model() -> None:
    metrics = pd.DataFrame(
        {
            "feature_set": ["temporal_calendar", "full"] * 2,
            "feature_set_label": ["Temporal + calendar", "Full"] * 2,
            "feature_count": [8, 25] * 2,
            "model": ["catboost", "catboost", "xgboost", "xgboost"],
            "split": ["validation"] * 4,
            "mae": [300.0, 150.0, 320.0, 160.0],
            "rmse": [400.0, 240.0, 420.0, 250.0],
            "mape": [12.0, 6.0, 13.0, 6.5],
            "r2": [0.90, 0.98, 0.89, 0.97],
        }
    )

    comparison = build_feature_set_comparison(metrics)

    best_rows = comparison.loc[comparison["rank_within_model"] == 1]
    assert set(best_rows["feature_set"]) == {"full"}
    assert (best_rows["rmse_improvement_vs_temporal_calendar_pct"] > 0).all()


def test_save_feature_set_results_writes_all_outputs(tmp_path) -> None:
    metrics = pd.DataFrame({"feature_set": ["full"], "rmse": [240.0]})
    comparison = metrics.assign(rank_within_model=1)
    predictions = pd.DataFrame({"feature_set": ["full"], "split": ["validation"]})
    paths = (
        tmp_path / "metrics.csv",
        tmp_path / "comparison.csv",
        tmp_path / "predictions.csv",
    )

    saved_paths = save_feature_set_experiment_results(
        metrics,
        comparison,
        predictions,
        metrics_path=paths[0],
        comparison_path=paths[1],
        predictions_path=paths[2],
    )

    assert saved_paths == paths
    assert all(path.is_file() for path in paths)


def test_feature_set_pipeline_uses_processed_data_and_never_outputs_test(
    tmp_path,
) -> None:
    feature_data_path = tmp_path / "features.csv"
    _make_feature_data().to_csv(feature_data_path, index=False)

    metrics, comparison, predictions, output_paths = run_feature_set_experiment_pipeline(
        feature_data_path=feature_data_path,
        scenario_names=("temporal_calendar",),
        model_registry={"ridge": Ridge()},
        metrics_path=tmp_path / "metrics.csv",
        comparison_path=tmp_path / "comparison.csv",
        predictions_path=tmp_path / "predictions.csv",
    )

    assert set(metrics["split"]) == {"validation"}
    assert set(comparison["split"]) == {"validation"}
    assert set(predictions["split"]) == {"validation"}
    assert all(path.is_file() for path in output_paths)
