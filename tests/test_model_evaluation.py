import numpy as np
import pandas as pd
import pytest
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
from traffic_forecasting.evaluation import (
    ABSOLUTE_ERROR_COLUMN,
    identify_large_errors,
    summarize_errors_by_hour,
    summarize_residuals,
)
from traffic_forecasting.features import get_feature_groups
from traffic_forecasting.pipeline import (
    build_locked_test_comparison,
    build_model_evaluation_candidates,
    extract_model_evaluation_feature_importance,
    run_model_evaluation_pipeline,
    save_model_evaluation_results,
    train_and_evaluate_locked_test_candidates,
)
from traffic_forecasting.preprocessing import split_chronologically


def _make_feature_data(periods: int = 72) -> pd.DataFrame:
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
    data["is_weekend"] = (data[DATETIME_COLUMN].dt.dayofweek >= 5).astype("int8")
    data["is_holiday"] = np.zeros(periods, dtype="int8")
    for offset, column in enumerate(feature_groups["cyclical"]):
        data[column] = np.sin(np.arange(periods) + offset)
    return data


def _lightweight_models() -> dict[str, RegressorMixin]:
    return {
        "catboost": RandomForestRegressor(n_estimators=5, random_state=42),
        "xgboost": Ridge(alpha=1.0),
        "random_forest": Ridge(alpha=2.0),
    }


def _feature_sets() -> dict[str, str]:
    return {
        "catboost": "temporal_calendar_lag",
        "xgboost": "full",
        "random_forest": "temporal_calendar_lag",
    }


def test_model_evaluation_candidates_are_fixed_before_locked_test_evaluation() -> None:
    candidates = build_model_evaluation_candidates(
        base_models=_lightweight_models(),
        base_feature_sets=_feature_sets(),
    )

    assert tuple(candidates) == ("catboost", "voting_regressor")


def test_model_evaluation_uses_train_validation_and_outputs_locked_test_rows() -> None:
    train, validation, test = split_chronologically(_make_feature_data())
    development = pd.concat([train, validation], ignore_index=True)

    metrics, predictions, fitted = train_and_evaluate_locked_test_candidates(
        development,
        test,
        base_models=_lightweight_models(),
        base_feature_sets=_feature_sets(),
    )

    assert set(metrics["split"]) == {"test"}
    assert set(predictions["split"]) == {"test"}
    assert set(metrics["training_data_scope"]) == {"train_validation"}
    assert set(predictions["training_data_scope"]) == {"train_validation"}
    assert not metrics["used_for_model_selection"].any()
    assert set(metrics["selection_basis"]) == {"completed_validation_experiments"}
    assert predictions.groupby("model").size().eq(len(test)).all()

    lag_columns = get_feature_groups()["lag"]
    scaler = (
        fitted["catboost"]
        .named_steps["preprocessing"]
        .named_transformers_["continuous"]
        .named_steps["scaler"]
    )
    assert scaler.mean_[0] == pytest.approx(development[lag_columns[0]].mean())
    assert scaler.mean_[0] != pytest.approx(test[lag_columns[0]].mean())

    prediction_matrix = predictions.pivot(
        index=DATETIME_COLUMN,
        columns="model",
        values=f"predicted_{TARGET_COLUMN}",
    )
    voting = fitted["voting_regressor"]
    base_predictions = np.column_stack(
        [
            estimator.predict(test.drop(columns=[DATETIME_COLUMN, TARGET_COLUMN]))
            for estimator in voting.estimators_
        ]
    )
    assert np.allclose(
        prediction_matrix["voting_regressor"],
        base_predictions.mean(axis=1),
    )


def test_model_evaluation_feature_importance_uses_supported_candidates_only() -> None:
    train, validation, test = split_chronologically(_make_feature_data())
    development = pd.concat([train, validation], ignore_index=True)

    _, _, fitted = train_and_evaluate_locked_test_candidates(
        development,
        test,
        base_models=_lightweight_models(),
        base_feature_sets=_feature_sets(),
    )

    importance = extract_model_evaluation_feature_importance(fitted)

    assert set(importance.columns) == {"model", "feature", "importance"}
    assert not importance.empty
    assert set(importance["model"]) == {"catboost"}


def test_locked_test_comparison_never_ranks_or_selects_by_test_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["catboost", "voting_regressor"],
            "model_group": ["strongest_individual", "extended_ensemble"],
            "selection_basis": ["completed_validation_experiments"] * 2,
            "split": ["test"] * 2,
            "used_for_model_selection": [False, False],
            "mae": [200.0, 190.0],
            "rmse": [300.0, 290.0],
            "mape": [8.0, 7.5],
            "r2": [0.96, 0.97],
        }
    )

    comparison = build_locked_test_comparison(metrics)

    assert comparison["model"].tolist() == ["catboost", "voting_regressor"]
    assert "test_rank" not in comparison
    assert not comparison["used_for_model_selection"].any()


def test_error_analysis_summarizes_test_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["catboost"] * 20,
            "split": ["test"] * 20,
            DATETIME_COLUMN: pd.date_range("2026-01-01", periods=20, freq="h"),
            f"actual_{TARGET_COLUMN}": np.arange(20, dtype=float) + 100,
            f"predicted_{TARGET_COLUMN}": np.arange(20, dtype=float) + 95,
        }
    )

    residual_summary = summarize_residuals(predictions)
    hourly_errors = summarize_errors_by_hour(predictions)
    large_errors = identify_large_errors(predictions, quantile=0.9)

    assert residual_summary.loc[0, "residual_mean"] == pytest.approx(5.0)
    assert residual_summary.loc[0, "mean_absolute_error"] == pytest.approx(5.0)
    assert hourly_errors["mae"].eq(5.0).all()
    assert not large_errors.empty
    assert ABSOLUTE_ERROR_COLUMN in large_errors
    assert large_errors["large_error_quantile"].eq(0.9).all()


def test_identify_large_errors_sorts_by_split_model_and_error() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["b", "a", "a", "b"] * 2,
            "split": ["validation"] * 4 + ["test"] * 4,
            f"actual_{TARGET_COLUMN}": [100.0] * 8,
            f"predicted_{TARGET_COLUMN}": [
                90.0,
                70.0,
                95.0,
                80.0,
                60.0,
                85.0,
                50.0,
                75.0,
            ],
        }
    )

    large_errors = identify_large_errors(predictions, quantile=0.5)

    expected = large_errors.sort_values(
        ["split", "model", ABSOLUTE_ERROR_COLUMN],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(large_errors, expected)


def test_save_model_evaluation_results_writes_all_tables(tmp_path) -> None:
    frame = pd.DataFrame({"value": [1.0]})
    paths = {
        "metrics_path": tmp_path / "metrics.csv",
        "comparison_path": tmp_path / "comparison.csv",
        "predictions_path": tmp_path / "predictions.csv",
        "residual_summary_path": tmp_path / "residuals.csv",
        "hourly_errors_path": tmp_path / "hourly.csv",
        "large_errors_path": tmp_path / "large.csv",
        "feature_importance_path": tmp_path / "importance.csv",
    }

    saved = save_model_evaluation_results(
        frame,
        frame,
        frame,
        frame,
        frame,
        frame,
        frame,
        **paths,
    )

    assert set(saved) == {
        "test_metrics",
        "test_comparison",
        "test_predictions",
        "residual_summary",
        "error_by_hour",
        "large_errors",
        "feature_importance",
    }
    assert all(path.is_file() for path in saved.values())


def test_model_evaluation_feature_importance_is_empty_without_supported_candidate(
    tmp_path,
) -> None:
    feature_data_path = tmp_path / "features.csv"
    _make_feature_data().to_csv(feature_data_path, index=False)
    output_paths = {
        "metrics_path": tmp_path / "metrics.csv",
        "comparison_path": tmp_path / "comparison.csv",
        "predictions_path": tmp_path / "predictions.csv",
        "residual_summary_path": tmp_path / "residuals.csv",
        "hourly_errors_path": tmp_path / "hourly.csv",
        "large_errors_path": tmp_path / "large.csv",
        "feature_importance_path": tmp_path / "importance.csv",
    }

    _, _, _, analyses, _, saved = run_model_evaluation_pipeline(
        feature_data_path=feature_data_path,
        candidate_names=("voting_regressor",),
        base_models=_lightweight_models(),
        base_feature_sets=_feature_sets(),
        **output_paths,
    )

    assert analyses["feature_importance"].empty
    assert list(analyses["feature_importance"].columns) == [
        "model",
        "feature",
        "importance",
    ]
    assert saved["feature_importance"].is_file()


def test_model_evaluation_pipeline_reads_processed_data_and_saves_outputs(tmp_path) -> None:
    feature_data_path = tmp_path / "features.csv"
    _make_feature_data().to_csv(feature_data_path, index=False)
    output_paths = {
        "metrics_path": tmp_path / "metrics.csv",
        "comparison_path": tmp_path / "comparison.csv",
        "predictions_path": tmp_path / "predictions.csv",
        "residual_summary_path": tmp_path / "residuals.csv",
        "hourly_errors_path": tmp_path / "hourly.csv",
        "large_errors_path": tmp_path / "large.csv",
        "feature_importance_path": tmp_path / "importance.csv",
    }

    metrics, comparison, predictions, analyses, _, saved = run_model_evaluation_pipeline(
        feature_data_path=feature_data_path,
        base_models=_lightweight_models(),
        base_feature_sets=_feature_sets(),
        **output_paths,
    )

    assert set(metrics["training_data_scope"]) == {"train_validation"}
    assert set(predictions["training_data_scope"]) == {"train_validation"}
    assert set(metrics["split"]) == {"test"}
    assert set(comparison["split"]) == {"test"}
    assert set(predictions["split"]) == {"test"}
    assert set(analyses) == {
        "residual_summary",
        "error_by_hour",
        "large_errors",
        "feature_importance",
    }
    assert set(analyses["feature_importance"].columns) == {
        "model",
        "feature",
        "importance",
    }
    assert not analyses["feature_importance"].empty
    assert saved["feature_importance"].is_file()
    assert all(path.is_file() for path in saved.values())
