import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
from traffic_forecasting.features import get_feature_groups
from traffic_forecasting.models import (
    build_voting_regressor,
    get_extended_ensemble_base_model_registry,
)
from traffic_forecasting.pipeline import (
    build_extended_ensemble_candidates,
    build_extended_ensemble_comparison,
    run_extended_ensemble_pipeline,
    save_extended_ensemble_results,
    train_and_evaluate_extended_ensembles,
)
from traffic_forecasting.preprocessing import (
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


def _lightweight_models() -> dict[str, Ridge]:
    return {
        "ridge_low": Ridge(alpha=0.5),
        "ridge_high": Ridge(alpha=2.0),
    }


def _lightweight_feature_sets() -> dict[str, str]:
    return {
        "ridge_low": "full",
        "ridge_high": "temporal_calendar_lag",
    }


def test_extended_ensemble_registry_contains_strongest_models() -> None:
    registry = get_extended_ensemble_base_model_registry()

    assert tuple(registry) == ("catboost", "xgboost", "random_forest")


def test_voting_regressor_requires_multiple_estimators() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_voting_regressor([("ridge", Ridge())])


def test_extended_candidates_use_pipeline_base_estimators() -> None:
    candidates = build_extended_ensemble_candidates(
        base_models=_lightweight_models(),
        base_feature_sets=_lightweight_feature_sets(),
    )

    assert tuple(candidates) == ("ridge_low", "ridge_high", "voting_regressor")
    assert isinstance(candidates["ridge_low"], Pipeline)
    assert isinstance(candidates["voting_regressor"], VotingRegressor)
    assert all(
        isinstance(estimator, Pipeline)
        for _, estimator in candidates["voting_regressor"].estimators
    )


def test_extended_candidates_require_matching_model_and_feature_set_names() -> None:
    with pytest.raises(ValueError, match="must match"):
        build_extended_ensemble_candidates(
            base_models={"ridge_low": Ridge(), "ridge_high": Ridge()},
            base_feature_sets={"ridge_low": "full"},
        )


def test_extended_ensemble_uses_train_only_preprocessing_and_validation_only() -> None:
    train, validation, _ = split_chronologically(_make_feature_data())

    metrics, predictions, fitted = train_and_evaluate_extended_ensembles(
        train,
        validation,
        base_models=_lightweight_models(),
        base_feature_sets=_lightweight_feature_sets(),
    )

    assert set(metrics["split"]) == {"validation"}
    assert set(predictions["split"]) == {"validation"}
    assert len(metrics) == 3
    assert predictions.groupby("model").size().nunique() == 1
    assert predictions.groupby("model").size().iloc[0] == len(validation)

    assert "feature_set_strategy" in metrics.columns
    assert "feature_set_strategy" in predictions.columns

    strategy_by_model = dict(zip(metrics["model"], metrics["feature_set_strategy"], strict=True))
    assert strategy_by_model["ridge_low"] == "full"
    assert strategy_by_model["ridge_high"] == "temporal_calendar_lag"
    assert strategy_by_model["voting_regressor"] == "mixed_best_feature_sets"

    prediction_matrix = predictions.pivot(
        index=DATETIME_COLUMN,
        columns="model",
        values=f"predicted_{TARGET_COLUMN}",
    )
    expected_voting = prediction_matrix[["ridge_low", "ridge_high"]].mean(axis=1)
    assert np.allclose(prediction_matrix["voting_regressor"], expected_voting)

    voting = fitted["voting_regressor"]
    check_is_fitted(voting)
    assert all(isinstance(estimator, Pipeline) for estimator in voting.estimators_)

    full_pipeline = voting.estimators_[0]
    continuous_columns = get_model_feature_groups()["continuous_numeric"]
    temp_index = continuous_columns.index("temp")
    scaler = (
        full_pipeline.named_steps["preprocessing"]
        .named_transformers_["continuous"]
        .named_steps["scaler"]
    )
    assert scaler.mean_[temp_index] == pytest.approx(train["temp"].mean())
    assert scaler.mean_[temp_index] != pytest.approx(validation["temp"].mean())


def test_extended_ensemble_requires_datetime_and_target_columns() -> None:
    train, validation, _ = split_chronologically(_make_feature_data())
    validation = validation.drop(columns=[DATETIME_COLUMN])

    with pytest.raises(ValueError, match="Missing required columns in validation data"):
        train_and_evaluate_extended_ensembles(
            train,
            validation,
            base_models=_lightweight_models(),
            base_feature_sets=_lightweight_feature_sets(),
        )


def test_extended_comparison_uses_best_individual_reference() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["catboost", "xgboost", "voting_regressor"],
            "model_group": [
                "strongest_individual",
                "strongest_individual",
                "extended_ensemble",
            ],
            "feature_set_strategy": [
                "temporal_calendar_lag",
                "full",
                "mixed_best_feature_sets",
            ],
            "split": ["validation"] * 3,
            "mae": [150.0, 160.0, 145.0],
            "rmse": [236.0, 249.0, 230.0],
            "mape": [6.0, 6.5, 5.8],
            "r2": [0.98, 0.97, 0.99],
        }
    )

    comparison = build_extended_ensemble_comparison(metrics)
    voting_row = comparison.loc[comparison["model"] == "voting_regressor"].iloc[0]

    assert comparison["validation_rank"].tolist() == [1, 2, 3]
    assert voting_row["best_individual_rmse"] == pytest.approx(236.0)
    assert voting_row["rmse_improvement_vs_best_individual_pct"] > 0


def test_save_extended_ensemble_results_writes_all_outputs(tmp_path) -> None:
    metrics = pd.DataFrame({"model": ["voting_regressor"], "rmse": [230.0]})
    comparison = metrics.assign(validation_rank=1)
    predictions = pd.DataFrame({"model": ["voting_regressor"], "split": ["validation"]})
    paths = (
        tmp_path / "metrics.csv",
        tmp_path / "comparison.csv",
        tmp_path / "predictions.csv",
    )

    saved_paths = save_extended_ensemble_results(
        metrics,
        comparison,
        predictions,
        metrics_path=paths[0],
        comparison_path=paths[1],
        predictions_path=paths[2],
    )

    assert saved_paths == paths
    assert all(path.is_file() for path in paths)


def test_extended_pipeline_never_outputs_test_rows(tmp_path) -> None:
    feature_data_path = tmp_path / "features.csv"
    _make_feature_data().to_csv(feature_data_path, index=False)

    metrics, comparison, predictions, output_paths = run_extended_ensemble_pipeline(
        feature_data_path=feature_data_path,
        base_models=_lightweight_models(),
        base_feature_sets=_lightweight_feature_sets(),
        metrics_path=tmp_path / "metrics.csv",
        comparison_path=tmp_path / "comparison.csv",
        predictions_path=tmp_path / "predictions.csv",
    )

    assert set(metrics["split"]) == {"validation"}
    assert set(comparison["split"]) == {"validation"}
    assert set(predictions["split"]) == {"validation"}
    assert all(path.is_file() for path in output_paths)
