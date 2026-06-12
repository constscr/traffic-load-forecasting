from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
from traffic_forecasting.visualization import (
    ABSOLUTE_ERROR_COLUMN,
    ACTUAL_COLUMN,
    PREDICTED_COLUMN,
    RESIDUAL_COLUMN,
    add_prediction_errors,
    extract_feature_importance,
    plot_actual_vs_predicted,
    plot_error_by_hour,
    plot_extended_ensemble_comparison,
    plot_feature_importance,
    plot_feature_set_comparison,
    plot_model_comparison,
    plot_residual_distribution,
    plot_traffic_volume_time_series,
    plot_true_vs_predicted,
    summarize_error_by_hour,
)


@pytest.fixture
def predictions() -> pd.DataFrame:
    periods = 24
    actual = np.linspace(1_000, 2_000, periods)
    return pd.DataFrame(
        {
            "model": ["catboost"] * periods,
            "split": ["validation"] * periods,
            DATETIME_COLUMN: pd.date_range("2026-01-01", periods=periods, freq="h"),
            ACTUAL_COLUMN: actual,
            PREDICTED_COLUMN: actual + np.sin(np.arange(periods)) * 50,
        }
    )


def _assert_plot_result(result: tuple[Figure, Axes], output_path: Path) -> None:
    figure, axis = result
    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert output_path.is_file()
    plt.close(figure)


def test_add_prediction_errors_preserves_input(predictions: pd.DataFrame) -> None:
    result = add_prediction_errors(predictions)

    assert RESIDUAL_COLUMN in result
    assert ABSOLUTE_ERROR_COLUMN in result
    assert RESIDUAL_COLUMN not in predictions
    assert np.allclose(result[ABSOLUTE_ERROR_COLUMN], result[RESIDUAL_COLUMN].abs())


def test_prediction_plots_return_figures_and_save_files(predictions, tmp_path) -> None:
    plot_functions = (
        plot_traffic_volume_time_series,
        plot_actual_vs_predicted,
        plot_residual_distribution,
        plot_true_vs_predicted,
        plot_error_by_hour,
    )

    for plot_function in plot_functions:
        output_path = tmp_path / f"{plot_function.__name__}.png"
        result = plot_function(
            predictions,
            model_name="catboost",
            output_path=output_path,
        )
        _assert_plot_result(result, output_path)


def test_prediction_plots_use_validation_rows_by_default(predictions) -> None:
    test_rows = predictions.assign(split="test", **{ACTUAL_COLUMN: 100_000})
    mixed_predictions = pd.concat([predictions, test_rows], ignore_index=True)

    hourly_error = summarize_error_by_hour(mixed_predictions, model_name="catboost")

    assert len(hourly_error) == 24
    assert hourly_error["mae"].max() < 100


def test_model_comparison_plot_uses_requested_metric(tmp_path) -> None:
    comparison = pd.DataFrame(
        {
            "model": ["catboost", "ridge"],
            "split": ["validation", "validation"],
            "rmse": [237.0, 414.0],
        }
    )
    output_path = tmp_path / "comparison.png"

    result = plot_model_comparison(comparison, output_path=output_path)

    _assert_plot_result(result, output_path)


def test_feature_set_comparison_plot_uses_validation_metrics(tmp_path) -> None:
    comparison = pd.DataFrame(
        {
            "feature_set_label": ["Temporal + calendar", "Full feature set"] * 2,
            "model": ["catboost", "catboost", "xgboost", "xgboost"],
            "split": ["validation"] * 4,
            "rmse": [400.0, 237.0, 420.0, 249.0],
        }
    )
    output_path = tmp_path / "feature_sets.png"

    result = plot_feature_set_comparison(comparison, output_path=output_path)

    _assert_plot_result(result, output_path)


def test_extended_ensemble_comparison_plot_highlights_voting(tmp_path) -> None:
    comparison = pd.DataFrame(
        {
            "model": ["catboost", "xgboost", "voting_regressor"],
            "model_group": [
                "strongest_individual",
                "strongest_individual",
                "extended_ensemble",
            ],
            "split": ["validation"] * 3,
            "rmse": [236.0, 249.0, 230.0],
        }
    )
    output_path = tmp_path / "extended_ensemble.png"

    result = plot_extended_ensemble_comparison(
        comparison,
        output_path=output_path,
    )

    _assert_plot_result(result, output_path)


def test_feature_importance_uses_transformed_feature_names(tmp_path) -> None:
    X = pd.DataFrame(
        {
            "numeric": np.arange(12, dtype=float),
            "category": ["clear", "rain"] * 6,
        }
    )
    y = pd.Series(np.arange(12, dtype=float))
    pipeline = Pipeline(
        [
            (
                "preprocessing",
                ColumnTransformer(
                    [
                        ("numeric", "passthrough", ["numeric"]),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["category"],
                        ),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(n_estimators=5, random_state=42),
            ),
        ]
    ).fit(X, y)
    output_path = tmp_path / "importance.png"

    importance = extract_feature_importance(pipeline)
    result = plot_feature_importance(pipeline, top_n=3, output_path=output_path)

    assert set(importance.columns) == {"feature", "importance"}
    assert len(importance) == 3
    assert importance["feature"].str.contains("numeric|category").all()
    _assert_plot_result(result, output_path)


def test_multiple_models_require_explicit_selection(predictions) -> None:
    mixed_predictions = pd.concat(
        [predictions, predictions.assign(model="ridge")],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="model_name is required"):
        plot_actual_vs_predicted(mixed_predictions)


def test_prediction_column_constants_match_stage_nine_schema() -> None:
    assert ACTUAL_COLUMN == f"actual_{TARGET_COLUMN}"
    assert PREDICTED_COLUMN == f"predicted_{TARGET_COLUMN}"
