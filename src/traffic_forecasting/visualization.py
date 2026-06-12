from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.pipeline import Pipeline

from traffic_forecasting.config import DATETIME_COLUMN, TARGET_COLUMN
from traffic_forecasting.feature_sets import get_feature_set_scenarios

ACTUAL_COLUMN = f"actual_{TARGET_COLUMN}"
PREDICTED_COLUMN = f"predicted_{TARGET_COLUMN}"
RESIDUAL_COLUMN = "residual"
ABSOLUTE_ERROR_COLUMN = "absolute_error"
DEFAULT_FIGURE_SIZE = (12, 6)

# Unified visualization color palette
PASTEL_BLUE = "#7FB3D5"
PASTEL_PEACH = "#F4B6A6"
PASTEL_LAVENDER = "#CDB4DB"
PASTEL_MINT = "#B7E4C7"
NEUTRAL_DARK_GRAY = "#3D4852"

ACTUAL_LINE_COLOR = PASTEL_BLUE
PREDICTED_LINE_COLOR = PASTEL_PEACH
SCATTER_COLOR = PASTEL_BLUE
RESIDUAL_COLOR = PASTEL_LAVENDER
HOURLY_ERROR_COLOR = PASTEL_MINT

MODEL_COMPARISON_BASE_COLOR = PASTEL_BLUE
MODEL_COMPARISON_HIGHLIGHT_COLOR = PASTEL_PEACH

FEATURE_IMPORTANCE_BASE_COLOR = PASTEL_BLUE
FEATURE_IMPORTANCE_HIGHLIGHT_COLOR = PASTEL_LAVENDER
FEATURE_IMPORTANCE_HIGHLIGHT_TOP_N = 3

REFERENCE_LINE_COLOR = NEUTRAL_DARK_GRAY
GRID_ALPHA = 0.25


def _save_figure(figure: Figure, output_path: str | Path | None) -> None:
    if output_path is None:
        return

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")


def _select_prediction_rows(
    predictions: pd.DataFrame,
    *,
    model_name: str | None,
    split: str,
) -> pd.DataFrame:
    required_columns = {
        "model",
        "split",
        DATETIME_COLUMN,
        ACTUAL_COLUMN,
        PREDICTED_COLUMN,
    }
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    selected = predictions.loc[predictions["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No prediction rows found for split: {split}")

    available_models = selected["model"].dropna().unique().tolist()
    if model_name is None:
        if len(available_models) != 1:
            raise ValueError("model_name is required when predictions contain multiple models.")
        model_name = str(available_models[0])

    selected = selected.loc[selected["model"] == model_name].copy()
    if selected.empty:
        raise ValueError(f"No prediction rows found for model: {model_name}")

    selected[DATETIME_COLUMN] = pd.to_datetime(selected[DATETIME_COLUMN], errors="raise")
    return selected.sort_values(DATETIME_COLUMN).reset_index(drop=True)


def add_prediction_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add signed residual and absolute error columns to prediction rows."""
    required_columns = {ACTUAL_COLUMN, PREDICTED_COLUMN}
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    result = predictions.copy()
    result[RESIDUAL_COLUMN] = result[ACTUAL_COLUMN] - result[PREDICTED_COLUMN]
    result[ABSOLUTE_ERROR_COLUMN] = result[RESIDUAL_COLUMN].abs()
    return result


def plot_traffic_volume_time_series(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    split: str = "validation",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot observed traffic volume over the selected prediction period."""
    selected = _select_prediction_rows(predictions, model_name=model_name, split=split)
    figure, axis = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    axis.plot(
        selected[DATETIME_COLUMN],
        selected[ACTUAL_COLUMN],
        color=ACTUAL_LINE_COLOR,
        linewidth=1.2,
    )
    axis.set(
        title=f"Observed Traffic Volume ({split.title()} Period)",
        xlabel="Date and time",
        ylabel="Traffic volume",
    )
    axis.grid(alpha=GRID_ALPHA)
    figure.autofmt_xdate()
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    split: str = "validation",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot actual and predicted traffic volume in chronological order."""
    selected = _select_prediction_rows(predictions, model_name=model_name, split=split)
    figure, axis = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    axis.plot(
        selected[DATETIME_COLUMN],
        selected[ACTUAL_COLUMN],
        label="Actual",
        color=ACTUAL_LINE_COLOR,
        linewidth=1.3,
    )
    axis.plot(
        selected[DATETIME_COLUMN],
        selected[PREDICTED_COLUMN],
        label="Predicted",
        color=PREDICTED_LINE_COLOR,
        linewidth=1.1,
        alpha=0.9,
    )
    axis.set(
        title=f"Actual vs Predicted Traffic Volume: {model_name or selected['model'].iloc[0]}",
        xlabel="Date and time",
        ylabel="Traffic volume",
    )
    axis.legend()
    axis.grid(alpha=GRID_ALPHA)
    figure.autofmt_xdate()
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_residual_distribution(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    split: str = "validation",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot the distribution of actual-minus-predicted residuals."""
    selected = add_prediction_errors(
        _select_prediction_rows(predictions, model_name=model_name, split=split)
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(
        selected[RESIDUAL_COLUMN],
        bins=40,
        kde=True,
        ax=axis,
        color=RESIDUAL_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    axis.axvline(0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.2)
    axis.set(
        title=f"Residual Distribution ({split.title()} Predictions)",
        xlabel="Residual: actual - predicted",
        ylabel="Count",
    )
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_true_vs_predicted(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    split: str = "validation",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot predicted values against observed traffic volume."""
    selected = _select_prediction_rows(predictions, model_name=model_name, split=split)
    lower_bound = min(selected[ACTUAL_COLUMN].min(), selected[PREDICTED_COLUMN].min())
    upper_bound = max(selected[ACTUAL_COLUMN].max(), selected[PREDICTED_COLUMN].max())

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        selected[ACTUAL_COLUMN],
        selected[PREDICTED_COLUMN],
        alpha=GRID_ALPHA,
        s=18,
        color=SCATTER_COLOR,
        edgecolors="none",
    )
    axis.plot(
        [lower_bound, upper_bound],
        [lower_bound, upper_bound],
        color=REFERENCE_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Ideal prediction",
    )
    axis.set(
        title=f"Observed vs Predicted Traffic Volume ({split.title()})",
        xlabel="Observed traffic volume",
        ylabel="Predicted traffic volume",
    )
    axis.legend()
    axis.grid(alpha=GRID_ALPHA)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def summarize_error_by_hour(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    split: str = "validation",
) -> pd.DataFrame:
    """Calculate mean absolute prediction error for each hour of day."""
    selected = add_prediction_errors(
        _select_prediction_rows(predictions, model_name=model_name, split=split)
    )
    selected["hour"] = selected[DATETIME_COLUMN].dt.hour
    return (
        selected.groupby("hour", as_index=False)[ABSOLUTE_ERROR_COLUMN]
        .mean()
        .rename(columns={ABSOLUTE_ERROR_COLUMN: "mae"})
    )


def plot_error_by_hour(
    predictions: pd.DataFrame,
    *,
    model_name: str | None = None,
    split: str = "validation",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot mean absolute validation error by hour of day."""
    hourly_error = summarize_error_by_hour(
        predictions,
        model_name=model_name,
        split=split,
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=hourly_error,
        x="hour",
        y="mae",
        ax=axis,
        color=HOURLY_ERROR_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set(
        title=f"Mean Absolute Error by Hour ({split.title()})",
        xlabel="Hour of day",
        ylabel="Mean absolute error",
    )
    axis.grid(axis="y", alpha=GRID_ALPHA)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_model_comparison(
    comparison: pd.DataFrame,
    *,
    metric: str = "rmse",
    split: str = "validation",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot model performance using validation comparison metrics."""
    required_columns = {"model", "split", metric}
    missing_columns = sorted(required_columns - set(comparison.columns))
    if missing_columns:
        raise ValueError(f"Missing comparison columns: {missing_columns}")

    selected = comparison.loc[comparison["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No comparison rows found for split: {split}")

    higher_is_better = metric.lower() in {"r2"}
    best_index = selected[metric].idxmax() if higher_is_better else selected[metric].idxmin()

    # For horizontal bar charts, the last row is displayed at the top.
    selected = selected.sort_values(metric, ascending=higher_is_better)

    figure, axis = plt.subplots(figsize=(10, 7))
    colors = [
        MODEL_COMPARISON_HIGHLIGHT_COLOR if index == best_index else MODEL_COMPARISON_BASE_COLOR
        for index in selected.index
    ]
    axis.barh(selected["model"], selected[metric], color=colors, edgecolor="white", linewidth=0.6)
    axis.set(
        title=f"Model Comparison by {metric.upper()} ({split.title()})",
        xlabel=metric.upper(),
        ylabel="Model",
    )
    axis.grid(axis="x", alpha=GRID_ALPHA)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def extract_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    """Extract transformed feature names and importance from a fitted model pipeline."""
    if not isinstance(model_pipeline, Pipeline):
        raise TypeError("model_pipeline must be an sklearn Pipeline.")
    required_steps = {"preprocessing", "model"}
    if not required_steps.issubset(model_pipeline.named_steps):
        raise ValueError("Pipeline must contain preprocessing and model steps.")

    preprocessor = model_pipeline.named_steps["preprocessing"]
    model = model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_feature_importance"):
        importances = np.asarray(model.get_feature_importance(), dtype=float)
    else:
        raise ValueError("The fitted model does not expose feature importance.")

    if len(feature_names) != len(importances):
        raise ValueError("Transformed feature names and importance values have different lengths.")

    clean_names = [name.split("__", maxsplit=1)[-1] for name in feature_names]
    return (
        pd.DataFrame({"feature": clean_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    model_pipeline: Pipeline,
    *,
    top_n: int = 20,
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot the most important transformed features of a fitted model pipeline."""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    importance = extract_feature_importance(model_pipeline).head(top_n).copy()
    highlighted_count = min(FEATURE_IMPORTANCE_HIGHLIGHT_TOP_N, len(importance))

    importance["color"] = FEATURE_IMPORTANCE_BASE_COLOR
    importance.loc[
        importance.index[:highlighted_count],
        "color",
    ] = FEATURE_IMPORTANCE_HIGHLIGHT_COLOR

    # For horizontal bar charts, sort ascending so the most important features appear at the top.
    importance = importance.sort_values("importance")

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(
        importance["feature"],
        importance["importance"],
        color=importance["color"],
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set(
        title=f"Top {min(top_n, len(importance))} Model Feature Importances",
        xlabel="Feature importance",
        ylabel="Transformed feature",
    )
    axis.grid(axis="x", alpha=GRID_ALPHA)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_feature_set_comparison(
    comparison: pd.DataFrame,
    *,
    metric: str = "rmse",
    split: str = "validation",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare feature scenarios across selected models on validation metrics."""
    required_columns = {"feature_set_label", "model", "split", metric}
    missing_columns = sorted(required_columns - set(comparison.columns))
    if missing_columns:
        raise ValueError(f"Missing feature set comparison columns: {missing_columns}")

    selected = comparison.loc[comparison["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No feature set comparison rows found for split: {split}")

    scenario_order = [
        scenario.label
        for scenario in get_feature_set_scenarios().values()
        if scenario.label in set(selected["feature_set_label"])
    ]

    default_model_order = ("catboost", "xgboost", "random_forest")
    available_models = set(selected["model"])

    model_order = [model for model in default_model_order if model in available_models]
    model_order.extend(sorted(available_models - set(default_model_order)))

    figure, axis = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=selected,
        x="feature_set_label",
        y=metric,
        hue="model",
        order=scenario_order,
        hue_order=model_order,
        ax=axis,
        palette=(PASTEL_BLUE, PASTEL_PEACH, PASTEL_LAVENDER),
        saturation=1.0,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set(
        title=f"Feature Set Comparison by {metric.upper()} ({split.title()})",
        xlabel="Feature set scenario",
        ylabel=metric.upper(),
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=GRID_ALPHA)
    axis.legend(title="Model")
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_extended_ensemble_comparison(
    comparison: pd.DataFrame,
    *,
    metric: str = "rmse",
    split: str = "validation",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare voting with the strongest individual validation models."""
    required_columns = {"model", "model_group", "split", metric}
    missing_columns = sorted(required_columns - set(comparison.columns))
    if missing_columns:
        raise ValueError(f"Missing extended ensemble comparison columns: {missing_columns}")

    selected = comparison.loc[comparison["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No extended ensemble rows found for split: {split}")
    selected = selected.sort_values(metric, ascending=False)
    colors = [
        (
            MODEL_COMPARISON_HIGHLIGHT_COLOR
            if group == "extended_ensemble"
            else MODEL_COMPARISON_BASE_COLOR
        )
        for group in selected["model_group"]
    ]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(
        selected["model"],
        selected[metric],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set(
        title=f"Extended Ensemble Comparison by {metric.upper()} ({split.title()})",
        xlabel=metric.upper(),
        ylabel="Model",
    )
    axis.grid(axis="x", alpha=GRID_ALPHA)
    axis.legend(
        handles=[
            Patch(
                facecolor=MODEL_COMPARISON_BASE_COLOR,
                edgecolor="white",
                label="Strongest individual",
            ),
            Patch(
                facecolor=MODEL_COMPARISON_HIGHLIGHT_COLOR,
                edgecolor="white",
                label="Extended ensemble",
            ),
        ],
        title="Model group",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis
