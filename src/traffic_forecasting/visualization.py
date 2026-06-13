from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.pipeline import Pipeline

from traffic_forecasting.config import DATETIME_COLUMN
from traffic_forecasting.evaluation import (
    ABSOLUTE_ERROR_COLUMN,
    ACTUAL_COLUMN,
    PREDICTED_COLUMN,
    RESIDUAL_COLUMN,
    add_prediction_errors,
    extract_feature_importance,
)
from traffic_forecasting.feature_sets import get_feature_set_scenarios

DEFAULT_FIGURE_SIZE = (12, 6)

# Unified visualization color palette
PASTEL_BLUE = "#6FA8CF"
PASTEL_PEACH = "#E69F8A"
PASTEL_LAVENDER = "#A98CCB"
PASTEL_MINT = "#8FCFB2"
NEUTRAL_DARK_GRAY = "#3D4852"

ACTUAL_LINE_COLOR = NEUTRAL_DARK_GRAY
PREDICTED_LINE_COLOR = PASTEL_PEACH
SCATTER_COLOR = PASTEL_BLUE
RESIDUAL_COLOR = PASTEL_LAVENDER
HOURLY_ERROR_COLOR = PASTEL_MINT

MODEL_LINE_COLORS = (
    PASTEL_BLUE,
    PASTEL_PEACH,
    PASTEL_LAVENDER,
    PASTEL_MINT,
    NEUTRAL_DARK_GRAY,
)

MODEL_COMPARISON_BASE_COLOR = PASTEL_BLUE
MODEL_COMPARISON_HIGHLIGHT_COLOR = PASTEL_PEACH

FEATURE_IMPORTANCE_BASE_COLOR = PASTEL_BLUE
FEATURE_IMPORTANCE_HIGHLIGHT_COLOR = PASTEL_LAVENDER
FEATURE_IMPORTANCE_HIGHLIGHT_TOP_N = 3

REFERENCE_LINE_COLOR = NEUTRAL_DARK_GRAY
GRID_ALPHA = 0.25

MODEL_DISPLAY_NAMES = {
    "catboost": "CatBoostRegressor",
    "xgboost": "XGBRegressor",
    "random_forest": "RandomForestRegressor",
    "voting_regressor": "VotingRegressor",
}

MODEL_GROUP_DISPLAY_NAMES = {
    "strongest_individual": "Сильнейшая индивидуальная модель",
    "extended_ensemble": "Расширенный ансамбль",
}

SPLIT_DISPLAY_NAMES = {
    "train": "обучающая выборка",
    "validation": "валидационная выборка",
    "test": "тестовая выборка",
}


def _display_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _display_model_group(group_name: str) -> str:
    return MODEL_GROUP_DISPLAY_NAMES.get(group_name, group_name)


def _display_split_name(split: str) -> str:
    return SPLIT_DISPLAY_NAMES.get(split, split)


def _build_label_palette(labels: list[str]) -> dict[str, str]:
    """Build a deterministic color mapping for categorical plot labels."""
    return {
        label: MODEL_LINE_COLORS[index % len(MODEL_LINE_COLORS)]
        for index, label in enumerate(labels)
    }


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
        title=(f"Динамика фактической транспортной нагрузки ({_display_split_name(split)})"),
        xlabel="Дата и время",
        ylabel="Интенсивность транспортного потока",
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
        label="Фактические значения",
        color=ACTUAL_LINE_COLOR,
        linewidth=1.3,
    )
    axis.plot(
        selected[DATETIME_COLUMN],
        selected[PREDICTED_COLUMN],
        label="Прогнозные значения",
        color=PREDICTED_LINE_COLOR,
        linewidth=1.1,
        alpha=0.9,
    )
    displayed_model_name = _display_model_name(model_name or str(selected["model"].iloc[0]))
    axis.set(
        title=(f"Фактические и прогнозные значения транспортной нагрузки: {displayed_model_name}"),
        xlabel="Дата и время",
        ylabel="Интенсивность транспортного потока",
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
        title=(f"Распределение остатков прогнозирования ({_display_split_name(split)})"),
        xlabel="Остаток прогноза: фактическое значение − прогноз",
        ylabel="Количество наблюдений",
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
        label="Идеальный прогноз",
    )
    axis.set(
        title=(f"Сопоставление фактических и прогнозных значений ({_display_split_name(split)})"),
        xlabel="Фактическая транспортная нагрузка",
        ylabel="Прогнозная транспортная нагрузка",
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
        title=(f"Средняя абсолютная ошибка по часам суток ({_display_split_name(split)})"),
        xlabel="Час суток",
        ylabel="Средняя абсолютная ошибка",
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
    selected["model_display"] = selected["model"].map(_display_model_name)

    figure, axis = plt.subplots(figsize=(10, 7))
    colors = [
        MODEL_COMPARISON_HIGHLIGHT_COLOR if index == best_index else MODEL_COMPARISON_BASE_COLOR
        for index in selected.index
    ]
    axis.barh(
        selected["model_display"],
        selected[metric],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set(
        title=(f"Сравнение моделей по {metric.upper()} ({_display_split_name(split)})"),
        xlabel=metric.upper(),
        ylabel="Модель",
    )
    axis.grid(axis="x", alpha=GRID_ALPHA)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


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
        title="Наиболее значимые признаки модели",
        xlabel="Важность признака",
        ylabel="Преобразованный признак",
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
    selected["model_display"] = selected["model"].map(_display_model_name)
    model_display_order = [_display_model_name(model) for model in model_order]

    figure, axis = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=selected,
        x="feature_set_label",
        y=metric,
        hue="model_display",
        order=scenario_order,
        hue_order=model_display_order,
        ax=axis,
        palette=(PASTEL_BLUE, PASTEL_PEACH, PASTEL_LAVENDER),
        saturation=1.0,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set(
        title=(f"Сравнение наборов признаков по {metric.upper()} ({_display_split_name(split)})"),
        xlabel="Сценарий набора признаков",
        ylabel=metric.upper(),
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=GRID_ALPHA)
    axis.legend(title="Модель")
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
    selected["model_display"] = selected["model"].map(_display_model_name)
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
        selected["model_display"],
        selected[metric],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set(
        title=(
            f"Сравнение расширенного ансамбля по {metric.upper()} ({_display_split_name(split)})"
        ),
        xlabel=metric.upper(),
        ylabel="Модель",
    )
    axis.grid(axis="x", alpha=GRID_ALPHA)
    axis.legend(
        handles=[
            Patch(
                facecolor=MODEL_COMPARISON_BASE_COLOR,
                edgecolor="white",
                label="Сильнейшая индивидуальная модель",
            ),
            Patch(
                facecolor=MODEL_COMPARISON_HIGHLIGHT_COLOR,
                edgecolor="white",
                label="Расширенный ансамбль",
            ),
        ],
        title="Группа модели",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_locked_test_comparison(
    metrics: pd.DataFrame,
    *,
    metric: str = "rmse",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare preselected candidates on the locked test split without ranking them."""
    required_columns = {"model", "model_group", "split", "used_for_model_selection", metric}
    missing_columns = sorted(required_columns - set(metrics.columns))
    if missing_columns:
        raise ValueError(f"Missing locked test evaluation columns: {missing_columns}")

    selected = metrics.loc[metrics["split"] == "test"].copy()
    if selected.empty:
        raise ValueError("Locked test comparison requires test metrics.")
    if selected["used_for_model_selection"].any():
        raise ValueError("Locked test metrics cannot be marked for model selection.")

    selected["model_display"] = selected["model"].map(_display_model_name)
    selected["model_group_display"] = selected["model_group"].map(_display_model_group)
    group_display_order = selected["model_group_display"].drop_duplicates().tolist()
    group_palette = _build_label_palette(group_display_order)

    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=selected,
        x="model_display",
        y=metric,
        hue="model_group_display",
        hue_order=group_display_order,
        ax=axis,
        palette=group_palette,
        saturation=1.0,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.set(
        title=f"Сравнение моделей-кандидатов по {metric.upper()} на тестовой выборке",
        xlabel="Модель-кандидат",
        ylabel=metric.upper(),
    )
    axis.grid(axis="y", alpha=GRID_ALPHA)
    axis.legend(title="Группа модели")
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_model_evaluation_actual_vs_predicted(
    predictions: pd.DataFrame,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot actual values and all fixed candidate predictions chronologically."""
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
    selected[DATETIME_COLUMN] = pd.to_datetime(selected[DATETIME_COLUMN], errors="raise")
    actual_by_timestamp = selected.drop_duplicates(DATETIME_COLUMN).sort_values(DATETIME_COLUMN)

    figure, axis = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    axis.plot(
        actual_by_timestamp[DATETIME_COLUMN],
        actual_by_timestamp[ACTUAL_COLUMN],
        label="Фактические значения",
        color=ACTUAL_LINE_COLOR,
        linewidth=1.4,
    )
    model_order = selected["model"].drop_duplicates().tolist()
    model_palette = _build_label_palette(model_order)

    for model_name, model_predictions in selected.groupby("model", sort=False):
        model_predictions = model_predictions.sort_values(DATETIME_COLUMN)
        axis.plot(
            model_predictions[DATETIME_COLUMN],
            model_predictions[PREDICTED_COLUMN],
            label=_display_model_name(model_name),
            color=model_palette[model_name],
            linewidth=1.0,
            alpha=0.85,
        )

    axis.set(
        title=("Фактические и прогнозные значения транспортной нагрузки на тестовой выборке"),
        xlabel="Дата и время",
        ylabel="Интенсивность транспортного потока",
    )
    axis.legend()
    axis.grid(alpha=GRID_ALPHA)
    figure.autofmt_xdate()
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_residual_comparison(
    predictions: pd.DataFrame,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare residual distributions for fixed evaluation candidates."""
    error_data = add_prediction_errors(predictions)
    selected = error_data.loc[error_data["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No prediction rows found for split: {split}")

    selected["model_display"] = selected["model"].map(_display_model_name)
    model_display_order = selected["model_display"].drop_duplicates().tolist()
    model_palette = _build_label_palette(model_display_order)

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=selected,
        x=RESIDUAL_COLUMN,
        hue="model_display",
        hue_order=model_display_order,
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
        ax=axis,
        palette=model_palette,
    )
    axis.axvline(0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.2)
    axis.set(
        title="Сравнение распределений остатков прогнозирования на тестовой выборке",
        xlabel="Остаток прогноза: фактическое значение − прогноз",
        ylabel="Плотность распределения",
    )
    axis.get_legend().set_title("Модель")
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_error_by_hour_comparison(
    hourly_errors: pd.DataFrame,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare mean absolute error by hour for fixed evaluation candidates."""
    required_columns = {"model", "split", "hour", "mae"}
    missing_columns = sorted(required_columns - set(hourly_errors.columns))
    if missing_columns:
        raise ValueError(f"Missing hourly error columns: {missing_columns}")

    selected = hourly_errors.loc[hourly_errors["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No hourly error rows found for split: {split}")

    selected["model_display"] = selected["model"].map(_display_model_name)
    model_display_order = selected["model_display"].drop_duplicates().tolist()
    model_palette = _build_label_palette(model_display_order)

    figure, axis = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=selected,
        x="hour",
        y="mae",
        hue="model_display",
        hue_order=model_display_order,
        marker="o",
        ax=axis,
        palette=model_palette,
    )
    axis.set(
        title="Средняя абсолютная ошибка по часам суток на тестовой выборке",
        xlabel="Час суток",
        ylabel="Средняя абсолютная ошибка",
    )
    axis.get_legend().set_title("Модель")
    axis.set_xticks(range(24))
    axis.grid(alpha=GRID_ALPHA)
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis


def plot_large_errors(
    large_errors: pd.DataFrame,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot the largest absolute errors over time for each candidate."""
    required_columns = {
        "model",
        "split",
        DATETIME_COLUMN,
        ABSOLUTE_ERROR_COLUMN,
    }
    missing_columns = sorted(required_columns - set(large_errors.columns))
    if missing_columns:
        raise ValueError(f"Missing large-error columns: {missing_columns}")

    selected = large_errors.loc[large_errors["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No large-error rows found for split: {split}")
    selected[DATETIME_COLUMN] = pd.to_datetime(selected[DATETIME_COLUMN], errors="raise")

    selected["model_display"] = selected["model"].map(_display_model_name)
    model_display_order = selected["model_display"].drop_duplicates().tolist()
    model_palette = _build_label_palette(model_display_order)

    figure, axis = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=selected,
        x=DATETIME_COLUMN,
        y=ABSOLUTE_ERROR_COLUMN,
        hue="model_display",
        hue_order=model_display_order,
        alpha=0.75,
        ax=axis,
        palette=model_palette,
    )
    axis.set(
        title="Крупные ошибки прогнозирования на тестовой выборке",
        xlabel="Дата и время",
        ylabel="Абсолютная ошибка",
    )
    axis.get_legend().set_title("Модель")
    axis.grid(alpha=GRID_ALPHA)
    figure.autofmt_xdate()
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure, axis
