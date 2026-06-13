from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from traffic_forecasting.config import (
    DATETIME_COLUMN,
    LARGE_ERROR_QUANTILE,
    TARGET_COLUMN,
    TIME_SERIES_CV_SPLITS,
)

ACTUAL_COLUMN = f"actual_{TARGET_COLUMN}"
PREDICTED_COLUMN = f"predicted_{TARGET_COLUMN}"
RESIDUAL_COLUMN = "residual"
ABSOLUTE_ERROR_COLUMN = "absolute_error"


def mean_absolute_percentage_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """Calculate MAPE while excluding undefined zero-target percentages."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    nonzero_mask = actual != 0
    if not nonzero_mask.any():
        raise ValueError("MAPE is undefined when all target values are zero.")

    percentage_errors = np.abs(
        (actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask]
    )
    return float(percentage_errors.mean() * 100)


def calculate_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Calculate the regression metrics used for model comparison."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def add_prediction_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add residual, absolute, and percentage errors without changing predictions."""
    required_columns = {ACTUAL_COLUMN, PREDICTED_COLUMN}
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    result = predictions.copy()
    result[RESIDUAL_COLUMN] = result[ACTUAL_COLUMN] - result[PREDICTED_COLUMN]
    result[ABSOLUTE_ERROR_COLUMN] = result[RESIDUAL_COLUMN].abs()
    nonzero_actual = result[ACTUAL_COLUMN].ne(0)
    result["absolute_percentage_error"] = np.nan
    result.loc[nonzero_actual, "absolute_percentage_error"] = (
        result.loc[nonzero_actual, ABSOLUTE_ERROR_COLUMN]
        / result.loc[nonzero_actual, ACTUAL_COLUMN].abs()
        * 100
    )
    return result


def summarize_residuals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize signed and absolute error behavior for each evaluated model."""
    error_data = add_prediction_errors(predictions)
    required_columns = {"model", "split"}
    missing_columns = sorted(required_columns - set(error_data.columns))
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    return (
        error_data.groupby(["model", "split"], as_index=False)
        .agg(
            observation_count=(RESIDUAL_COLUMN, "size"),
            residual_mean=(RESIDUAL_COLUMN, "mean"),
            residual_median=(RESIDUAL_COLUMN, "median"),
            residual_std=(RESIDUAL_COLUMN, "std"),
            mean_absolute_error=(ABSOLUTE_ERROR_COLUMN, "mean"),
            median_absolute_error=(ABSOLUTE_ERROR_COLUMN, "median"),
            maximum_absolute_error=(ABSOLUTE_ERROR_COLUMN, "max"),
        )
        .sort_values(["split", "model"])
        .reset_index(drop=True)
    )


def summarize_errors_by_hour(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate hourly error patterns for each evaluated model."""
    error_data = add_prediction_errors(predictions)
    required_columns = {"model", "split", DATETIME_COLUMN}
    missing_columns = sorted(required_columns - set(error_data.columns))
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    error_data[DATETIME_COLUMN] = pd.to_datetime(
        error_data[DATETIME_COLUMN],
        errors="raise",
    )
    error_data["hour"] = error_data[DATETIME_COLUMN].dt.hour
    return (
        error_data.groupby(["model", "split", "hour"], as_index=False)
        .agg(
            observation_count=(ABSOLUTE_ERROR_COLUMN, "size"),
            mae=(ABSOLUTE_ERROR_COLUMN, "mean"),
            mean_residual=(RESIDUAL_COLUMN, "mean"),
        )
        .sort_values(["split", "model", "hour"])
        .reset_index(drop=True)
    )


def identify_large_errors(
    predictions: pd.DataFrame,
    *,
    quantile: float = LARGE_ERROR_QUANTILE,
) -> pd.DataFrame:
    """Return model-specific errors at or above the selected absolute-error quantile."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be greater than 0 and less than 1.")

    error_data = add_prediction_errors(predictions)
    required_columns = {"model", "split"}
    missing_columns = sorted(required_columns - set(error_data.columns))
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    thresholds = error_data.groupby(["model", "split"])[ABSOLUTE_ERROR_COLUMN].transform(
        lambda values: values.quantile(quantile)
    )
    large_errors = error_data.loc[error_data[ABSOLUTE_ERROR_COLUMN] >= thresholds].copy()
    large_errors["error_quantile_threshold"] = thresholds.loc[large_errors.index]
    large_errors["large_error_quantile"] = quantile
    return large_errors.sort_values(
        ["split", "model", ABSOLUTE_ERROR_COLUMN],
        ascending=[True, True, False],
    ).reset_index(drop=True)


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


def build_time_series_split(
    n_splits: int = TIME_SERIES_CV_SPLITS,
) -> TimeSeriesSplit:
    """Build expanding-window cross-validation without shuffling."""
    if n_splits < 2:
        raise ValueError("TimeSeriesSplit requires at least two splits.")
    return TimeSeriesSplit(n_splits=n_splits)


def get_tuning_scoring() -> dict[str, str | object]:
    """Return primary RMSE and supporting regression scorers for tuning."""
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "mape": make_scorer(
            mean_absolute_percentage_error,
            greater_is_better=False,
        ),
        "r2": "r2",
    }


def summarize_time_series_splits(
    n_samples: int,
    splitter: TimeSeriesSplit | None = None,
) -> pd.DataFrame:
    """Summarize expanding train and validation boundaries for audit."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    cv = splitter or build_time_series_split()
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(
        cv.split(np.arange(n_samples)),
        start=1,
    ):
        rows.append(
            {
                "fold": fold,
                "train_start": int(train_indices[0]),
                "train_end": int(train_indices[-1]),
                "train_size": len(train_indices),
                "validation_start": int(validation_indices[0]),
                "validation_end": int(validation_indices[-1]),
                "validation_size": len(validation_indices),
            }
        )

    return pd.DataFrame(rows)
