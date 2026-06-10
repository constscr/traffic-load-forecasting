from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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
