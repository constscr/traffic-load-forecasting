import numpy as np
import pytest

from traffic_forecasting.evaluation import (
    calculate_regression_metrics,
    mean_absolute_percentage_error,
)


def test_calculate_regression_metrics_returns_expected_values() -> None:
    metrics = calculate_regression_metrics(
        y_true=[100.0, 200.0, 300.0],
        y_pred=[90.0, 220.0, 330.0],
    )

    assert set(metrics) == {"mae", "rmse", "mape", "r2"}
    assert metrics["mae"] == pytest.approx(20.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1400 / 3))
    assert metrics["mape"] == pytest.approx(10.0)
    assert metrics["r2"] == pytest.approx(0.93)


def test_mean_absolute_percentage_error_excludes_zero_targets() -> None:
    result = mean_absolute_percentage_error(
        y_true=[0.0, 100.0, 200.0],
        y_pred=[50.0, 90.0, 220.0],
    )

    assert result == pytest.approx(10.0)


def test_mean_absolute_percentage_error_rejects_all_zero_targets() -> None:
    with pytest.raises(ValueError, match="all target values are zero"):
        mean_absolute_percentage_error(
            y_true=[0.0, 0.0],
            y_pred=[1.0, 2.0],
        )


def test_mean_absolute_percentage_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mean_absolute_percentage_error(
            y_true=[1.0, 2.0],
            y_pred=[1.0],
        )
