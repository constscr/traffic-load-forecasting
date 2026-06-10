import numpy as np
import pytest

from traffic_forecasting.evaluation import (
    build_time_series_split,
    calculate_regression_metrics,
    get_tuning_scoring,
    mean_absolute_percentage_error,
    summarize_time_series_splits,
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


def test_time_series_split_uses_expanding_ordered_windows() -> None:
    splitter = build_time_series_split(n_splits=3)
    summary = summarize_time_series_splits(n_samples=20, splitter=splitter)

    assert summary["train_size"].tolist() == [5, 10, 15]
    assert summary["validation_size"].tolist() == [5, 5, 5]
    assert (summary["train_end"] < summary["validation_start"]).all()
    assert summary["train_start"].eq(0).all()


def test_tuning_scoring_uses_rmse_and_supporting_metrics() -> None:
    scoring = get_tuning_scoring()

    assert set(scoring) == {"rmse", "mae", "mape", "r2"}
    assert scoring["rmse"] == "neg_root_mean_squared_error"


def test_time_series_split_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_time_series_split(n_splits=1)

    with pytest.raises(ValueError, match="positive"):
        summarize_time_series_splits(n_samples=0)
