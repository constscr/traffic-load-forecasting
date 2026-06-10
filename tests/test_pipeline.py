import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

from traffic_forecasting.config import DATETIME_COLUMN, RANDOM_STATE, TARGET_COLUMN
from traffic_forecasting.pipeline import (
    build_model_comparison,
    build_tuning_pipeline,
    save_baseline_metrics,
    save_model_tuning_results,
    train_and_evaluate_baselines,
    train_and_evaluate_ensembles,
    tune_and_compare_models,
    tune_model_with_time_series_cv,
)
from traffic_forecasting.preprocessing import (
    get_model_feature_groups,
    prepare_model_inputs,
    split_chronologically,
)


def _make_synthetic_feature_data(periods: int = 60) -> pd.DataFrame:
    feature_groups = get_model_feature_groups()
    data = pd.DataFrame(
        {
            DATETIME_COLUMN: pd.date_range("2026-01-01", periods=periods, freq="h"),
            TARGET_COLUMN: np.linspace(1_000, 2_000, periods),
            "holiday": [pd.NA] * periods,
        }
    )

    for offset, column in enumerate(feature_groups["continuous_numeric"]):
        data[column] = np.arange(periods, dtype=float) + offset

    data["weather_main"] = ["Clear"] * periods
    data["weather_description"] = ["clear sky"] * periods
    data["is_weekend"] = (data[DATETIME_COLUMN].dt.dayofweek >= 5).astype("int8")
    data["is_holiday"] = np.zeros(periods, dtype="int8")

    for offset, column in enumerate(feature_groups["cyclical"]):
        data[column] = np.sin(np.arange(periods) + offset)

    return data


def _prepare_inputs() -> tuple[pd.DataFrame | pd.Series, ...]:
    splits = split_chronologically(_make_synthetic_feature_data())
    return prepare_model_inputs(*splits)[:6]


def test_train_and_evaluate_baselines_returns_validation_metrics_by_default() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = _prepare_inputs()

    metrics, fitted_models, preprocessor = train_and_evaluate_baselines(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

    assert len(metrics) == len(fitted_models)
    assert set(metrics["split"]) == {"validation"}
    assert metrics["used_for_model_comparison"].all()
    assert np.isfinite(metrics[["mae", "rmse", "mape", "r2"]].to_numpy()).all()

    check_is_fitted(preprocessor)
    for model in fitted_models.values():
        check_is_fitted(model)


def test_train_and_evaluate_baselines_calculates_locked_test_metrics_explicitly() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = _prepare_inputs()

    metrics, fitted_models, _ = train_and_evaluate_baselines(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=True,
    )

    assert len(metrics) == len(fitted_models) * 2
    assert set(metrics["split"]) == {"validation", "test"}
    assert not metrics.loc[
        metrics["split"] == "test",
        "used_for_model_comparison",
    ].any()


def test_preprocessing_statistics_are_fitted_from_train_only() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = _prepare_inputs()
    X_validation = X_validation.copy()
    X_test = X_test.copy()
    X_validation["temp"] = 100_000
    X_test["temp"] = 200_000

    _, _, preprocessor = train_and_evaluate_baselines(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

    continuous_columns = get_model_feature_groups()["continuous_numeric"]
    temp_index = continuous_columns.index("temp")
    scaler = preprocessor.named_transformers_["continuous"].named_steps["scaler"]

    assert scaler.mean_[temp_index] == pytest.approx(X_train["temp"].mean())
    assert scaler.mean_[temp_index] != pytest.approx(X_validation["temp"].mean())


def test_save_baseline_metrics_writes_csv(tmp_path) -> None:
    metrics = pd.DataFrame(
        {
            "model": ["dummy_mean"],
            "split": ["validation"],
            "used_for_model_comparison": [True],
            "mae": [100.0],
            "rmse": [120.0],
            "mape": [5.0],
            "r2": [0.5],
        }
    )
    output_path = tmp_path / "metrics" / "baseline.csv"

    saved_path = save_baseline_metrics(metrics, output_path)

    assert saved_path == output_path
    pd.testing.assert_frame_equal(pd.read_csv(saved_path), metrics)


def test_train_and_evaluate_ensembles_supports_all_core_models() -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = _prepare_inputs()

    metrics, fitted_models, preprocessor = train_and_evaluate_ensembles(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

    assert tuple(fitted_models) == (
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    )
    assert len(metrics) == len(fitted_models)
    assert set(metrics["split"]) == {"validation"}
    assert metrics["used_for_model_comparison"].all()
    assert np.isfinite(metrics[["mae", "rmse", "mape", "r2"]].to_numpy()).all()

    check_is_fitted(preprocessor)
    for model in fitted_models.values():
        check_is_fitted(model)


def test_ensemble_test_metrics_require_explicit_opt_in(monkeypatch) -> None:
    X_train, X_validation, X_test, y_train, y_validation, y_test = _prepare_inputs()
    monkeypatch.setattr(
        "traffic_forecasting.pipeline.get_core_ensemble_model_registry",
        lambda: {
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=5,
                random_state=RANDOM_STATE,
            )
        },
    )

    metrics, _, _ = train_and_evaluate_ensembles(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=True,
    )

    assert set(metrics["split"]) == {"validation", "test"}
    assert not metrics.loc[
        metrics["split"] == "test",
        "used_for_model_comparison",
    ].any()


def test_build_model_comparison_labels_groups_and_ranks_by_rmse() -> None:
    baseline_metrics = pd.DataFrame(
        {
            "model": ["decision_tree"],
            "split": ["validation"],
            "used_for_model_comparison": [True],
            "mae": [230.0],
            "rmse": [370.0],
            "mape": [9.6],
            "r2": [0.96],
        }
    )
    ensemble_metrics = pd.DataFrame(
        {
            "model": ["random_forest"],
            "split": ["validation"],
            "used_for_model_comparison": [True],
            "mae": [200.0],
            "rmse": [320.0],
            "mape": [8.0],
            "r2": [0.97],
        }
    )

    comparison = build_model_comparison(baseline_metrics, ensemble_metrics)

    assert comparison["model"].tolist() == ["random_forest", "decision_tree"]
    assert comparison["model_group"].tolist() == ["core_ensemble", "baseline"]


class _FitSizeRecorder(BaseEstimator, TransformerMixin):
    fit_sizes: list[int] = []

    def fit(self, X, y=None):
        self.fit_sizes.append(len(X))
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


def test_tuning_pipeline_places_preprocessing_before_model() -> None:
    pipeline = build_tuning_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE))

    assert list(pipeline.named_steps) == ["preprocessing", "model"]


def test_tuning_fits_preprocessing_inside_each_time_series_fold(monkeypatch) -> None:
    X_train = pd.DataFrame({"feature": np.arange(20, dtype=float)})
    y_train = pd.Series(np.arange(20, dtype=float))
    _FitSizeRecorder.fit_sizes = []
    monkeypatch.setattr(
        "traffic_forecasting.pipeline.build_preprocessor",
        _FitSizeRecorder,
    )

    search = tune_model_with_time_series_cv(
        "ridge",
        X_train,
        y_train,
        n_iter=1,
    )

    assert search.refit == "rmse"
    assert _FitSizeRecorder.fit_sizes == [5, 10, 15, 20]


def test_tune_and_compare_models_returns_default_and_tuned_validation_metrics() -> None:
    X_train, X_validation, _, y_train, y_validation, _ = _prepare_inputs()

    comparison, best_parameters, tuning_results = tune_and_compare_models(
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_names=("ridge",),
        n_iter=1,
    )

    assert comparison["configuration"].tolist() == ["default", "tuned"]
    assert set(comparison["model"]) == {"ridge"}
    assert set(best_parameters["model"]) == {"ridge"}
    assert set(tuning_results["model"]) == {"ridge"}
    assert tuning_results["rank"].tolist() == [1]
    assert np.isfinite(comparison[["mae", "rmse", "mape", "r2"]].to_numpy()).all()


def test_save_model_tuning_results_writes_all_tables(tmp_path) -> None:
    comparison = pd.DataFrame({"model": ["ridge"], "rmse": [400.0]})
    best_parameters = pd.DataFrame(
        {
            "model": ["ridge"],
            "best_parameters": ['{"model__alpha": 10.0}'],
        }
    )
    tuning_results = pd.DataFrame(
        {
            "model": ["ridge"],
            "rank": [1],
            "mean_cv_rmse": [450.0],
        }
    )
    paths = (
        tmp_path / "comparison.csv",
        tmp_path / "best_parameters.csv",
        tmp_path / "tuning_results.csv",
    )

    saved_paths = save_model_tuning_results(
        comparison,
        best_parameters,
        tuning_results,
        comparison_path=paths[0],
        best_params_path=paths[1],
        tuning_results_path=paths[2],
    )

    assert saved_paths == paths
    pd.testing.assert_frame_equal(pd.read_csv(paths[0]), comparison)
    pd.testing.assert_frame_equal(pd.read_csv(paths[1]), best_parameters)
    pd.testing.assert_frame_equal(pd.read_csv(paths[2]), tuning_results)
