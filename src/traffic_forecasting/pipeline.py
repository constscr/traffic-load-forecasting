import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from traffic_forecasting.config import (
    BASELINE_METRICS_PATH,
    DATETIME_COLUMN,
    ENSEMBLE_METRICS_PATH,
    EXPERIMENT_COMPARISON_PATH,
    EXPERIMENT_METRICS_PATH,
    INTERIM_DATA_PATH,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    PERSISTED_EXPERIMENT_MODELS,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_PREDICTIONS_PATH,
    TUNING_BEST_PARAMS_PATH,
    TUNING_COMPARISON_PATH,
    TUNING_N_ITER,
    TUNING_RESULTS_PATH,
    VALIDATION_PREDICTIONS_PATH,
)
from traffic_forecasting.data_loader import load_raw_data
from traffic_forecasting.evaluation import (
    build_time_series_split,
    calculate_regression_metrics,
    get_tuning_scoring,
)
from traffic_forecasting.features import build_feature_dataset
from traffic_forecasting.models import (
    get_baseline_model_registry,
    get_core_ensemble_model_registry,
    get_experiment_model_registry,
    get_hyperparameter_search_spaces,
    get_tuning_model_registry,
)
from traffic_forecasting.preprocessing import (
    build_preprocessor,
    prepare_model_inputs,
    split_chronologically,
    transform_model_inputs,
)


def _fit_and_evaluate_model_registry(
    model_registry: dict[str, RegressorMixin],
    X_train_transformed: object,
    X_validation_transformed: object,
    X_test_transformed: object,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    *,
    include_test_metrics: bool,
) -> tuple[pd.DataFrame, dict[str, RegressorMixin]]:
    fitted_models: dict[str, RegressorMixin] = {}
    metric_rows: list[dict[str, str | bool | float]] = []

    for model_name, model in model_registry.items():
        model.fit(X_train_transformed, y_train)
        fitted_models[model_name] = model

        validation_predictions = model.predict(X_validation_transformed)
        metric_rows.append(
            {
                "model": model_name,
                "split": "validation",
                "used_for_model_comparison": True,
                **calculate_regression_metrics(y_validation, validation_predictions),
            }
        )

        if include_test_metrics:
            test_predictions = model.predict(X_test_transformed)
            metric_rows.append(
                {
                    "model": model_name,
                    "split": "test",
                    "used_for_model_comparison": False,
                    **calculate_regression_metrics(y_test, test_predictions),
                }
            )

    return pd.DataFrame(metric_rows), fitted_models


def train_and_evaluate_baselines(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    *,
    include_test_metrics: bool = False,
) -> tuple[pd.DataFrame, dict[str, RegressorMixin], ColumnTransformer]:
    """Train baselines, compare validation metrics, and optionally evaluate locked test data."""
    (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        preprocessor,
    ) = transform_model_inputs(X_train, X_validation, X_test)

    metrics, fitted_models = _fit_and_evaluate_model_registry(
        get_baseline_model_registry(),
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    return metrics, fitted_models, preprocessor


def train_and_evaluate_ensembles(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    *,
    include_test_metrics: bool = False,
) -> tuple[pd.DataFrame, dict[str, RegressorMixin], ColumnTransformer]:
    """Train core ensembles and optionally evaluate the locked test split."""
    (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        preprocessor,
    ) = transform_model_inputs(X_train, X_validation, X_test)

    metrics, fitted_models = _fit_and_evaluate_model_registry(
        get_core_ensemble_model_registry(),
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    return metrics, fitted_models, preprocessor


def save_baseline_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path = BASELINE_METRICS_PATH,
) -> Path:
    """Save the reproducible baseline metric table."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination, index=False)
    return destination


def save_ensemble_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path = ENSEMBLE_METRICS_PATH,
) -> Path:
    """Save the reproducible core ensemble metric table."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination, index=False)
    return destination


def build_model_comparison(
    baseline_metrics: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine validation metrics while preserving model-family labels."""
    baseline_results = baseline_metrics.copy()
    baseline_results.insert(0, "model_group", "baseline")

    ensemble_results = ensemble_metrics.copy()
    ensemble_results.insert(0, "model_group", "core_ensemble")

    return (
        pd.concat([baseline_results, ensemble_results], ignore_index=True)
        .sort_values(["split", "rmse"])
        .reset_index(drop=True)
    )


def save_model_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path = MODEL_COMPARISON_PATH,
) -> Path:
    """Save the baseline and core ensemble comparison table."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(destination, index=False)
    return destination


def build_model_pipeline(model: BaseEstimator) -> Pipeline:
    """Build a self-contained preprocessing and estimator pipeline."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", model),
        ]
    )


def build_tuning_pipeline(model: BaseEstimator) -> Pipeline:
    """Build a fold-safe preprocessing and estimator pipeline."""
    return build_model_pipeline(model)


def tune_model_with_time_series_cv(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = TUNING_N_ITER,
) -> RandomizedSearchCV:
    """Tune one model with expanding-window CV and fold-local preprocessing."""
    model_registry = get_tuning_model_registry()
    search_spaces = get_hyperparameter_search_spaces()

    if model_name not in model_registry:
        raise ValueError(f"Unknown tuning model: {model_name}")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    search = RandomizedSearchCV(
        estimator=build_tuning_pipeline(model_registry[model_name]),
        param_distributions=search_spaces[model_name],
        n_iter=n_iter,
        scoring=get_tuning_scoring(),
        refit="rmse",
        cv=build_time_series_split(),
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def _summarize_search_results(
    model_name: str,
    search: RandomizedSearchCV,
) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    return pd.DataFrame(
        {
            "model": model_name,
            "rank": results["rank_test_rmse"],
            "parameters": results["params"].map(lambda params: json.dumps(params, sort_keys=True)),
            "mean_cv_rmse": -results["mean_test_rmse"],
            "std_cv_rmse": results["std_test_rmse"],
            "mean_cv_mae": -results["mean_test_mae"],
            "mean_cv_mape": -results["mean_test_mape"],
            "mean_cv_r2": results["mean_test_r2"],
        }
    ).sort_values("rank")


def tune_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    model_names: tuple[str, ...] | None = None,
    n_iter: int = TUNING_N_ITER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune selected models and compare default and tuned validation metrics."""
    model_registry = get_tuning_model_registry()
    selected_names = model_names or tuple(model_registry)
    unknown_names = sorted(set(selected_names) - set(model_registry))
    if unknown_names:
        raise ValueError(f"Unknown tuning models: {unknown_names}")

    comparison_rows: list[dict[str, str | float]] = []
    parameter_rows: list[dict[str, str | float]] = []
    search_results: list[pd.DataFrame] = []

    for model_name in selected_names:
        default_pipeline = build_tuning_pipeline(model_registry[model_name])
        default_pipeline.fit(X_train, y_train)
        default_predictions = default_pipeline.predict(X_validation)
        comparison_rows.append(
            {
                "model": model_name,
                "configuration": "default",
                **calculate_regression_metrics(y_validation, default_predictions),
            }
        )

        search = tune_model_with_time_series_cv(
            model_name,
            X_train,
            y_train,
            n_iter=n_iter,
        )
        tuned_predictions = search.best_estimator_.predict(X_validation)
        comparison_rows.append(
            {
                "model": model_name,
                "configuration": "tuned",
                **calculate_regression_metrics(y_validation, tuned_predictions),
            }
        )
        parameter_rows.append(
            {
                "model": model_name,
                "best_cv_rmse": -float(search.best_score_),
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        search_results.append(_summarize_search_results(model_name, search))

    comparison = pd.DataFrame(comparison_rows).sort_values(["configuration", "rmse"])
    best_parameters = pd.DataFrame(parameter_rows).sort_values("best_cv_rmse")
    tuning_results = pd.concat(search_results, ignore_index=True)
    return comparison, best_parameters, tuning_results


def save_model_tuning_results(
    comparison: pd.DataFrame,
    best_parameters: pd.DataFrame,
    tuning_results: pd.DataFrame,
    *,
    comparison_path: str | Path = TUNING_COMPARISON_PATH,
    best_params_path: str | Path = TUNING_BEST_PARAMS_PATH,
    tuning_results_path: str | Path = TUNING_RESULTS_PATH,
) -> tuple[Path, Path, Path]:
    """Save validation comparison, best parameters, and CV search results."""
    destinations = (
        Path(comparison_path),
        Path(best_params_path),
        Path(tuning_results_path),
    )
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(destinations[0], index=False)
    best_parameters.to_csv(destinations[1], index=False)
    tuning_results.to_csv(destinations[2], index=False)
    return destinations


def run_baseline_pipeline(
    *,
    include_test_metrics: bool = False,
    metrics_path: str | Path = BASELINE_METRICS_PATH,
) -> pd.DataFrame:
    """Run validation comparison with optional locked-reference test evaluation."""
    feature_data = build_feature_dataset(load_raw_data())
    train_data, validation_data, test_data = split_chronologically(feature_data)
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        _,
        _,
        _,
    ) = prepare_model_inputs(train_data, validation_data, test_data)

    metrics, _, _ = train_and_evaluate_baselines(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    save_baseline_metrics(metrics, metrics_path)
    return metrics


def run_core_ensemble_pipeline(
    *,
    include_test_metrics: bool = False,
    ensemble_metrics_path: str | Path = ENSEMBLE_METRICS_PATH,
    comparison_path: str | Path = MODEL_COMPARISON_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run core ensemble evaluation and compare validation results with baselines."""
    feature_data = build_feature_dataset(load_raw_data())
    train_data, validation_data, test_data = split_chronologically(feature_data)
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        _,
        _,
        _,
    ) = prepare_model_inputs(train_data, validation_data, test_data)

    (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        _,
    ) = transform_model_inputs(X_train, X_validation, X_test)
    baseline_metrics, _ = _fit_and_evaluate_model_registry(
        get_baseline_model_registry(),
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    ensemble_metrics, _ = _fit_and_evaluate_model_registry(
        get_core_ensemble_model_registry(),
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        include_test_metrics=include_test_metrics,
    )
    comparison = build_model_comparison(baseline_metrics, ensemble_metrics)

    save_ensemble_metrics(ensemble_metrics, ensemble_metrics_path)
    save_model_comparison(comparison, comparison_path)
    return ensemble_metrics, comparison


def run_model_tuning_pipeline(
    *,
    model_names: tuple[str, ...] | None = None,
    n_iter: int = TUNING_N_ITER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune models on train folds and compare them on validation only."""
    feature_data = build_feature_dataset(load_raw_data())
    train_data, validation_data, test_data = split_chronologically(feature_data)
    (
        X_train,
        X_validation,
        _,
        y_train,
        y_validation,
        _,
        _,
        _,
        _,
    ) = prepare_model_inputs(train_data, validation_data, test_data)

    comparison, best_parameters, tuning_results = tune_and_compare_models(
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_names=model_names,
        n_iter=n_iter,
    )
    save_model_tuning_results(comparison, best_parameters, tuning_results)
    return comparison, best_parameters, tuning_results


def save_prepared_datasets(
    hourly_data: pd.DataFrame,
    feature_data: pd.DataFrame,
    *,
    interim_path: str | Path = INTERIM_DATA_PATH,
    processed_path: str | Path = PROCESSED_DATA_PATH,
) -> tuple[Path, Path]:
    """Save reproducible hourly and feature-enriched datasets."""
    destinations = (Path(interim_path), Path(processed_path))
    for data, destination in zip(
        (hourly_data, feature_data),
        destinations,
        strict=True,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(destination, index=False)

    return destinations


def prepare_and_save_datasets(
    *,
    interim_path: str | Path = INTERIM_DATA_PATH,
    processed_path: str | Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Path, Path]]:
    """Load raw data, build features, and save reproducible dataset stages."""
    hourly_data = load_raw_data()
    feature_data = build_feature_dataset(hourly_data)
    saved_paths = save_prepared_datasets(
        hourly_data,
        feature_data,
        interim_path=interim_path,
        processed_path=processed_path,
    )
    return hourly_data, feature_data, saved_paths


def train_and_evaluate_experiment_models(
    model_registry: dict[str, RegressorMixin],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    validation_timestamps: pd.Series,
    test_timestamps: pd.Series,
    *,
    include_test_outputs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    """Fit model pipelines and evaluate validation with optional locked test outputs."""
    if not model_registry:
        raise ValueError("At least one experiment model is required.")

    metric_rows: list[dict[str, str | bool | float]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    split_inputs = [
        (
            "validation",
            X_validation,
            y_validation,
            validation_timestamps,
            True,
        )
    ]
    if include_test_outputs:
        split_inputs.append(("test", X_test, y_test, test_timestamps, False))

    for model_name, model in model_registry.items():
        model_pipeline = build_model_pipeline(model)
        model_pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = model_pipeline

        for split_name, X_split, y_split, timestamps, used_for_comparison in split_inputs:
            predictions = model_pipeline.predict(X_split)
            metric_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "used_for_model_comparison": used_for_comparison,
                    **calculate_regression_metrics(y_split, predictions),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "split": split_name,
                        DATETIME_COLUMN: timestamps.reset_index(drop=True),
                        f"actual_{TARGET_COLUMN}": y_split.reset_index(drop=True),
                        f"predicted_{TARGET_COLUMN}": predictions,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows).sort_values(["split", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions, fitted_pipelines


def build_experiment_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank validation models without consulting locked test results."""
    validation_metrics = metrics.loc[
        (metrics["split"] == "validation") & metrics["used_for_model_comparison"]
    ].copy()
    validation_metrics = validation_metrics.sort_values("rmse").reset_index(drop=True)
    validation_metrics.insert(0, "validation_rank", range(1, len(validation_metrics) + 1))
    return validation_metrics


def save_experiment_tables(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    metrics_path: str | Path = EXPERIMENT_METRICS_PATH,
    comparison_path: str | Path = EXPERIMENT_COMPARISON_PATH,
) -> tuple[Path, Path]:
    """Save experiment metrics and validation-only model comparison."""
    destinations = (Path(metrics_path), Path(comparison_path))
    for data, destination in zip((metrics, comparison), destinations, strict=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(destination, index=False)
    return destinations


def save_experiment_predictions(
    predictions: pd.DataFrame,
    *,
    validation_path: str | Path = VALIDATION_PREDICTIONS_PATH,
    test_path: str | Path = TEST_PREDICTIONS_PATH,
) -> dict[str, Path]:
    """Save validation predictions and optional explicitly requested test predictions."""
    saved_paths: dict[str, Path] = {}
    for split_name, output_path in {
        "validation": validation_path,
        "test": test_path,
    }.items():
        split_predictions = predictions.loc[predictions["split"] == split_name]
        if split_predictions.empty:
            continue

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        split_predictions.to_csv(destination, index=False)
        saved_paths[split_name] = destination

    return saved_paths


def save_trained_model_pipelines(
    fitted_pipelines: dict[str, Pipeline],
    model_names: tuple[str, ...],
    *,
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, Path]:
    """Persist only explicitly selected fitted model pipelines."""
    unknown_names = sorted(set(model_names) - set(fitted_pipelines))
    if unknown_names:
        raise ValueError(f"Cannot save unknown or unfitted models: {unknown_names}")

    destination_dir = Path(models_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}
    for model_name in model_names:
        destination = destination_dir / f"{model_name}_pipeline.joblib"
        joblib.dump(fitted_pipelines[model_name], destination)
        saved_paths[model_name] = destination
    return saved_paths


def run_experiment_pipeline(
    *,
    model_names: tuple[str, ...] | None = None,
    persisted_model_names: tuple[str, ...] = PERSISTED_EXPERIMENT_MODELS,
    include_test_outputs: bool = False,
    interim_path: str | Path = INTERIM_DATA_PATH,
    processed_path: str | Path = PROCESSED_DATA_PATH,
    metrics_path: str | Path = EXPERIMENT_METRICS_PATH,
    comparison_path: str | Path = EXPERIMENT_COMPARISON_PATH,
    validation_predictions_path: str | Path = VALIDATION_PREDICTIONS_PATH,
    test_predictions_path: str | Path = TEST_PREDICTIONS_PATH,
    models_dir: str | Path = MODELS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Run the reproducible validation experiment with the global test split locked."""
    available_models = get_experiment_model_registry()
    selected_names = tuple(available_models) if model_names is None else model_names
    unknown_names = sorted(set(selected_names) - set(available_models))
    if unknown_names:
        raise ValueError(f"Unknown experiment models: {unknown_names}")

    unscheduled_persisted_names = sorted(set(persisted_model_names) - set(selected_names))
    if unscheduled_persisted_names:
        raise ValueError(
            f"Persisted models must be included in the experiment: {unscheduled_persisted_names}"
        )
    selected_models = {name: available_models[name] for name in selected_names}

    _, feature_data, dataset_paths = prepare_and_save_datasets(
        interim_path=interim_path,
        processed_path=processed_path,
    )
    train_data, validation_data, test_data = split_chronologically(feature_data)
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        _,
        validation_timestamps,
        test_timestamps,
    ) = prepare_model_inputs(train_data, validation_data, test_data)

    metrics, predictions, fitted_pipelines = train_and_evaluate_experiment_models(
        selected_models,
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        validation_timestamps,
        test_timestamps,
        include_test_outputs=include_test_outputs,
    )
    comparison = build_experiment_comparison(metrics)
    metric_paths = save_experiment_tables(
        metrics,
        comparison,
        metrics_path=metrics_path,
        comparison_path=comparison_path,
    )
    prediction_paths = save_experiment_predictions(
        predictions,
        validation_path=validation_predictions_path,
        test_path=test_predictions_path,
    )
    model_paths = save_trained_model_pipelines(
        fitted_pipelines,
        persisted_model_names,
        models_dir=models_dir,
    )

    output_paths = {
        "interim_dataset": dataset_paths[0],
        "processed_dataset": dataset_paths[1],
        "metrics": metric_paths[0],
        "comparison": metric_paths[1],
        **{f"{split}_predictions": path for split, path in prediction_paths.items()},
        **{f"{model_name}_pipeline": path for model_name, path in model_paths.items()},
    }
    return metrics, comparison, predictions, output_paths


def run_pipeline() -> None:
    """Prepare and save the reproducible hourly and feature datasets."""
    hourly_data, feature_data, saved_paths = prepare_and_save_datasets()
    print(f"Hourly dataset: {hourly_data.shape} -> {saved_paths[0]}")
    print(f"Feature dataset: {feature_data.shape} -> {saved_paths[1]}")
