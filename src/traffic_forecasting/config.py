from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# Raw dataset configuration
RAW_DATA_FILENAME = "metro_interstate_traffic_volume.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
INTERIM_DATA_PATH = INTERIM_DATA_DIR / "metro_traffic_hourly.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "metro_traffic_features.csv"

# Dataset schema
DATETIME_COLUMN = "date_time"
TARGET_COLUMN = "traffic_volume"

# Forecasting setup
TIME_FREQUENCY = "h"
FORECAST_HORIZON = 1

# Lag values selected for short-term, daily, and weekly traffic dependencies
LAG_HOURS = (1, 2, 3, 24, 168)

# Rolling window sizes for moving average features to capture short-term and weekly trends
ROLLING_WINDOWS = (3, 6)

# Default proportions for chronological train, validation, and test splits
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.20

# Baseline evaluation outputs
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_model_metrics.csv"
ENSEMBLE_METRICS_PATH = METRICS_DIR / "core_ensemble_model_metrics.csv"
MODEL_COMPARISON_PATH = METRICS_DIR / "baseline_ensemble_comparison.csv"

# Time-series tuning configuration and outputs
TIME_SERIES_CV_SPLITS = 3
TUNING_N_ITER = 3
TUNING_BEST_PARAMS_PATH = METRICS_DIR / "tuning_best_parameters.csv"
TUNING_RESULTS_PATH = METRICS_DIR / "time_series_tuning_results.csv"
TUNING_COMPARISON_PATH = METRICS_DIR / "default_tuned_model_comparison.csv"

# End-to-end experiment outputs
EXPERIMENT_METRICS_PATH = METRICS_DIR / "experiment_model_metrics.csv"
EXPERIMENT_COMPARISON_PATH = TABLES_DIR / "experiment_model_comparison.csv"
VALIDATION_PREDICTIONS_PATH = PREDICTIONS_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_predictions.csv"

# Model result visualization outputs
TRAFFIC_VOLUME_TIME_SERIES_FIGURE_PATH = FIGURES_DIR / "model_result_traffic_time_series.png"
ACTUAL_VS_PREDICTED_FIGURE_PATH = FIGURES_DIR / "actual_vs_predicted_validation.png"
RESIDUAL_DISTRIBUTION_FIGURE_PATH = FIGURES_DIR / "validation_residual_distribution.png"
TRUE_VS_PREDICTED_FIGURE_PATH = FIGURES_DIR / "validation_true_vs_predicted.png"
ERROR_BY_HOUR_FIGURE_PATH = FIGURES_DIR / "validation_error_by_hour.png"
MODEL_COMPARISON_FIGURE_PATH = FIGURES_DIR / "validation_model_comparison.png"
FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "catboost_feature_importance.png"

# Feature set experiment outputs
FEATURE_SET_METRICS_PATH = METRICS_DIR / "feature_set_experiment_metrics.csv"
FEATURE_SET_COMPARISON_PATH = TABLES_DIR / "feature_set_comparison.csv"
FEATURE_SET_PREDICTIONS_PATH = PREDICTIONS_DIR / "feature_set_validation_predictions.csv"
FEATURE_SET_COMPARISON_FIGURE_PATH = FIGURES_DIR / "feature_set_comparison.png"

# Extended ensemble experiment outputs
EXTENDED_ENSEMBLE_METRICS_PATH = METRICS_DIR / "extended_ensemble_metrics.csv"
EXTENDED_ENSEMBLE_COMPARISON_PATH = TABLES_DIR / "extended_ensemble_comparison.csv"
EXTENDED_ENSEMBLE_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "extended_ensemble_validation_predictions.csv"
)
EXTENDED_ENSEMBLE_FIGURE_PATH = FIGURES_DIR / "extended_ensemble_comparison.png"

# Persist only the model selected by validation results from the completed model stages
PERSISTED_EXPERIMENT_MODELS = ("catboost",)

# Reproducibility
RANDOM_STATE = 42
