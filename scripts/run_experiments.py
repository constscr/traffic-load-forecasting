import argparse

import matplotlib.pyplot as plt

from traffic_forecasting.config import (
    EXTENDED_ENSEMBLE_FIGURE_PATH,
    FEATURE_SET_COMPARISON_FIGURE_PATH,
    MODEL_EVALUATION_ACTUAL_VS_PREDICTED_FIGURE_PATH,
    MODEL_EVALUATION_ERROR_BY_HOUR_FIGURE_PATH,
    MODEL_EVALUATION_FEATURE_IMPORTANCE_FIGURE_PATH,
    MODEL_EVALUATION_LARGE_ERRORS_FIGURE_PATH,
    MODEL_EVALUATION_RESIDUAL_COMPARISON_FIGURE_PATH,
    MODEL_EVALUATION_TEST_COMPARISON_FIGURE_PATH,
)
from traffic_forecasting.pipeline import (
    run_experiment_pipeline,
    run_extended_ensemble_pipeline,
    run_feature_set_experiment_pipeline,
    run_model_evaluation_pipeline,
)
from traffic_forecasting.visualization import (
    plot_error_by_hour_comparison,
    plot_extended_ensemble_comparison,
    plot_feature_importance,
    plot_feature_set_comparison,
    plot_large_errors,
    plot_locked_test_comparison,
    plot_model_evaluation_actual_vs_predicted,
    plot_residual_comparison,
)


def parse_args() -> argparse.Namespace:
    """Parse the requested experiment workflow."""
    parser = argparse.ArgumentParser(
        description="Run reproducible traffic forecasting experiments."
    )
    experiment_group = parser.add_mutually_exclusive_group()
    experiment_group.add_argument(
        "--feature-sets",
        action="store_true",
        help="Run Stage 11 feature set comparison instead of the Stage 9 model comparison.",
    )
    experiment_group.add_argument(
        "--extended-ensembles",
        action="store_true",
        help="Run Stage 12 extended ensemble comparison.",
    )
    experiment_group.add_argument(
        "--model-evaluation",
        action="store_true",
        help="Run Stage 13 independent locked test evaluation.",
    )
    return parser.parse_args()


def run_model_experiment() -> None:
    """Run the unified validation experiment and report generated outputs."""
    metrics, comparison, _, output_paths = run_experiment_pipeline()

    print("Validation model comparison:")
    print(comparison.to_string(index=False))
    print("\nGenerated outputs:")
    for output_name, output_path in output_paths.items():
        print(f"- {output_name}: {output_path}")
    print(f"\nMetric rows: {len(metrics)}")


def run_feature_set_experiment() -> None:
    """Run Stage 11 validation-only feature set experiments."""
    metrics, comparison, _, output_paths = run_feature_set_experiment_pipeline()
    figure, _ = plot_feature_set_comparison(
        comparison,
        output_path=FEATURE_SET_COMPARISON_FIGURE_PATH,
    )
    plt.close(figure)

    print("Feature set comparison:")
    print(comparison.to_string(index=False))
    print("\nGenerated outputs:")
    for output_path in output_paths:
        print(f"- {output_path}")
    print(f"- {FEATURE_SET_COMPARISON_FIGURE_PATH}")
    print(f"\nMetric rows: {len(metrics)}")


def run_extended_ensemble_experiment() -> None:
    """Run Stage 12 validation-only extended ensemble experiments."""
    metrics, comparison, _, output_paths = run_extended_ensemble_pipeline()
    figure, _ = plot_extended_ensemble_comparison(
        comparison,
        output_path=EXTENDED_ENSEMBLE_FIGURE_PATH,
    )
    plt.close(figure)

    print("Extended ensemble comparison:")
    print(comparison.to_string(index=False))
    print("\nGenerated outputs:")
    for output_path in output_paths:
        print(f"- {output_path}")
    print(f"- {EXTENDED_ENSEMBLE_FIGURE_PATH}")
    print(f"\nMetric rows: {len(metrics)}")


def run_model_evaluation_experiment() -> None:
    """Run Stage 13 locked test evaluation and research analyses."""
    (
        metrics,
        comparison,
        predictions,
        analyses,
        fitted_candidates,
        output_paths,
    ) = run_model_evaluation_pipeline()
    figures = [
        plot_locked_test_comparison(
            metrics,
            output_path=MODEL_EVALUATION_TEST_COMPARISON_FIGURE_PATH,
        )[0],
        plot_model_evaluation_actual_vs_predicted(
            predictions,
            output_path=MODEL_EVALUATION_ACTUAL_VS_PREDICTED_FIGURE_PATH,
        )[0],
        plot_residual_comparison(
            predictions,
            output_path=MODEL_EVALUATION_RESIDUAL_COMPARISON_FIGURE_PATH,
        )[0],
        plot_error_by_hour_comparison(
            analyses["error_by_hour"],
            output_path=MODEL_EVALUATION_ERROR_BY_HOUR_FIGURE_PATH,
        )[0],
        plot_large_errors(
            analyses["large_errors"],
            output_path=MODEL_EVALUATION_LARGE_ERRORS_FIGURE_PATH,
        )[0],
    ]

    figure_paths = [
        MODEL_EVALUATION_TEST_COMPARISON_FIGURE_PATH,
        MODEL_EVALUATION_ACTUAL_VS_PREDICTED_FIGURE_PATH,
        MODEL_EVALUATION_RESIDUAL_COMPARISON_FIGURE_PATH,
        MODEL_EVALUATION_ERROR_BY_HOUR_FIGURE_PATH,
        MODEL_EVALUATION_LARGE_ERRORS_FIGURE_PATH,
    ]

    if "catboost" in fitted_candidates and not analyses["feature_importance"].empty:
        figures.append(
            plot_feature_importance(
                fitted_candidates["catboost"],
                output_path=MODEL_EVALUATION_FEATURE_IMPORTANCE_FIGURE_PATH,
            )[0]
        )
        figure_paths.append(MODEL_EVALUATION_FEATURE_IMPORTANCE_FIGURE_PATH)

    for figure in figures:
        plt.close(figure)

    print("Independent locked test evaluation:")
    print(comparison.to_string(index=False))
    print("\nGenerated outputs:")
    for output_name, output_path in output_paths.items():
        print(f"- {output_name}: {output_path}")
    for figure_path in figure_paths:
        print(f"- {figure_path}")


def main() -> None:
    """Run the selected experiment workflow."""
    args = parse_args()
    if args.model_evaluation:
        run_model_evaluation_experiment()
    elif args.extended_ensembles:
        run_extended_ensemble_experiment()
    elif args.feature_sets:
        run_feature_set_experiment()
    else:
        run_model_experiment()


if __name__ == "__main__":
    main()
