import argparse

import matplotlib.pyplot as plt

from traffic_forecasting.config import FEATURE_SET_COMPARISON_FIGURE_PATH
from traffic_forecasting.pipeline import (
    run_experiment_pipeline,
    run_feature_set_experiment_pipeline,
)
from traffic_forecasting.visualization import plot_feature_set_comparison


def parse_args() -> argparse.Namespace:
    """Parse the requested experiment workflow."""
    parser = argparse.ArgumentParser(
        description="Run reproducible traffic forecasting experiments."
    )
    parser.add_argument(
        "--feature-sets",
        action="store_true",
        help="Run Stage 11 feature set comparison instead of the Stage 9 model comparison.",
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


def main() -> None:
    """Run the selected experiment workflow."""
    args = parse_args()
    if args.feature_sets:
        run_feature_set_experiment()
    else:
        run_model_experiment()


if __name__ == "__main__":
    main()
