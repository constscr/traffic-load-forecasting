from traffic_forecasting.pipeline import run_experiment_pipeline


def main() -> None:
    """Run the unified validation experiment and report generated outputs."""
    metrics, comparison, _, output_paths = run_experiment_pipeline()

    print("Validation model comparison:")
    print(comparison.to_string(index=False))
    print("\nGenerated outputs:")
    for output_name, output_path in output_paths.items():
        print(f"- {output_name}: {output_path}")
    print(f"\nMetric rows: {len(metrics)}")


if __name__ == "__main__":
    main()
