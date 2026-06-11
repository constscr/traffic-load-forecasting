from traffic_forecasting.pipeline import run_pipeline


def main() -> None:
    """Generate the reproducible interim and processed datasets."""
    run_pipeline()


if __name__ == "__main__":
    main()
