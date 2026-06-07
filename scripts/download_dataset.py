from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from traffic_forecasting.config import RAW_DATA_PATH
from traffic_forecasting.logging_utils import setup_logger

UCI_DATASET_ID = 492
UCI_DATASET_NAME = "Metro Interstate Traffic Volume"

logger = setup_logger("download_dataset")


def fetch_dataset_from_uci(dataset_id: int = UCI_DATASET_ID) -> pd.DataFrame:
    """Fetch dataset from UCI Machine Learning Repository."""
    dataset = fetch_ucirepo(id=dataset_id)

    features = dataset.data.features
    targets = dataset.data.targets

    if features is None or targets is None:
        raise ValueError("UCI response does not contain features or targets.")

    # Preserve the source feature order and append the target as the final column.
    return pd.concat([features, targets], axis=1)


def save_raw_dataset(data: pd.DataFrame, output_path: Path = RAW_DATA_PATH) -> None:
    """Save dataset to the local data/raw directory."""
    if data.empty:
        raise ValueError("Downloaded dataset is empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    logger.info("Dataset saved to: %s", output_path)
    logger.info("Dataset dimensions: %s rows, %s columns", data.shape[0], data.shape[1])


def download_dataset(output_path: Path = RAW_DATA_PATH) -> None:
    """Download Metro Interstate Traffic Volume dataset and save it to data/raw."""
    if output_path.exists():
        logger.info("Dataset already exists: %s", output_path)
        logger.info("Skip downloading.")
        return

    logger.info("Downloading dataset: %s (UCI id=%s)", UCI_DATASET_NAME, UCI_DATASET_ID)

    data = fetch_dataset_from_uci()
    save_raw_dataset(data, output_path)


if __name__ == "__main__":
    download_dataset()
