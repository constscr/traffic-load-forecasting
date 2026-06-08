import numpy as np
import pandas as pd

from traffic_forecasting.config import DATETIME_COLUMN

# Time constants for feature engineering
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
MONTHS_PER_YEAR = 12
WEEKEND_DAYS = (5, 6)


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add numeric time components derived from the observation timestamp."""
    result = data.copy()
    result["hour"] = result[DATETIME_COLUMN].dt.hour
    result["day_of_week"] = result[DATETIME_COLUMN].dt.dayofweek
    result["month"] = result[DATETIME_COLUMN].dt.month
    return result


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add weekend and holiday indicator features."""
    result = data.copy()
    result["is_weekend"] = result["day_of_week"].isin(WEEKEND_DAYS).astype("int8")
    holiday_values = result["holiday"].astype("string").str.strip()
    result["is_holiday"] = (holiday_values.notna() & holiday_values.ne("")).astype("int8")
    return result


def add_cyclical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Encode periodic calendar components with sine and cosine pairs."""
    result = data.copy()
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / HOURS_PER_DAY)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / HOURS_PER_DAY)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / DAYS_PER_WEEK)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / DAYS_PER_WEEK)
    # Month is shifted so January starts at the beginning of the annual cycle.
    month_position = result["month"] - 1
    result["month_sin"] = np.sin(2 * np.pi * month_position / MONTHS_PER_YEAR)
    result["month_cos"] = np.cos(2 * np.pi * month_position / MONTHS_PER_YEAR)
    return result
