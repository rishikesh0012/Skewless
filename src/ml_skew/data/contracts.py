from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RAW_COLUMN_MAP = {
    "tpep_pickup_datetime": "pickup_datetime",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance_miles",
    "PULocationID": "pickup_location_id",
    "DOLocationID": "dropoff_location_id",
    "fare_amount": "fare_amount",
}

REQUIRED_RAW_COLUMNS = tuple(RAW_COLUMN_MAP)

FEATURE_COLUMNS = (
    "trip_distance_miles",
    "passenger_count",
    "pickup_location_id",
    "dropoff_location_id",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
)

TARGET_COLUMN = "fare_amount"


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    rows_received: int
    rows_valid: int
    rows_removed: int


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    features: pd.DataFrame
    target: pd.Series[float]
    summary: PreparationSummary
