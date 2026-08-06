from __future__ import annotations

import pandas as pd

from ml_skew.data.contracts import (
    FEATURE_COLUMNS,
    RAW_COLUMN_MAP,
    REQUIRED_RAW_COLUMNS,
    PreparationSummary,
    PreparedDataset,
)
from ml_skew.data.validation import DataValidationError, require_columns

DEFAULT_PASSENGER_COUNT = 1
MAX_TRIP_DISTANCE_MILES = 100.0
MAX_FARE_AMOUNT = 500.0


def prepare_training_data(frame: pd.DataFrame) -> PreparedDataset:
    require_columns(frame, REQUIRED_RAW_COLUMNS)

    prepared = frame.rename(columns=RAW_COLUMN_MAP).copy()
    rows_received = len(prepared)

    prepared["pickup_datetime"] = pd.to_datetime(
        prepared["pickup_datetime"],
        errors="coerce",
    )

    numeric_columns = (
        "passenger_count",
        "trip_distance_miles",
        "pickup_location_id",
        "dropoff_location_id",
        "fare_amount",
    )

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared["passenger_count"] = prepared["passenger_count"].fillna(DEFAULT_PASSENGER_COUNT)

    valid_rows = (
        prepared["pickup_datetime"].notna()
        & prepared["trip_distance_miles"].between(
            0.01,
            MAX_TRIP_DISTANCE_MILES,
        )
        & prepared["fare_amount"].between(0.01, MAX_FARE_AMOUNT)
        & prepared["passenger_count"].between(1, 8)
        & prepared["pickup_location_id"].ge(1)
        & prepared["dropoff_location_id"].ge(1)
    )

    prepared = prepared.loc[valid_rows].copy()

    if prepared.empty:
        raise DataValidationError("No valid training rows remain after validation")

    pickup_datetime = prepared["pickup_datetime"].dt

    prepared["pickup_hour"] = pickup_datetime.hour
    prepared["pickup_day_of_week"] = pickup_datetime.dayofweek
    prepared["pickup_month"] = pickup_datetime.month
    prepared["is_weekend"] = (prepared["pickup_day_of_week"] >= 5).astype("int8")
    prepared["is_rush_hour"] = (
        prepared["pickup_hour"].between(7, 9) | prepared["pickup_hour"].between(16, 18)
    ).astype("int8")

    integer_columns = (
        "passenger_count",
        "pickup_location_id",
        "dropoff_location_id",
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
    )

    for column in integer_columns:
        prepared[column] = prepared[column].astype("int32")

    features = prepared.loc[:, FEATURE_COLUMNS].reset_index(drop=True)
    target = prepared["fare_amount"].astype("float64").reset_index(drop=True)

    rows_valid = len(features)

    return PreparedDataset(
        features=features,
        target=target,
        summary=PreparationSummary(
            rows_received=rows_received,
            rows_valid=rows_valid,
            rows_removed=rows_received - rows_valid,
        ),
    )
