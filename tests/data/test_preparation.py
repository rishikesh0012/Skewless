from pathlib import Path

import pandas as pd
import pytest

from ml_skew.data import (
    DataValidationError,
    load_trip_records,
    prepare_training_data,
)


@pytest.fixture
def raw_trip_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2025-01-06 08:30:00",
                "2025-01-11 14:00:00",
                "2025-01-07 18:00:00",
                "2025-01-08 10:00:00",
            ],
            "passenger_count": [1, None, 2, 1],
            "trip_distance": [3.5, 7.2, -1.0, 2.1],
            "PULocationID": [161, 132, 142, 0],
            "DOLocationID": [236, 230, 263, 161],
            "fare_amount": [18.5, 31.0, 12.0, 14.0],
        }
    )


def test_prepare_training_data_filters_invalid_rows(
    raw_trip_data: pd.DataFrame,
) -> None:
    dataset = prepare_training_data(raw_trip_data)

    assert dataset.summary.rows_received == 4
    assert dataset.summary.rows_valid == 2
    assert dataset.summary.rows_removed == 2
    assert dataset.features["passenger_count"].tolist() == [1, 1]
    assert dataset.target.tolist() == [18.5, 31.0]


def test_prepare_training_data_creates_time_features(
    raw_trip_data: pd.DataFrame,
) -> None:
    dataset = prepare_training_data(raw_trip_data)

    first = dataset.features.iloc[0]
    second = dataset.features.iloc[1]

    assert first["pickup_hour"] == 8
    assert first["is_rush_hour"] == 1
    assert first["is_weekend"] == 0

    assert second["pickup_hour"] == 14
    assert second["is_rush_hour"] == 0
    assert second["is_weekend"] == 1


def test_missing_required_column_is_rejected(
    raw_trip_data: pd.DataFrame,
) -> None:
    invalid = raw_trip_data.drop(columns=["fare_amount"])

    with pytest.raises(
        DataValidationError,
        match="fare_amount",
    ):
        prepare_training_data(invalid)


def test_loader_reads_parquet(
    tmp_path: Path,
    raw_trip_data: pd.DataFrame,
) -> None:
    path = tmp_path / "trips.parquet"
    raw_trip_data.to_parquet(path, index=False)

    loaded = load_trip_records(path, row_limit=2)

    assert len(loaded) == 2
    assert list(loaded.columns) == list(raw_trip_data.columns)
