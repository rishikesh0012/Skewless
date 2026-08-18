from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from model.schema import RawTaxiTripSchema, filter_valid_trips


def _base_row() -> dict[str, object]:
    return {
        "tpep_pickup_datetime": pd.Timestamp("2024-01-08T13:30:00"),
        "passenger_count": 2.0,
        "trip_distance": 4.5,
        "PULocationID": 132.0,
        "DOLocationID": 236.0,
        "fare_amount": 20.0,
    }


def _raw_frame(*rows: dict[str, object]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    frame["tpep_pickup_datetime"] = pd.to_datetime(frame["tpep_pickup_datetime"])
    return frame


def test_filter_valid_trips_keeps_fully_valid_rows() -> None:
    raw = _raw_frame(_base_row(), _base_row())

    filtered = filter_valid_trips(raw)

    assert len(filtered) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"tpep_pickup_datetime": pd.NaT},
        {"passenger_count": 9.0},
        {"passenger_count": 0.0},
        {"trip_distance": 0.0},
        {"trip_distance": 150.0},
        {"fare_amount": 0.0},
        {"fare_amount": -5.0},
        {"fare_amount": 600.0},
        {"PULocationID": 0.0},
        {"DOLocationID": 0.0},
    ],
)
def test_filter_valid_trips_drops_rows_violating_a_single_constraint(
    overrides: dict[str, object],
) -> None:
    raw = _raw_frame(_base_row(), {**_base_row(), **overrides})

    filtered = filter_valid_trips(raw)

    assert len(filtered) == 1
    assert filtered.index.tolist() == [0]


def test_filter_valid_trips_returns_empty_frame_when_all_rows_invalid() -> None:
    raw = _raw_frame({**_base_row(), "fare_amount": -1.0})

    filtered = filter_valid_trips(raw)

    assert filtered.empty


def test_raw_taxi_trip_schema_rejects_unexpected_columns() -> None:
    raw = _raw_frame(_base_row())
    raw["unexpected_column"] = ["oops"]

    with pytest.raises(pa.errors.SchemaErrors):
        RawTaxiTripSchema.validate(raw, lazy=True)
