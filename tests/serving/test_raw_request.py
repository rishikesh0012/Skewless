from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ml_skew.features.contracts import TaxiTripInput
from ml_skew.features.fault_injector import SkewMode
from ml_skew.serving import RawFarePredictionRequest


def build_request(
    *,
    skew_mode: SkewMode = SkewMode.NONE,
) -> RawFarePredictionRequest:
    return RawFarePredictionRequest(
        pickup_datetime=datetime(
            2024,
            1,
            8,
            8,
            30,
            tzinfo=UTC,
        ),
        pickup_location_id=132,
        dropoff_location_id=236,
        passenger_count=2,
        trip_distance_miles=4.5,
        skew_mode=skew_mode,
    )


def test_raw_request_creates_trip_input() -> None:
    request = build_request()

    trip = request.to_trip_input()

    assert isinstance(trip, TaxiTripInput)
    assert trip.pickup_datetime == request.pickup_datetime
    assert trip.pickup_location_id == 132
    assert trip.dropoff_location_id == 236
    assert trip.passenger_count == 2
    assert trip.trip_distance_miles == 4.5


def test_raw_request_defaults_to_no_skew() -> None:
    request = build_request()

    assert request.skew_mode is SkewMode.NONE


def test_raw_request_accepts_skew_mode() -> None:
    request = build_request(
        skew_mode=SkewMode.DISTANCE_UNIT,
    )

    assert request.skew_mode is SkewMode.DISTANCE_UNIT


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("pickup_location_id", 0),
        ("dropoff_location_id", 0),
        ("passenger_count", 9),
        ("trip_distance_miles", 0),
    ],
)
def test_raw_request_rejects_invalid_values(
    field_name: str,
    invalid_value: int,
) -> None:
    payload = build_request().model_dump()
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        RawFarePredictionRequest.model_validate(payload)


def test_raw_request_rejects_unknown_fields() -> None:
    payload = build_request().model_dump()
    payload["unknown_field"] = "invalid"

    with pytest.raises(ValidationError):
        RawFarePredictionRequest.model_validate(payload)
