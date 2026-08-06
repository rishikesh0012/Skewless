import pandas as pd
import pytest
from pydantic import ValidationError

from ml_skew.data.contracts import FEATURE_COLUMNS
from ml_skew.serving import FarePredictionRequest


def build_request() -> FarePredictionRequest:
    return FarePredictionRequest(
        trip_distance_miles=4.5,
        passenger_count=2,
        pickup_location_id=132,
        dropoff_location_id=236,
        pickup_hour=8,
        pickup_day_of_week=1,
        pickup_month=1,
        is_weekend=0,
        is_rush_hour=1,
    )


def test_request_creates_model_ready_frame() -> None:
    frame = build_request().to_model_frame()

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (1, len(FEATURE_COLUMNS))
    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert all(dtype == "float64" for dtype in frame.dtypes)
    assert frame.iloc[0]["trip_distance_miles"] == 4.5
    assert frame.iloc[0]["pickup_hour"] == 8.0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("trip_distance_miles", -1),
        ("pickup_location_id", 0),
        ("pickup_hour", 24),
        ("pickup_day_of_week", 7),
        ("pickup_month", 13),
        ("is_weekend", 2),
        ("is_rush_hour", -1),
    ],
)
def test_request_rejects_invalid_feature_values(
    field_name: str,
    invalid_value: int,
) -> None:
    payload = build_request().model_dump()
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        FarePredictionRequest.model_validate(payload)


def test_request_rejects_unknown_features() -> None:
    payload = build_request().model_dump()
    payload["unknown_feature"] = 10

    with pytest.raises(ValidationError):
        FarePredictionRequest.model_validate(payload)
