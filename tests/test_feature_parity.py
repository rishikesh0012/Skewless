from datetime import datetime

import pytest
from pydantic import ValidationError

from ml_skew.features import (
    OfflineFeatureAdapter,
    OnlineFeatureAdapter,
    SkewMode,
    TaxiTripInput,
    compare_feature_vectors,
)


@pytest.fixture
def taxi_trip() -> TaxiTripInput:
    return TaxiTripInput(
        pickup_datetime=datetime.fromisoformat("2026-08-06T08:30:00-04:00"),
        pickup_location_id=161,
        dropoff_location_id=236,
        passenger_count=1,
        trip_distance_miles=3.5,
    )


def test_offline_and_online_features_match(
    taxi_trip: TaxiTripInput,
) -> None:
    offline = OfflineFeatureAdapter().transform(taxi_trip)
    online = OnlineFeatureAdapter(skew_mode=SkewMode.NONE).transform(taxi_trip)

    assert compare_feature_vectors(offline, online) == ()


def test_distance_unit_skew_is_detected(
    taxi_trip: TaxiTripInput,
) -> None:
    offline = OfflineFeatureAdapter().transform(taxi_trip)
    online = OnlineFeatureAdapter(skew_mode=SkewMode.DISTANCE_UNIT).transform(taxi_trip)

    mismatches = compare_feature_vectors(offline, online)

    assert len(mismatches) == 1
    assert mismatches[0].feature == "trip_distance_miles"
    assert mismatches[0].offline_value == 3.5
    assert mismatches[0].online_value == pytest.approx(5.632704)


def test_missing_value_skew_is_detected() -> None:
    trip = TaxiTripInput(
        pickup_datetime=datetime.fromisoformat("2026-08-06T12:00:00-04:00"),
        pickup_location_id=161,
        dropoff_location_id=236,
        passenger_count=None,
        trip_distance_miles=3.5,
    )

    offline = OfflineFeatureAdapter().transform(trip)
    online = OnlineFeatureAdapter(skew_mode=SkewMode.MISSING_VALUE).transform(trip)

    mismatches = compare_feature_vectors(offline, online)

    assert len(mismatches) == 1
    assert mismatches[0].feature == "passenger_count"
    assert mismatches[0].offline_value == 1
    assert mismatches[0].online_value == 0


def test_rush_hour_rule_skew_is_detected() -> None:
    trip = TaxiTripInput(
        pickup_datetime=datetime.fromisoformat("2026-08-06T09:30:00-04:00"),
        pickup_location_id=161,
        dropoff_location_id=236,
        passenger_count=1,
        trip_distance_miles=3.5,
    )

    offline = OfflineFeatureAdapter().transform(trip)
    online = OnlineFeatureAdapter(skew_mode=SkewMode.RUSH_HOUR_RULE).transform(trip)

    mismatches = compare_feature_vectors(offline, online)

    assert len(mismatches) == 1
    assert mismatches[0].feature == "is_rush_hour"


def test_location_mapping_skew_is_detected(
    taxi_trip: TaxiTripInput,
) -> None:
    offline = OfflineFeatureAdapter().transform(taxi_trip)
    online = OnlineFeatureAdapter(skew_mode=SkewMode.LOCATION_MAPPING).transform(taxi_trip)

    mismatches = compare_feature_vectors(offline, online)

    assert len(mismatches) == 1
    assert mismatches[0].feature == "pickup_location_id"
    assert mismatches[0].offline_value == 161
    assert mismatches[0].online_value == 162


def test_naive_pickup_datetime_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="pickup_datetime must include a timezone offset",
    ):
        TaxiTripInput(
            pickup_datetime=datetime(2026, 8, 6, 8, 30),
            pickup_location_id=161,
            dropoff_location_id=236,
            passenger_count=1,
            trip_distance_miles=3.5,
        )
