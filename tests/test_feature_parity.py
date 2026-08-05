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
        pickup_datetime=datetime.fromisoformat(
            "2026-08-06T08:30:00-04:00"
        ),
        pickup_longitude=-73.9857,
        pickup_latitude=40.7484,
        dropoff_longitude=-73.9851,
        dropoff_latitude=40.7580,
        passenger_count=1,
        trip_distance_miles=3.5,
    )


def test_offline_and_online_features_match(
    taxi_trip: TaxiTripInput,
) -> None:
    offline_features = OfflineFeatureAdapter().transform(taxi_trip)
    online_features = OnlineFeatureAdapter(
        skew_mode=SkewMode.NONE
    ).transform(taxi_trip)

    mismatches = compare_feature_vectors(
        offline=offline_features,
        online=online_features,
    )

    assert mismatches == ()


def test_distance_unit_skew_is_detected(
    taxi_trip: TaxiTripInput,
) -> None:
    offline_features = OfflineFeatureAdapter().transform(taxi_trip)
    online_features = OnlineFeatureAdapter(
        skew_mode=SkewMode.DISTANCE_UNIT
    ).transform(taxi_trip)

    mismatches = compare_feature_vectors(
        offline=offline_features,
        online=online_features,
    )

    assert len(mismatches) == 1

    mismatch = mismatches[0]

    assert mismatch.feature == "trip_distance_miles"
    assert mismatch.offline_value == 3.5
    assert mismatch.online_value == pytest.approx(5.632704)


def test_missing_value_skew_is_detected() -> None:
    taxi_trip = TaxiTripInput(
        pickup_datetime=datetime.fromisoformat(
            "2026-08-06T12:00:00-04:00"
        ),
        pickup_longitude=-73.9857,
        pickup_latitude=40.7484,
        dropoff_longitude=-73.9851,
        dropoff_latitude=40.7580,
        passenger_count=None,
        trip_distance_miles=3.5,
    )

    offline_features = OfflineFeatureAdapter().transform(taxi_trip)
    online_features = OnlineFeatureAdapter(
        skew_mode=SkewMode.MISSING_VALUE
    ).transform(taxi_trip)

    mismatches = compare_feature_vectors(
        offline=offline_features,
        online=online_features,
    )

    assert len(mismatches) == 1
    assert mismatches[0].feature == "passenger_count"
    assert mismatches[0].offline_value == 1
    assert mismatches[0].online_value == 0


def test_rush_hour_rule_skew_is_detected() -> None:
    taxi_trip = TaxiTripInput(
        pickup_datetime=datetime.fromisoformat(
            "2026-08-06T09:30:00-04:00"
        ),
        pickup_longitude=-73.9857,
        pickup_latitude=40.7484,
        dropoff_longitude=-73.9851,
        dropoff_latitude=40.7580,
        passenger_count=1,
        trip_distance_miles=3.5,
    )

    offline_features = OfflineFeatureAdapter().transform(taxi_trip)
    online_features = OnlineFeatureAdapter(
        skew_mode=SkewMode.RUSH_HOUR_RULE
    ).transform(taxi_trip)

    mismatches = compare_feature_vectors(
        offline=offline_features,
        online=online_features,
    )

    assert len(mismatches) == 1
    assert mismatches[0].feature == "is_rush_hour"
    assert mismatches[0].offline_value == 1
    assert mismatches[0].online_value == 0


def test_naive_pickup_datetime_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="pickup_datetime must include a timezone offset",
    ):
        TaxiTripInput(
            pickup_datetime=datetime(2026, 8, 6, 8, 30),
            pickup_longitude=-73.9857,
            pickup_latitude=40.7484,
            dropoff_longitude=-73.9851,
            dropoff_latitude=40.7580,
            passenger_count=1,
            trip_distance_miles=3.5,
        )
