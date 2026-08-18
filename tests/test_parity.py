from datetime import UTC, datetime

from features import TaxiTrip
from features.canonical import transform_trip as canonical_transform
from features.faults import SkewMode
from features.online import transform_trip as online_transform
from features.parity import compare_feature_vectors
from features.shared import transform_trip as shared_transform


def build_trip() -> TaxiTrip:
    return TaxiTrip(
        pickup_datetime=datetime(2024, 1, 8, 13, 30, tzinfo=UTC),
        pickup_location_id=132,
        dropoff_location_id=236,
        passenger_count=2,
        trip_distance_miles=4.5,
    )


def test_broken_mode_can_expose_skew() -> None:
    trip = build_trip()
    mismatches = compare_feature_vectors(
        canonical_transform(trip),
        online_transform(trip, SkewMode.DISTANCE_UNIT),
    )

    assert [mismatch.feature for mismatch in mismatches] == ["trip_distance_miles"]
    assert mismatches[0].absolute_difference > 0


def test_timezone_mode_can_expose_skew() -> None:
    trip = build_trip()
    mismatches = compare_feature_vectors(
        canonical_transform(trip),
        online_transform(trip, SkewMode.TIMEZONE),
    )

    assert {mismatch.feature for mismatch in mismatches} == {"pickup_hour", "is_rush_hour"}
    assert all(mismatch.absolute_difference > 0 for mismatch in mismatches)


def test_shared_path_has_perfect_parity() -> None:
    trip = build_trip()

    assert compare_feature_vectors(shared_transform(trip), shared_transform(trip)) == ()
