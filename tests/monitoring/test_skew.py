from datetime import UTC, datetime

import pytest

from ml_skew.features.contracts import TaxiTripInput
from ml_skew.features.fault_injector import SkewMode
from ml_skew.monitoring import detect_training_serving_skew


def build_trip() -> TaxiTripInput:
    return TaxiTripInput(
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
    )


def test_monitor_reports_feature_parity() -> None:
    report = detect_training_serving_skew(
        build_trip(),
        skew_mode=SkewMode.NONE,
    )

    assert report.skew_mode is SkewMode.NONE
    assert report.detected is False
    assert report.mismatch_count == 0
    assert report.mismatched_features == ()
    assert report.mismatches == ()


def test_monitor_detects_distance_unit_skew() -> None:
    report = detect_training_serving_skew(
        build_trip(),
        skew_mode=SkewMode.DISTANCE_UNIT,
    )

    assert report.skew_mode is SkewMode.DISTANCE_UNIT
    assert report.detected is True
    assert report.mismatch_count >= 1
    assert "trip_distance_miles" in report.mismatched_features

    distance_mismatch = next(
        mismatch for mismatch in report.mismatches if mismatch.feature == "trip_distance_miles"
    )

    assert distance_mismatch.offline_value != distance_mismatch.online_value


@pytest.mark.parametrize(
    ("field_name", "invalid_tolerance"),
    [
        ("relative_tolerance", -0.1),
        ("absolute_tolerance", -0.1),
    ],
)
def test_monitor_rejects_negative_tolerance(
    field_name: str,
    invalid_tolerance: float,
) -> None:
    arguments = {field_name: invalid_tolerance}

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        detect_training_serving_skew(
            build_trip(),
            **arguments,
        )
