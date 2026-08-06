from datetime import UTC, datetime

from ml_skew.features.contracts import TaxiTripInput
from ml_skew.features.fault_injector import SkewMode
from ml_skew.monitoring import detect_training_serving_skew
from ml_skew.serving import (
    MonitoredFarePredictionResponse,
    SkewReportResponse,
)


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


def test_clean_skew_report_response() -> None:
    report = detect_training_serving_skew(build_trip())

    response = SkewReportResponse.from_report(report)

    assert response.detected is False
    assert response.skew_mode is SkewMode.NONE
    assert response.mismatch_count == 0
    assert response.mismatches == ()


def test_skew_report_contains_mismatch_values() -> None:
    report = detect_training_serving_skew(
        build_trip(),
        skew_mode=SkewMode.DISTANCE_UNIT,
    )

    response = SkewReportResponse.from_report(report)

    assert response.detected is True
    assert response.mismatch_count >= 1

    mismatch = next(item for item in response.mismatches if item.feature == "trip_distance_miles")

    assert mismatch.offline_value != mismatch.online_value


def test_monitored_prediction_serializes_skew_report() -> None:
    report = detect_training_serving_skew(build_trip())

    response = MonitoredFarePredictionResponse(
        predicted_fare_amount=23.98,
        model_tag="ml-skew-fare-regressor:test",
        skew=SkewReportResponse.from_report(report),
    )

    payload = response.model_dump(mode="json")

    assert payload["predicted_fare_amount"] == 23.98
    assert payload["skew"]["detected"] is False
    assert payload["skew"]["skew_mode"] == "none"
