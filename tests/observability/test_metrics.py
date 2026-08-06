from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
)

from ml_skew.features.fault_injector import SkewMode
from ml_skew.observability import record_monitored_prediction


def test_record_clean_prediction_updates_only_gauge() -> None:
    registry = CollectorRegistry()

    counter = Counter(
        "test_skew_detections",
        "Test skew detections.",
        labelnames=("skew_mode",),
        registry=registry,
    )
    gauge = Gauge(
        "test_latest_prediction",
        "Test latest prediction.",
        registry=registry,
    )

    record_monitored_prediction(
        predicted_fare_amount=23.5,
        skew_mode=SkewMode.NONE,
        skew_detected=False,
        skew_counter=counter,
        prediction_gauge=gauge,
    )

    assert registry.get_sample_value("test_latest_prediction") == 23.5
    assert (
        registry.get_sample_value(
            "test_skew_detections_total",
            {"skew_mode": "none"},
        )
        is None
    )


def test_record_skewed_prediction_increments_counter() -> None:
    registry = CollectorRegistry()

    counter = Counter(
        "test_skew_detections",
        "Test skew detections.",
        labelnames=("skew_mode",),
        registry=registry,
    )
    gauge = Gauge(
        "test_latest_prediction",
        "Test latest prediction.",
        registry=registry,
    )

    record_monitored_prediction(
        predicted_fare_amount=31.63,
        skew_mode=SkewMode.DISTANCE_UNIT,
        skew_detected=True,
        skew_counter=counter,
        prediction_gauge=gauge,
    )

    assert registry.get_sample_value("test_latest_prediction") == 31.63
    assert (
        registry.get_sample_value(
            "test_skew_detections_total",
            {"skew_mode": "distance_unit"},
        )
        == 1.0
    )
