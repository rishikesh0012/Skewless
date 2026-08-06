from __future__ import annotations

from prometheus_client import Counter, Gauge

from ml_skew.features.fault_injector import SkewMode

SKEW_DETECTIONS = Counter(
    "ml_skew_training_serving_skew_detections",
    ("Number of monitored prediction requests where training-serving skew was detected."),
    labelnames=("skew_mode",),
)

LATEST_PREDICTED_FARE = Gauge(
    "ml_skew_latest_predicted_fare_amount",
    "Latest fare prediction returned by the monitored endpoint.",
)


def record_monitored_prediction(
    *,
    predicted_fare_amount: float,
    skew_mode: SkewMode,
    skew_detected: bool,
    skew_counter: Counter = SKEW_DETECTIONS,
    prediction_gauge: Gauge = LATEST_PREDICTED_FARE,
) -> None:
    prediction_gauge.set(predicted_fare_amount)

    if skew_detected:
        skew_counter.labels(
            skew_mode=skew_mode.value,
        ).inc()
