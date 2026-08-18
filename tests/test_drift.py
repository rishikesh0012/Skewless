from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_drift_monitor, get_predictor
from features import FEATURE_NAMES, FeatureVector
from model.drift import (
    MINIMUM_SAMPLES_FOR_DRIFT,
    DriftMonitor,
    DriftStatus,
    compute_feature_psi,
    compute_reference_stats,
    load_reference_stats,
    save_reference_stats,
)


def _synthetic_features(rows: int = 300, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "trip_distance_miles": rng.uniform(0.5, 20.0, rows),
            "passenger_count": rng.integers(1, 5, rows).astype(float),
            "pickup_location_id": rng.integers(1, 260, rows).astype(float),
            "dropoff_location_id": rng.integers(1, 260, rows).astype(float),
            "pickup_hour": rng.integers(0, 24, rows).astype(float),
            "pickup_day_of_week": rng.integers(0, 7, rows).astype(float),
            "pickup_month": rng.integers(1, 13, rows).astype(float),
            "is_weekend": rng.integers(0, 2, rows).astype(float),
            "is_rush_hour": rng.integers(0, 2, rows).astype(float),
        }
    )[list(FEATURE_NAMES)]
    return frame


def _feature_vectors(frame: pd.DataFrame) -> list[FeatureVector]:
    return [
        FeatureVector(
            trip_distance_miles=row.trip_distance_miles,
            passenger_count=int(row.passenger_count),
            pickup_location_id=int(row.pickup_location_id),
            dropoff_location_id=int(row.dropoff_location_id),
            pickup_hour=int(row.pickup_hour),
            pickup_day_of_week=int(row.pickup_day_of_week),
            pickup_month=int(row.pickup_month),
            is_weekend=int(row.is_weekend),
            is_rush_hour=int(row.is_rush_hour),
        )
        for row in frame.itertuples(index=False)
    ]


# --- compute_reference_stats / save+load round trip ------------------------


def test_compute_reference_stats_covers_all_features() -> None:
    stats = compute_reference_stats(_synthetic_features())

    assert set(stats) == set(FEATURE_NAMES)
    for feature_stats in stats.values():
        assert feature_stats.bin_edges[0] == float("-inf")
        assert feature_stats.bin_edges[-1] == float("inf")
        assert sum(feature_stats.bin_proportions) == pytest.approx(1.0)


def test_save_and_load_reference_stats_round_trip(tmp_path: Path) -> None:
    stats = compute_reference_stats(_synthetic_features())
    path = tmp_path / "reference_stats.json"

    save_reference_stats(stats, path)
    loaded = load_reference_stats(path)

    assert loaded is not None
    assert set(loaded) == set(FEATURE_NAMES)
    assert loaded["trip_distance_miles"].mean == pytest.approx(stats["trip_distance_miles"].mean)
    assert loaded["trip_distance_miles"].bin_edges == stats["trip_distance_miles"].bin_edges


def test_load_reference_stats_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_reference_stats(tmp_path / "does_not_exist.json") is None


# --- PSI math ----------------------------------------------------------------


def test_compute_feature_psi_is_near_zero_for_the_reference_distribution_itself() -> None:
    frame = _synthetic_features()
    reference = compute_reference_stats(frame)
    sample = frame["trip_distance_miles"].to_numpy(dtype="float64")

    psi = compute_feature_psi(reference["trip_distance_miles"], sample)

    assert psi < 0.05


def test_compute_feature_psi_is_high_for_a_clearly_shifted_distribution() -> None:
    frame = _synthetic_features()
    reference = compute_reference_stats(frame)
    shifted_sample = frame["trip_distance_miles"].to_numpy(dtype="float64") + 500.0

    psi = compute_feature_psi(reference["trip_distance_miles"], shifted_sample)

    assert psi > 1.0


# --- DriftMonitor --------------------------------------------------------------


def test_drift_monitor_sample_count_increments_on_record() -> None:
    monitor = DriftMonitor()
    vector = _feature_vectors(_synthetic_features(rows=1))[0]

    assert monitor.sample_count == 0
    monitor.record(vector)
    monitor.record(vector)
    assert monitor.sample_count == 2


def test_drift_monitor_reports_unavailable_without_reference() -> None:
    monitor = DriftMonitor()

    report = monitor.compute_drift(None)

    assert report.status is DriftStatus.UNAVAILABLE
    assert report.reference_available is False
    assert report.features == ()


def test_drift_monitor_reports_insufficient_data_below_minimum_samples() -> None:
    frame = _synthetic_features()
    reference = compute_reference_stats(frame)
    monitor = DriftMonitor()
    for vector in _feature_vectors(frame.head(MINIMUM_SAMPLES_FOR_DRIFT - 1)):
        monitor.record(vector)

    report = monitor.compute_drift(reference)

    assert report.status is DriftStatus.INSUFFICIENT_DATA
    assert report.reference_available is True
    assert report.sample_count == MINIMUM_SAMPLES_FOR_DRIFT - 1
    assert report.features == ()


def test_drift_monitor_reports_stable_when_serving_matches_training() -> None:
    frame = _synthetic_features()
    reference = compute_reference_stats(frame)
    monitor = DriftMonitor()
    # Feed the reference's own rows back in as "current" traffic. A 100-row
    # subsample would introduce real sampling noise against decile bins
    # (flaky); comparing the reference against itself is the deterministic
    # way to assert "identical distribution => stable".
    for vector in _feature_vectors(frame):
        monitor.record(vector)

    report = monitor.compute_drift(reference)

    assert report.status is DriftStatus.STABLE
    assert {result.feature for result in report.features} == set(FEATURE_NAMES)
    assert all(result.status is DriftStatus.STABLE for result in report.features)


def test_drift_monitor_flags_a_feature_that_has_drifted() -> None:
    frame = _synthetic_features()
    reference = compute_reference_stats(frame)
    drifted = frame.tail(100).copy()
    drifted["trip_distance_miles"] = drifted["trip_distance_miles"] + 500.0
    monitor = DriftMonitor()
    for vector in _feature_vectors(drifted):
        monitor.record(vector)

    report = monitor.compute_drift(reference)

    assert report.status is DriftStatus.SIGNIFICANT
    distance_result = next(r for r in report.features if r.feature == "trip_distance_miles")
    assert distance_result.status is DriftStatus.SIGNIFICANT
    # An untouched feature should stay stable.
    passenger_result = next(r for r in report.features if r.feature == "passenger_count")
    assert passenger_result.status is DriftStatus.STABLE


# --- API wiring ----------------------------------------------------------------


class FakePredictor:
    model_name = "fake-model.joblib"
    metadata: ClassVar[dict[str, str]] = {}

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def predict(self, features: FeatureVector) -> float:
        return 10.0


def test_drift_endpoint_reports_unavailable_for_the_real_committed_model() -> None:
    # The committed models/fare_model.joblib predates drift monitoring, so it
    # has no reference_stats.json sibling yet. This must degrade gracefully,
    # not error.
    app.dependency_overrides[get_drift_monitor] = lambda: DriftMonitor()
    try:
        with TestClient(app) as client:
            response = client.get("/monitoring/drift")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["reference_available"] is False
    assert payload["status"] == "unavailable"
    assert payload["features"] == []


def test_predict_records_samples_and_drift_endpoint_reflects_them(tmp_path: Path) -> None:
    frame = _synthetic_features()
    reference_stats = compute_reference_stats(frame)
    model_path = tmp_path / "fake-model.joblib"
    save_reference_stats(reference_stats, model_path.with_name("reference_stats.json"))

    request_rows = frame.tail(MINIMUM_SAMPLES_FOR_DRIFT + 5)
    request_body = {
        "pickup_datetime": "2024-01-08T13:30:00Z",
        "pickup_location_id": 132,
        "dropoff_location_id": 236,
        "feature_mode": "correct",
    }

    monitor = DriftMonitor()
    app.dependency_overrides[get_predictor] = lambda: FakePredictor(model_path)
    app.dependency_overrides[get_drift_monitor] = lambda: monitor
    try:
        with TestClient(app) as client:
            for row in request_rows.itertuples(index=False):
                response = client.post(
                    "/predict",
                    json={
                        **request_body,
                        "passenger_count": int(row.passenger_count),
                        "trip_distance_miles": float(row.trip_distance_miles),
                    },
                )
                assert response.status_code == 200

            drift_response = client.get("/monitoring/drift")
    finally:
        app.dependency_overrides.clear()

    assert drift_response.status_code == 200
    payload = drift_response.json()
    assert payload["reference_available"] is True
    assert payload["sample_count"] == len(request_rows)
    assert {item["feature"] for item in payload["features"]} == set(FEATURE_NAMES)
    assert payload["status"] in {status.value for status in DriftStatus}
