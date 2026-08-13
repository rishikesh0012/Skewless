from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_predictor
from features import FeatureVector


class FakePredictor:
    model_name = "test-model.joblib"
    metadata: ClassVar[dict[str, str]] = {"model_type": "FakeRegressor"}

    def predict(self, features: FeatureVector) -> float:
        return 3.0 + features.trip_distance_miles * 2.5


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_predictor] = FakePredictor
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


BASE_REQUEST = {
    "pickup_datetime": "2024-01-08T13:30:00Z",
    "pickup_location_id": 132,
    "dropoff_location_id": 236,
    "passenger_count": 2,
    "trip_distance_miles": 4.5,
}


def test_broken_mode_reports_distance_skew_and_scores_serving_vector(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict",
        json={**BASE_REQUEST, "feature_mode": "broken", "skew_mode": "distance_unit"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["parity"]["matched"] is False
    assert payload["parity"]["matched_count"] == 8
    assert payload["parity"]["mismatches"][0]["feature"] == "trip_distance_miles"
    assert payload["predicted_fare_amount"] == 3.0 + 7.242048 * 2.5


def test_correct_mode_ignores_fault_and_guarantees_parity(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={**BASE_REQUEST, "feature_mode": "correct", "skew_mode": "timezone"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["applied_skew_mode"] == "none"
    assert payload["requested_skew_mode"] == "timezone"
    assert payload["parity"]["matched"] is True
    assert payload["parity"]["matched_count"] == 9
    assert payload["training_features"] == payload["serving_features"]


def test_health_and_model_info_are_simple_get_endpoints(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "model_name": "test-model.joblib",
    }
    info = client.get("/model-info").json()
    assert info["model_name"] == "test-model.joblib"
    assert len(info["feature_names"]) == 9


@pytest.mark.parametrize("removed_mode", ["missing_value", "rush_hour_rule", "location_mapping"])
def test_removed_skew_modes_are_rejected(client: TestClient, removed_mode: str) -> None:
    response = client.post(
        "/predict",
        json={**BASE_REQUEST, "feature_mode": "broken", "skew_mode": removed_mode},
    )

    assert response.status_code == 422
