from fastapi.testclient import TestClient

from api.main import app

DEMO_REQUEST = {
    "pickup_datetime": "2024-01-08T13:30:00Z",
    "pickup_location_id": 132,
    "dropoff_location_id": 236,
    "passenger_count": 2,
    "trip_distance_miles": 4.5,
    "skew_mode": "distance_unit",
}


def test_real_model_broken_to_correct_demo() -> None:
    with TestClient(app) as client:
        broken = client.post(
            "/predict",
            json={**DEMO_REQUEST, "feature_mode": "broken"},
        )
        correct = client.post(
            "/predict",
            json={**DEMO_REQUEST, "feature_mode": "correct"},
        )

    assert broken.status_code == 200
    assert correct.status_code == 200

    broken_payload = broken.json()
    correct_payload = correct.json()

    assert broken_payload["parity"]["matched_count"] == 8
    assert broken_payload["parity"]["mismatches"][0]["feature"] == "trip_distance_miles"
    assert broken_payload["serving_features"]["trip_distance_miles"] == 7.242048

    assert correct_payload["parity"]["matched_count"] == 9
    assert correct_payload["training_features"] == correct_payload["serving_features"]
    assert correct_payload["applied_skew_mode"] == "none"

    assert broken_payload["predicted_fare_amount"] != correct_payload["predicted_fare_amount"]
