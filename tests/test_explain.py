from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMRegressor

from api.main import app
from features import FEATURE_NAMES, FeatureVector
from model.explain import REFERENCE_TRIPS, FareExplainer


def _fitted_model() -> LGBMRegressor:
    rng = np.random.default_rng(0)
    rows = 200
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
    target = 5.0 + frame["trip_distance_miles"] * 2.5 + frame["is_rush_hour"] * 3.0
    return LGBMRegressor(n_estimators=50, verbosity=-1).fit(frame, target)


@pytest.fixture(scope="module")
def explainer() -> FareExplainer:
    return FareExplainer(_fitted_model())


def _trip(**overrides: float) -> FeatureVector:
    base = REFERENCE_TRIPS[1].model_dump()
    base.update(overrides)
    return FeatureVector.model_validate(base)


def test_explain_prediction_covers_all_features_and_matches_additivity(
    explainer: FareExplainer,
) -> None:
    trip = _trip()
    explanation = explainer.explain_prediction(trip)

    assert [c.feature for c in explanation.contributions] == list(FEATURE_NAMES)
    total = explanation.base_value + sum(c.shap_value for c in explanation.contributions)
    assert total == pytest.approx(explanation.predicted_fare_amount)
    for contribution in explanation.contributions:
        assert contribution.value == getattr(trip, contribution.feature)


def test_explain_prediction_distance_contribution_grows_with_distance(
    explainer: FareExplainer,
) -> None:
    short_explanation = explainer.explain_prediction(_trip(trip_distance_miles=1.0))
    long_explanation = explainer.explain_prediction(_trip(trip_distance_miles=15.0))

    short_distance_shap = next(
        c.shap_value for c in short_explanation.contributions if c.feature == "trip_distance_miles"
    )
    long_distance_shap = next(
        c.shap_value for c in long_explanation.contributions if c.feature == "trip_distance_miles"
    )
    assert long_distance_shap > short_distance_shap


def test_global_feature_importance_covers_all_features_and_is_non_negative(
    explainer: FareExplainer,
) -> None:
    importance = explainer.global_feature_importance()

    assert set(importance) == set(FEATURE_NAMES)
    assert all(value >= 0 for value in importance.values())
    assert importance["trip_distance_miles"] > importance["passenger_count"]


def test_global_feature_importance_accepts_a_custom_sample(explainer: FareExplainer) -> None:
    importance = explainer.global_feature_importance(
        sample=[_trip(), _trip(trip_distance_miles=10.0)]
    )

    assert set(importance) == set(FEATURE_NAMES)


DEMO_REQUEST = {
    "pickup_datetime": "2024-01-08T13:30:00Z",
    "pickup_location_id": 132,
    "dropoff_location_id": 236,
    "passenger_count": 2,
    "trip_distance_miles": 4.5,
}


def test_explain_endpoint_contributions_sum_to_predicted_fare() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/explain",
            json={**DEMO_REQUEST, "feature_mode": "broken", "skew_mode": "none"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert {c["feature"] for c in payload["contributions"]} == set(FEATURE_NAMES)
    total = payload["base_value"] + sum(c["shap_value"] for c in payload["contributions"])
    assert total == pytest.approx(payload["predicted_fare_amount"])


def test_explain_endpoint_reflects_distance_unit_skew_in_serving_contribution() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/explain",
            json={**DEMO_REQUEST, "feature_mode": "broken", "skew_mode": "distance_unit"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["applied_skew_mode"] == "distance_unit"
    distance_contribution = next(
        c for c in payload["contributions"] if c["feature"] == "trip_distance_miles"
    )
    assert distance_contribution["value"] == pytest.approx(7.242048)


def test_global_importance_endpoint_returns_all_feature_names() -> None:
    with TestClient(app) as client:
        response = client.get("/explain/global-importance")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["feature_importance"]) == set(FEATURE_NAMES)
    assert all(value >= 0 for value in payload["feature_importance"].values())
