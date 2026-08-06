from unittest.mock import Mock

import numpy as np
import pytest

from ml_skew.serving import FarePredictionRequest
from ml_skew.serving.service import (
    DEFAULT_MODEL_TAG,
    predict_fare,
    resolve_model_tag,
)


def build_request() -> FarePredictionRequest:
    return FarePredictionRequest(
        trip_distance_miles=4.5,
        passenger_count=2,
        pickup_location_id=132,
        dropoff_location_id=236,
        pickup_hour=8,
        pickup_day_of_week=1,
        pickup_month=1,
        is_weekend=0,
        is_rush_hour=1,
    )


def test_resolve_model_tag_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENTO_MODEL_TAG", raising=False)

    assert resolve_model_tag() == DEFAULT_MODEL_TAG


def test_resolve_model_tag_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BENTO_MODEL_TAG",
        "ml-skew-fare-regressor:test-version",
    )

    assert resolve_model_tag() == "ml-skew-fare-regressor:test-version"


def test_predict_fare_returns_typed_response() -> None:
    model = Mock()
    model.predict.return_value = np.array([18.75])

    response = predict_fare(
        model=model,
        request=build_request(),
        model_tag="ml-skew-fare-regressor:test",
    )

    assert response.predicted_fare_amount == 18.75
    assert response.model_tag == "ml-skew-fare-regressor:test"

    model_input = model.predict.call_args.args[0]
    assert model_input.shape == (1, 9)


@pytest.mark.parametrize(
    "prediction",
    [
        np.array([]),
        np.array([10.0, 20.0]),
    ],
)
def test_predict_fare_rejects_invalid_result_size(
    prediction: np.ndarray,
) -> None:
    model = Mock()
    model.predict.return_value = prediction

    with pytest.raises(
        RuntimeError,
        match="exactly one prediction",
    ):
        predict_fare(
            model=model,
            request=build_request(),
            model_tag="test-model",
        )


@pytest.mark.parametrize(
    "prediction",
    [
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_predict_fare_rejects_non_finite_results(
    prediction: np.ndarray,
) -> None:
    model = Mock()
    model.predict.return_value = prediction

    with pytest.raises(
        RuntimeError,
        match="non-finite prediction",
    ):
        predict_fare(
            model=model,
            request=build_request(),
            model_tag="test-model",
        )
