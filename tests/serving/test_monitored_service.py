from datetime import UTC, datetime
from unittest.mock import Mock

import numpy as np

from ml_skew.features.fault_injector import SkewMode
from ml_skew.serving import RawFarePredictionRequest
from ml_skew.serving.service import predict_monitored_fare


def build_request(
    *,
    skew_mode: SkewMode = SkewMode.NONE,
) -> RawFarePredictionRequest:
    return RawFarePredictionRequest(
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
        skew_mode=skew_mode,
    )


def test_monitored_prediction_reports_feature_parity() -> None:
    model = Mock()
    model.predict.return_value = np.array([23.98])

    response = predict_monitored_fare(
        model=model,
        request=build_request(),
        model_tag="ml-skew-fare-regressor:test",
    )

    assert response.predicted_fare_amount == 23.98
    assert response.skew.detected is False
    assert response.skew.skew_mode is SkewMode.NONE
    assert response.skew.mismatch_count == 0

    model_input = model.predict.call_args.args[0]

    assert model_input.shape == (1, 9)
    assert model_input.iloc[0]["trip_distance_miles"] == 4.5


def test_monitored_prediction_detects_distance_skew() -> None:
    model = Mock()
    model.predict.return_value = np.array([30.5])

    response = predict_monitored_fare(
        model=model,
        request=build_request(
            skew_mode=SkewMode.DISTANCE_UNIT,
        ),
        model_tag="ml-skew-fare-regressor:test",
    )

    assert response.predicted_fare_amount == 30.5
    assert response.skew.detected is True
    assert response.skew.skew_mode is SkewMode.DISTANCE_UNIT
    assert "trip_distance_miles" in {mismatch.feature for mismatch in response.skew.mismatches}

    model_input = model.predict.call_args.args[0]

    assert model_input.iloc[0]["trip_distance_miles"] != 4.5
