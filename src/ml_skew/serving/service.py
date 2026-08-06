from __future__ import annotations

import os
from typing import Final, Protocol

import bentoml
import bentoml.mlflow  # type: ignore[import-not-found]
import numpy as np
import pandas as pd

from ml_skew.features.online_adapter import OnlineFeatureAdapter
from ml_skew.monitoring import detect_training_serving_skew
from ml_skew.serving.contracts import (
    FarePredictionRequest,
    FarePredictionResponse,
    MonitoredFarePredictionResponse,
    RawFarePredictionRequest,
    SkewReportResponse,
)

DEFAULT_MODEL_TAG: Final = "ml-skew-fare-regressor:ybaxmkurs6mumjrr"
MODEL_TAG_ENVIRONMENT_VARIABLE: Final = "BENTO_MODEL_TAG"


class PredictionModel(Protocol):
    def predict(self, data: pd.DataFrame) -> object: ...


def resolve_model_tag() -> str:
    model_tag = os.getenv(
        MODEL_TAG_ENVIRONMENT_VARIABLE,
        DEFAULT_MODEL_TAG,
    ).strip()

    if not model_tag:
        raise ValueError(f"{MODEL_TAG_ENVIRONMENT_VARIABLE} cannot be empty")

    return model_tag


def predict_fare(
    *,
    model: PredictionModel,
    request: FarePredictionRequest,
    model_tag: str,
) -> FarePredictionResponse:
    raw_predictions = model.predict(request.to_model_frame())
    predictions = np.asarray(raw_predictions).reshape(-1)

    if predictions.size != 1:
        raise RuntimeError("The fare model must return exactly one prediction")

    predicted_fare = float(predictions[0])

    if not np.isfinite(predicted_fare):
        raise RuntimeError("The fare model returned a non-finite prediction")

    return FarePredictionResponse(
        predicted_fare_amount=predicted_fare,
        model_tag=model_tag,
    )


def predict_monitored_fare(
    *,
    model: PredictionModel,
    request: RawFarePredictionRequest,
    model_tag: str,
) -> MonitoredFarePredictionResponse:
    trip = request.to_trip_input()

    skew_report = detect_training_serving_skew(
        trip,
        skew_mode=request.skew_mode,
    )

    online_features = OnlineFeatureAdapter(skew_mode=request.skew_mode).transform(trip)

    feature_request = FarePredictionRequest.model_validate(online_features.model_dump())

    prediction = predict_fare(
        model=model,
        request=feature_request,
        model_tag=model_tag,
    )

    return MonitoredFarePredictionResponse(
        predicted_fare_amount=prediction.predicted_fare_amount,
        model_tag=prediction.model_tag,
        skew=SkewReportResponse.from_report(skew_report),
    )


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class FarePredictionService:
    bento_model = bentoml.models.BentoModel(resolve_model_tag())

    def __init__(self) -> None:
        self.model_tag = str(self.bento_model.tag)
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(  # type: ignore[untyped-decorator]
        input_spec=FarePredictionRequest,
        output_spec=FarePredictionResponse,
    )
    def predict(
        self,
        **payload: object,
    ) -> FarePredictionResponse:
        request = FarePredictionRequest.model_validate(payload)

        return predict_fare(
            model=self.model,
            request=request,
            model_tag=self.model_tag,
        )

    @bentoml.api(  # type: ignore[untyped-decorator]
        route="/predict-raw",
        input_spec=RawFarePredictionRequest,
        output_spec=MonitoredFarePredictionResponse,
    )
    def predict_raw(
        self,
        **payload: object,
    ) -> MonitoredFarePredictionResponse:
        request = RawFarePredictionRequest.model_validate(payload)

        return predict_monitored_fare(
            model=self.model,
            request=request,
            model_tag=self.model_tag,
        )
