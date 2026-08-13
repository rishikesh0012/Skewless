from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from features import FEATURE_NAMES, FeatureVector, TaxiTrip
from features.canonical import transform_trip as canonical_transform
from features.faults import SkewMode
from features.online import transform_trip as online_transform
from features.parity import FeatureMismatch, compare_feature_vectors
from features.shared import transform_trip as shared_transform
from model.predictor import FarePredictor


class FeatureMode(StrEnum):
    BROKEN = "broken"
    CORRECT = "correct"


class PredictionRequest(TaxiTrip):
    feature_mode: FeatureMode = FeatureMode.BROKEN
    skew_mode: SkewMode = SkewMode.NONE

    def to_trip(self) -> TaxiTrip:
        return TaxiTrip.model_validate(self.model_dump(exclude={"feature_mode", "skew_mode"}))


class FeatureComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    training_value: int | float
    serving_value: int | float
    matched: bool
    absolute_difference: float


class MismatchResponse(FeatureComparison):
    relative_difference: float

    @classmethod
    def from_mismatch(cls, mismatch: FeatureMismatch) -> MismatchResponse:
        return cls(
            feature=mismatch.feature,
            training_value=mismatch.training_value,
            serving_value=mismatch.serving_value,
            matched=False,
            absolute_difference=mismatch.absolute_difference,
            relative_difference=mismatch.relative_difference,
        )


class ParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matched: bool
    matched_count: int = Field(ge=0, le=9)
    mismatch_count: int = Field(ge=0, le=9)
    total_features: Literal[9] = 9
    comparisons: tuple[FeatureComparison, ...]
    mismatches: tuple[MismatchResponse, ...]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predicted_fare_amount: float
    model_name: str
    feature_mode: FeatureMode
    requested_skew_mode: SkewMode
    applied_skew_mode: SkewMode
    training_features: FeatureVector
    serving_features: FeatureVector
    parity: ParityReport


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_name: str


class ModelInfoResponse(BaseModel):
    model_name: str
    feature_names: tuple[str, ...]
    metadata: dict[str, object]


@lru_cache
def get_predictor() -> FarePredictor:
    return FarePredictor()


def _build_comparisons(
    training_features: FeatureVector,
    serving_features: FeatureVector,
) -> tuple[FeatureComparison, ...]:
    mismatch_names = {
        mismatch.feature
        for mismatch in compare_feature_vectors(training_features, serving_features)
    }
    return tuple(
        FeatureComparison(
            feature=name,
            training_value=getattr(training_features, name),
            serving_value=getattr(serving_features, name),
            matched=name not in mismatch_names,
            absolute_difference=abs(
                float(getattr(training_features, name)) - float(getattr(serving_features, name))
            ),
        )
        for name in FEATURE_NAMES
    )


app = FastAPI(
    title="Skewless",
    summary="Training-Serving Feature Parity",
    description=(
        "A focused demonstration of how duplicated feature transformations create skew, "
        "and how one shared transformation eliminates it."
    ),
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    predictor: Annotated[FarePredictor, Depends(get_predictor)],
) -> PredictionResponse:
    trip = request.to_trip()

    if request.feature_mode is FeatureMode.BROKEN:
        training_features = canonical_transform(trip)
        serving_features = online_transform(trip, request.skew_mode)
        applied_skew_mode = request.skew_mode
    else:
        training_features = shared_transform(trip)
        serving_features = shared_transform(trip)
        applied_skew_mode = SkewMode.NONE

    mismatches = compare_feature_vectors(training_features, serving_features)
    mismatch_responses = tuple(MismatchResponse.from_mismatch(item) for item in mismatches)
    comparisons = _build_comparisons(training_features, serving_features)

    return PredictionResponse(
        predicted_fare_amount=predictor.predict(serving_features),
        model_name=predictor.model_name,
        feature_mode=request.feature_mode,
        requested_skew_mode=request.skew_mode,
        applied_skew_mode=applied_skew_mode,
        training_features=training_features,
        serving_features=serving_features,
        parity=ParityReport(
            matched=not mismatches,
            matched_count=len(FEATURE_NAMES) - len(mismatches),
            mismatch_count=len(mismatches),
            comparisons=comparisons,
            mismatches=mismatch_responses,
        ),
    )


@app.get("/health", response_model=HealthResponse)
def health(
    predictor: Annotated[FarePredictor, Depends(get_predictor)],
) -> HealthResponse:
    return HealthResponse(model_name=predictor.model_name)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    predictor: Annotated[FarePredictor, Depends(get_predictor)],
) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=predictor.model_name,
        feature_names=FEATURE_NAMES,
        metadata=dict(predictor.metadata),
    )
