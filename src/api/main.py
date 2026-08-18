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
from model.drift import DriftMonitor, DriftStatus, load_reference_stats
from model.explain import FareExplainer
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


class FeatureContributionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    value: float
    shap_value: float


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_value: float
    predicted_fare_amount: float
    feature_mode: FeatureMode
    applied_skew_mode: SkewMode
    contributions: tuple[FeatureContributionResponse, ...]


class GlobalImportanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    feature_importance: dict[str, float]


class FeatureDriftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    psi: float
    status: DriftStatus
    reference_mean: float
    current_mean: float


class DriftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DriftStatus
    sample_count: int
    reference_available: bool
    features: tuple[FeatureDriftResponse, ...]


@lru_cache
def get_predictor() -> FarePredictor:
    return FarePredictor()


@lru_cache
def get_explainer() -> FareExplainer:
    return FareExplainer(get_predictor().raw_model)


@lru_cache
def get_drift_monitor() -> DriftMonitor:
    return DriftMonitor()


def _build_features(
    request: PredictionRequest,
) -> tuple[FeatureVector, FeatureVector, SkewMode]:
    trip = request.to_trip()

    if request.feature_mode is FeatureMode.BROKEN:
        training_features = canonical_transform(trip)
        serving_features = online_transform(trip, request.skew_mode)
        applied_skew_mode = request.skew_mode
    else:
        training_features = shared_transform(trip)
        serving_features = shared_transform(trip)
        applied_skew_mode = SkewMode.NONE

    return training_features, serving_features, applied_skew_mode


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
    drift_monitor: Annotated[DriftMonitor, Depends(get_drift_monitor)],
) -> PredictionResponse:
    training_features, serving_features, applied_skew_mode = _build_features(request)
    drift_monitor.record(serving_features)

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


@app.post("/explain", response_model=ExplanationResponse)
def explain(
    request: PredictionRequest,
    explainer: Annotated[FareExplainer, Depends(get_explainer)],
) -> ExplanationResponse:
    _, serving_features, applied_skew_mode = _build_features(request)
    explanation = explainer.explain_prediction(serving_features)

    return ExplanationResponse(
        base_value=explanation.base_value,
        predicted_fare_amount=explanation.predicted_fare_amount,
        feature_mode=request.feature_mode,
        applied_skew_mode=applied_skew_mode,
        contributions=tuple(
            FeatureContributionResponse(
                feature=item.feature, value=item.value, shap_value=item.shap_value
            )
            for item in explanation.contributions
        ),
    )


@app.get("/explain/global-importance", response_model=GlobalImportanceResponse)
def global_importance(
    predictor: Annotated[FarePredictor, Depends(get_predictor)],
    explainer: Annotated[FareExplainer, Depends(get_explainer)],
) -> GlobalImportanceResponse:
    return GlobalImportanceResponse(
        model_name=predictor.model_name,
        feature_importance=explainer.global_feature_importance(),
    )


@app.get("/monitoring/drift", response_model=DriftResponse)
def monitoring_drift(
    predictor: Annotated[FarePredictor, Depends(get_predictor)],
    drift_monitor: Annotated[DriftMonitor, Depends(get_drift_monitor)],
) -> DriftResponse:
    reference = load_reference_stats(predictor.model_path.with_name("reference_stats.json"))
    report = drift_monitor.compute_drift(reference)

    return DriftResponse(
        status=report.status,
        sample_count=report.sample_count,
        reference_available=report.reference_available,
        features=tuple(
            FeatureDriftResponse(
                feature=item.feature,
                psi=item.psi,
                status=item.status,
                reference_mean=item.reference_mean,
                current_mean=item.current_mean,
            )
            for item in report.features
        ),
    )
