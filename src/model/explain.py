from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import shap

from features import FEATURE_NAMES, FeatureVector

# A small, fixed set of representative trips spanning realistic ranges of every
# feature. Used only to compute a global (dataset-level) feature-importance
# summary, so explainability stays self-contained and doesn't depend on the
# raw training parquet being present at request time.
REFERENCE_TRIPS: tuple[FeatureVector, ...] = (
    FeatureVector(
        trip_distance_miles=1.2,
        passenger_count=1,
        pickup_location_id=48,
        dropoff_location_id=68,
        pickup_hour=8,
        pickup_day_of_week=0,
        pickup_month=1,
        is_weekend=0,
        is_rush_hour=1,
    ),
    FeatureVector(
        trip_distance_miles=4.5,
        passenger_count=2,
        pickup_location_id=132,
        dropoff_location_id=236,
        pickup_hour=13,
        pickup_day_of_week=0,
        pickup_month=1,
        is_weekend=0,
        is_rush_hour=0,
    ),
    FeatureVector(
        trip_distance_miles=12.0,
        passenger_count=1,
        pickup_location_id=138,
        dropoff_location_id=161,
        pickup_hour=17,
        pickup_day_of_week=4,
        pickup_month=6,
        is_weekend=0,
        is_rush_hour=1,
    ),
    FeatureVector(
        trip_distance_miles=0.8,
        passenger_count=4,
        pickup_location_id=79,
        dropoff_location_id=232,
        pickup_hour=23,
        pickup_day_of_week=5,
        pickup_month=12,
        is_weekend=1,
        is_rush_hour=0,
    ),
    FeatureVector(
        trip_distance_miles=25.0,
        passenger_count=1,
        pickup_location_id=132,
        dropoff_location_id=1,
        pickup_hour=6,
        pickup_day_of_week=2,
        pickup_month=7,
        is_weekend=0,
        is_rush_hour=0,
    ),
    FeatureVector(
        trip_distance_miles=3.0,
        passenger_count=3,
        pickup_location_id=161,
        dropoff_location_id=234,
        pickup_hour=19,
        pickup_day_of_week=6,
        pickup_month=3,
        is_weekend=1,
        is_rush_hour=0,
    ),
)


class TreeModel(Protocol):
    def predict(self, x: pd.DataFrame) -> object: ...


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    feature: str
    value: float
    shap_value: float


@dataclass(frozen=True, slots=True)
class PredictionExplanation:
    base_value: float
    predicted_fare_amount: float
    contributions: tuple[FeatureContribution, ...]


def _to_frame(features: FeatureVector) -> pd.DataFrame:
    return pd.DataFrame([features.ordered_values()], columns=FEATURE_NAMES, dtype="float64")


class FareExplainer:
    """SHAP-based explainability for the trained fare model.

    Uses `shap.TreeExplainer`, which supports tree-ensemble regressors
    (LightGBM, Random Forest) and needs no background dataset: contributions
    are computed exactly from the model's own tree structure.
    """

    def __init__(self, model: TreeModel) -> None:
        self._explainer = shap.TreeExplainer(model)

    def explain_prediction(self, features: FeatureVector) -> PredictionExplanation:
        explanation = self._explainer(_to_frame(features))
        shap_values = np.asarray(explanation.values[0], dtype=np.float64)
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])

        contributions = tuple(
            FeatureContribution(
                feature=name,
                value=float(getattr(features, name)),
                shap_value=float(shap_values[index]),
            )
            for index, name in enumerate(FEATURE_NAMES)
        )
        return PredictionExplanation(
            base_value=base_value,
            predicted_fare_amount=base_value + float(shap_values.sum()),
            contributions=contributions,
        )

    def global_feature_importance(
        self, sample: Sequence[FeatureVector] = REFERENCE_TRIPS
    ) -> dict[str, float]:
        frame = pd.concat([_to_frame(features) for features in sample], ignore_index=True)
        explanation = self._explainer(frame)
        mean_abs_shap = np.abs(np.asarray(explanation.values, dtype=np.float64)).mean(axis=0)
        return dict(zip(FEATURE_NAMES, (float(value) for value in mean_abs_shap), strict=True))
