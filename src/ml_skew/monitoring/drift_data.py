from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ml_skew.data.contracts import (
    FEATURE_COLUMNS,
    PreparedDataset,
)

MILES_TO_KILOMETRES = 1.609344


@dataclass(frozen=True, slots=True)
class DriftDatasetPair:
    reference: pd.DataFrame
    analysis: pd.DataFrame
    shifted_feature: str
    shift_multiplier: float


def build_drift_datasets(
    dataset: PreparedDataset,
    *,
    reference_rows: int,
    analysis_rows: int,
    shifted_feature: str = "trip_distance_miles",
    shift_multiplier: float = MILES_TO_KILOMETRES,
) -> DriftDatasetPair:
    _validate_row_count(reference_rows, name="reference_rows")
    _validate_row_count(analysis_rows, name="analysis_rows")

    if shifted_feature not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown shifted feature: {shifted_feature}")

    if not math.isfinite(shift_multiplier) or shift_multiplier <= 0:
        raise ValueError("shift_multiplier must be a positive finite number")

    required_rows = reference_rows + analysis_rows
    available_rows = len(dataset.features)

    if required_rows > available_rows:
        raise ValueError(
            "The prepared dataset does not contain enough rows: "
            f"required {required_rows}, available {available_rows}"
        )

    features = dataset.features.loc[:, FEATURE_COLUMNS].reset_index(drop=True).astype("float64")

    reference = features.iloc[:reference_rows].copy()
    analysis = features.iloc[reference_rows:required_rows].copy()

    reference = reference.reset_index(drop=True)
    analysis = analysis.reset_index(drop=True)

    analysis[shifted_feature] *= shift_multiplier

    return DriftDatasetPair(
        reference=reference,
        analysis=analysis,
        shifted_feature=shifted_feature,
        shift_multiplier=shift_multiplier,
    )


def _validate_row_count(value: int, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
