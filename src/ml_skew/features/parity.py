from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import cast

from ml_skew.features.contracts import FeatureVector

NumericValue = int | float


@dataclass(frozen=True, slots=True)
class FeatureMismatch:
    feature: str
    offline_value: NumericValue
    online_value: NumericValue

    @property
    def absolute_difference(self) -> float:
        return abs(float(self.offline_value) - float(self.online_value))

    @property
    def relative_difference(self) -> float:
        denominator = max(abs(float(self.offline_value)), 1e-12)
        return self.absolute_difference / denominator


def compare_feature_vectors(
    offline: FeatureVector,
    online: FeatureVector,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-9,
) -> tuple[FeatureMismatch, ...]:
    mismatches: list[FeatureMismatch] = []

    for feature_name in FeatureVector.model_fields:
        offline_value = cast(
            NumericValue,
            getattr(offline, feature_name),
        )
        online_value = cast(
            NumericValue,
            getattr(online, feature_name),
        )

        if isclose(
            float(offline_value),
            float(online_value),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            continue

        mismatches.append(
            FeatureMismatch(
                feature=feature_name,
                offline_value=offline_value,
                online_value=online_value,
            )
        )

    return tuple(mismatches)
