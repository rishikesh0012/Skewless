from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from features import FEATURE_NAMES, FeatureVector

NumericValue = int | float


@dataclass(frozen=True, slots=True)
class FeatureMismatch:
    feature: str
    training_value: NumericValue
    serving_value: NumericValue

    @property
    def absolute_difference(self) -> float:
        return abs(float(self.training_value) - float(self.serving_value))

    @property
    def relative_difference(self) -> float:
        denominator = max(abs(float(self.training_value)), 1e-12)
        return self.absolute_difference / denominator


def compare_feature_vectors(
    training: FeatureVector,
    serving: FeatureVector,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-9,
) -> tuple[FeatureMismatch, ...]:
    mismatches: list[FeatureMismatch] = []

    for feature_name in FEATURE_NAMES:
        training_value: NumericValue = getattr(training, feature_name)
        serving_value: NumericValue = getattr(serving, feature_name)

        if isclose(
            float(training_value),
            float(serving_value),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            continue

        mismatches.append(
            FeatureMismatch(
                feature=feature_name,
                training_value=training_value,
                serving_value=serving_value,
            )
        )

    return tuple(mismatches)
