from __future__ import annotations

from dataclasses import dataclass

from ml_skew.features.contracts import TaxiTripInput
from ml_skew.features.fault_injector import SkewMode
from ml_skew.features.offline_adapter import OfflineFeatureAdapter
from ml_skew.features.online_adapter import OnlineFeatureAdapter
from ml_skew.features.parity import (
    FeatureMismatch,
    compare_feature_vectors,
)


@dataclass(frozen=True, slots=True)
class SkewReport:
    skew_mode: SkewMode
    mismatches: tuple[FeatureMismatch, ...]

    @property
    def detected(self) -> bool:
        return bool(self.mismatches)

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)

    @property
    def mismatched_features(self) -> tuple[str, ...]:
        return tuple(mismatch.feature for mismatch in self.mismatches)


def detect_training_serving_skew(
    trip: TaxiTripInput,
    *,
    skew_mode: SkewMode = SkewMode.NONE,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-9,
) -> SkewReport:
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance cannot be negative")

    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance cannot be negative")

    offline_features = OfflineFeatureAdapter().transform(trip)
    online_features = OnlineFeatureAdapter(skew_mode=skew_mode).transform(trip)

    mismatches = compare_feature_vectors(
        offline_features,
        online_features,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )

    return SkewReport(
        skew_mode=skew_mode,
        mismatches=mismatches,
    )
