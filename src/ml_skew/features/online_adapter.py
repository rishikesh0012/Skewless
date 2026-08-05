from ml_skew.config.settings import Settings
from ml_skew.features.canonical import build_features
from ml_skew.features.contracts import FeatureVector, TaxiTripInput
from ml_skew.features.fault_injector import SkewMode, apply_fault


class OnlineFeatureAdapter:
    def __init__(self, skew_mode: SkewMode | None = None) -> None:
        self._skew_mode = (
            skew_mode
            if skew_mode is not None
            else Settings().skew_mode
        )

    def transform(self, trip: TaxiTripInput) -> FeatureVector:
        features = build_features(trip)

        return apply_fault(
            features=features,
            trip=trip,
            mode=self._skew_mode,
        )
