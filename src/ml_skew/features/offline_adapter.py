from ml_skew.features.canonical import build_features
from ml_skew.features.contracts import FeatureVector, TaxiTripInput


class OfflineFeatureAdapter:
    def transform(self, trip: TaxiTripInput) -> FeatureVector:
        return build_features(trip)
