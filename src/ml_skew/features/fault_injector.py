from __future__ import annotations

from datetime import UTC
from enum import StrEnum

from ml_skew.features.contracts import FeatureVector, TaxiTripInput

MILES_TO_KILOMETRES = 1.609344


class SkewMode(StrEnum):
    NONE = "none"
    DISTANCE_UNIT = "distance_unit"
    TIMEZONE = "timezone"
    MISSING_VALUE = "missing_value"
    RUSH_HOUR_RULE = "rush_hour_rule"


def apply_fault(
    features: FeatureVector,
    trip: TaxiTripInput,
    mode: SkewMode,
) -> FeatureVector:
    if mode is SkewMode.NONE:
        return features

    values = features.model_dump()

    match mode:
        case SkewMode.DISTANCE_UNIT:
            values["trip_distance_miles"] = round(
                features.trip_distance_miles * MILES_TO_KILOMETRES,
                6,
            )

        case SkewMode.TIMEZONE:
            pickup_datetime = trip.pickup_datetime.astimezone(UTC)

            values["pickup_hour"] = pickup_datetime.hour
            values["pickup_day_of_week"] = pickup_datetime.weekday()
            values["pickup_month"] = pickup_datetime.month
            values["is_weekend"] = int(pickup_datetime.weekday() >= 5)

        case SkewMode.MISSING_VALUE:
            if trip.passenger_count is None:
                values["passenger_count"] = 0

        case SkewMode.RUSH_HOUR_RULE:
            hour = features.pickup_hour
            values["is_rush_hour"] = int(
                6 <= hour < 9 or 15 <= hour < 18
            )

    return FeatureVector.model_validate(values)
