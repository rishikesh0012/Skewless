from __future__ import annotations

from datetime import UTC
from enum import StrEnum

from features import FeatureVector, TaxiTrip

MILES_TO_KILOMETRES = 1.609344


class SkewMode(StrEnum):
    NONE = "none"
    DISTANCE_UNIT = "distance_unit"
    TIMEZONE = "timezone"


def apply_fault(*, features: FeatureVector, trip: TaxiTrip, mode: SkewMode) -> FeatureVector:
    if mode is SkewMode.NONE:
        return features

    values = features.model_dump()

    if mode is SkewMode.DISTANCE_UNIT:
        values["trip_distance_miles"] = round(
            features.trip_distance_miles * MILES_TO_KILOMETRES,
            6,
        )
    elif mode is SkewMode.TIMEZONE:
        pickup_datetime = trip.pickup_datetime.astimezone(UTC)
        values["pickup_hour"] = pickup_datetime.hour
        values["pickup_day_of_week"] = pickup_datetime.weekday()
        values["pickup_month"] = pickup_datetime.month
        values["is_weekend"] = int(pickup_datetime.weekday() >= 5)
        values["is_rush_hour"] = int(
            7 <= pickup_datetime.hour < 10 or 16 <= pickup_datetime.hour < 19
        )

    return FeatureVector.model_validate(values)
