from __future__ import annotations

from zoneinfo import ZoneInfo

from features import FeatureVector, TaxiTrip

NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_PASSENGER_COUNT = 1


def _is_rush_hour(hour: int) -> bool:
    return 7 <= hour < 10 or 16 <= hour < 19


def transform_trip(trip: TaxiTrip) -> FeatureVector:
    """Reference transformation representing the training feature path."""
    pickup_datetime = trip.pickup_datetime.astimezone(NEW_YORK_TIMEZONE)
    passenger_count = (
        trip.passenger_count if trip.passenger_count is not None else DEFAULT_PASSENGER_COUNT
    )

    return FeatureVector(
        trip_distance_miles=round(trip.trip_distance_miles, 6),
        passenger_count=passenger_count,
        pickup_location_id=trip.pickup_location_id,
        dropoff_location_id=trip.dropoff_location_id,
        pickup_hour=pickup_datetime.hour,
        pickup_day_of_week=pickup_datetime.weekday(),
        pickup_month=pickup_datetime.month,
        is_weekend=int(pickup_datetime.weekday() >= 5),
        is_rush_hour=int(_is_rush_hour(pickup_datetime.hour)),
    )
