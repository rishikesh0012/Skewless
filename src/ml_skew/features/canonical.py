from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from zoneinfo import ZoneInfo

from ml_skew.features.contracts import FeatureVector, TaxiTripInput

NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")

EARTH_RADIUS_MILES = 3_958.7613
DEFAULT_PASSENGER_COUNT = 1


def _haversine_distance_miles(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
) -> float:
    pickup_latitude_radians = radians(pickup_latitude)
    dropoff_latitude_radians = radians(dropoff_latitude)

    latitude_delta = radians(dropoff_latitude - pickup_latitude)
    longitude_delta = radians(dropoff_longitude - pickup_longitude)

    haversine = (
        sin(latitude_delta / 2) ** 2
        + cos(pickup_latitude_radians)
        * cos(dropoff_latitude_radians)
        * sin(longitude_delta / 2) ** 2
    )

    central_angle = 2 * asin(sqrt(haversine))
    return EARTH_RADIUS_MILES * central_angle


def _is_rush_hour(hour: int) -> bool:
    return 7 <= hour < 10 or 16 <= hour < 19


def build_features(trip: TaxiTripInput) -> FeatureVector:
    pickup_datetime = trip.pickup_datetime.astimezone(NEW_YORK_TIMEZONE)

    straight_line_distance = _haversine_distance_miles(
        pickup_latitude=trip.pickup_latitude,
        pickup_longitude=trip.pickup_longitude,
        dropoff_latitude=trip.dropoff_latitude,
        dropoff_longitude=trip.dropoff_longitude,
    )

    passenger_count = (
        trip.passenger_count
        if trip.passenger_count is not None
        else DEFAULT_PASSENGER_COUNT
    )

    return FeatureVector(
        trip_distance_miles=round(trip.trip_distance_miles, 6),
        straight_line_distance_miles=round(straight_line_distance, 6),
        passenger_count=passenger_count,
        pickup_hour=pickup_datetime.hour,
        pickup_day_of_week=pickup_datetime.weekday(),
        pickup_month=pickup_datetime.month,
        is_weekend=int(pickup_datetime.weekday() >= 5),
        is_rush_hour=int(_is_rush_hour(pickup_datetime.hour)),
    )
