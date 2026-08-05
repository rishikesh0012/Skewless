from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaxiTripInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pickup_datetime: datetime
    pickup_location_id: int = Field(ge=1)
    dropoff_location_id: int = Field(ge=1)
    passenger_count: int | None = Field(default=None, ge=1, le=8)
    trip_distance_miles: float = Field(gt=0.0, le=500.0)

    @field_validator("pickup_datetime")
    @classmethod
    def validate_pickup_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("pickup_datetime must include a timezone offset")

        return value


class FeatureVector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trip_distance_miles: float = Field(gt=0.0)
    passenger_count: int = Field(ge=0, le=8)
    pickup_location_id: int = Field(ge=1)
    dropoff_location_id: int = Field(ge=1)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_day_of_week: int = Field(ge=0, le=6)
    pickup_month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    is_rush_hour: int = Field(ge=0, le=1)
