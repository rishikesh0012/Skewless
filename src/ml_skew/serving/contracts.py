from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ml_skew.data.contracts import FEATURE_COLUMNS
from ml_skew.features.contracts import TaxiTripInput
from ml_skew.features.fault_injector import SkewMode
from ml_skew.monitoring.skew import SkewReport


class FarePredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    trip_distance_miles: float = Field(ge=0)
    passenger_count: int = Field(ge=0)
    pickup_location_id: int = Field(ge=1)
    dropoff_location_id: int = Field(ge=1)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_day_of_week: int = Field(ge=0, le=6)
    pickup_month: int = Field(ge=1, le=12)
    is_weekend: Literal[0, 1]
    is_rush_hour: Literal[0, 1]

    def to_model_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [self.model_dump()],
            columns=FEATURE_COLUMNS,
        ).astype("float64")


class RawFarePredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    pickup_datetime: datetime
    pickup_location_id: int = Field(ge=1)
    dropoff_location_id: int = Field(ge=1)
    passenger_count: int | None = Field(
        default=None,
        ge=1,
        le=8,
    )
    trip_distance_miles: float = Field(gt=0, le=500)
    skew_mode: SkewMode = SkewMode.NONE

    def to_trip_input(self) -> TaxiTripInput:
        return TaxiTripInput(
            pickup_datetime=self.pickup_datetime,
            pickup_location_id=self.pickup_location_id,
            dropoff_location_id=self.dropoff_location_id,
            passenger_count=self.passenger_count,
            trip_distance_miles=self.trip_distance_miles,
        )


class FeatureMismatchResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    feature: str
    offline_value: int | float
    online_value: int | float


class SkewReportResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    detected: bool
    skew_mode: SkewMode
    mismatch_count: int = Field(ge=0)
    mismatches: tuple[FeatureMismatchResponse, ...]

    @classmethod
    def from_report(
        cls,
        report: SkewReport,
    ) -> SkewReportResponse:
        mismatches = tuple(
            FeatureMismatchResponse(
                feature=mismatch.feature,
                offline_value=mismatch.offline_value,
                online_value=mismatch.online_value,
            )
            for mismatch in report.mismatches
        )

        return cls(
            detected=report.detected,
            skew_mode=report.skew_mode,
            mismatch_count=report.mismatch_count,
            mismatches=mismatches,
        )


class FarePredictionResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    predicted_fare_amount: float
    model_tag: str


class MonitoredFarePredictionResponse(FarePredictionResponse):
    skew: SkewReportResponse
