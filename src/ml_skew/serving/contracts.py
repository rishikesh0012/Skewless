from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ml_skew.data.contracts import FEATURE_COLUMNS


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


class FarePredictionResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    predicted_fare_amount: float
    model_tag: str
