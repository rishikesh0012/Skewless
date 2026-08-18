from __future__ import annotations

from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class RawTaxiTripSchema(pa.DataFrameModel):
    """Schema for the raw NYC taxi columns after datetime/numeric coercion.

    Field constraints mirror the range checks `load_training_data` has always
    enforced (pickup datetime present, distance/fare/passenger ranges, valid
    zone ids), now expressed as a single, checkable source of truth.
    """

    tpep_pickup_datetime: Series[pd.Timestamp] = pa.Field(nullable=False)
    passenger_count: Series[float] = pa.Field(ge=1, le=8)
    trip_distance: Series[float] = pa.Field(ge=0.01, le=100.0)
    PULocationID: Series[float] = pa.Field(ge=1)
    DOLocationID: Series[float] = pa.Field(ge=1)
    fare_amount: Series[float] = pa.Field(ge=0.01, le=500.0)

    class Config:
        coerce = True
        strict = True


def filter_valid_trips(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate `raw` against RawTaxiTripSchema and drop any row that fails a
    check, mirroring the previous hand-rolled boolean-mask filter."""
    try:
        RawTaxiTripSchema.validate(raw, lazy=True)
    except pa.errors.SchemaErrors as exc:
        invalid_indices = exc.failure_cases["index"].dropna().unique()
        return cast("pd.DataFrame", raw.drop(index=invalid_indices))
    return raw
