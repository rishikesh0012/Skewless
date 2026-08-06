from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_skew.data.contracts import REQUIRED_RAW_COLUMNS


def load_trip_records(
    path: Path,
    *,
    row_limit: int | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_parquet(
        path,
        columns=list(REQUIRED_RAW_COLUMNS),
    )

    if row_limit is not None:
        if row_limit < 1:
            raise ValueError("row_limit must be greater than zero")

        frame = frame.head(row_limit).copy()

    return frame
