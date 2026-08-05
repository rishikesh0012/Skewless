from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class DataValidationError(ValueError):
    pass


def require_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    required = set(required_columns)
    missing = sorted(required.difference(frame.columns))

    if missing:
        formatted = ", ".join(missing)
        raise DataValidationError(f"Dataset is missing required columns: {formatted}")
