import numpy as np
import pandas as pd
import pytest

from ml_skew.data.contracts import (
    FEATURE_COLUMNS,
    PreparationSummary,
    PreparedDataset,
)
from ml_skew.monitoring.drift_data import (
    MILES_TO_KILOMETRES,
    build_drift_datasets,
)


def build_dataset(rows: int = 20) -> PreparedDataset:
    features = pd.DataFrame(
        {
            "trip_distance_miles": np.arange(
                1,
                rows + 1,
                dtype="float64",
            ),
            "passenger_count": np.ones(rows),
            "pickup_location_id": np.arange(1, rows + 1),
            "dropoff_location_id": np.arange(101, 101 + rows),
            "pickup_hour": np.arange(rows) % 24,
            "pickup_day_of_week": np.arange(rows) % 7,
            "pickup_month": (np.arange(rows) % 12) + 1,
            "is_weekend": (np.arange(rows) % 7 >= 5).astype(int),
            "is_rush_hour": np.zeros(rows),
        },
        columns=FEATURE_COLUMNS,
    )

    target = pd.Series(
        np.arange(rows, dtype="float64"),
        name="fare_amount",
    )

    return PreparedDataset(
        features=features,
        target=target,
        summary=PreparationSummary(
            rows_received=rows,
            rows_valid=rows,
            rows_removed=0,
        ),
    )


def test_build_drift_datasets_creates_distinct_periods() -> None:
    dataset = build_dataset()

    pair = build_drift_datasets(
        dataset,
        reference_rows=8,
        analysis_rows=6,
    )

    assert pair.reference.shape == (8, len(FEATURE_COLUMNS))
    assert pair.analysis.shape == (6, len(FEATURE_COLUMNS))
    assert tuple(pair.reference.columns) == FEATURE_COLUMNS
    assert tuple(pair.analysis.columns) == FEATURE_COLUMNS
    assert pair.shifted_feature == "trip_distance_miles"
    assert pair.shift_multiplier == MILES_TO_KILOMETRES


def test_analysis_dataset_contains_distance_shift() -> None:
    dataset = build_dataset()

    pair = build_drift_datasets(
        dataset,
        reference_rows=8,
        analysis_rows=6,
    )

    expected = (
        dataset.features.iloc[8:14]["trip_distance_miles"].reset_index(drop=True)
        * MILES_TO_KILOMETRES
    )

    pd.testing.assert_series_equal(
        pair.analysis["trip_distance_miles"],
        expected,
        check_names=True,
    )


def test_dataset_preparation_does_not_modify_source() -> None:
    dataset = build_dataset()
    original = dataset.features.copy(deep=True)

    build_drift_datasets(
        dataset,
        reference_rows=8,
        analysis_rows=6,
    )

    pd.testing.assert_frame_equal(dataset.features, original)


@pytest.mark.parametrize(
    ("reference_rows", "analysis_rows"),
    [
        (0, 5),
        (5, 0),
        (-1, 5),
    ],
)
def test_build_drift_datasets_rejects_invalid_row_counts(
    reference_rows: int,
    analysis_rows: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        build_drift_datasets(
            build_dataset(),
            reference_rows=reference_rows,
            analysis_rows=analysis_rows,
        )


def test_build_drift_datasets_rejects_insufficient_rows() -> None:
    with pytest.raises(
        ValueError,
        match="does not contain enough rows",
    ):
        build_drift_datasets(
            build_dataset(rows=10),
            reference_rows=8,
            analysis_rows=5,
        )


@pytest.mark.parametrize(
    "shift_multiplier",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_build_drift_datasets_rejects_invalid_multiplier(
    shift_multiplier: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive finite number",
    ):
        build_drift_datasets(
            build_dataset(),
            reference_rows=8,
            analysis_rows=6,
            shift_multiplier=shift_multiplier,
        )


def test_build_drift_datasets_rejects_unknown_feature() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown shifted feature",
    ):
        build_drift_datasets(
            build_dataset(),
            reference_rows=8,
            analysis_rows=6,
            shifted_feature="unknown_feature",
        )
