import numpy as np
import pandas as pd
import pytest

from ml_skew.data.contracts import (
    FEATURE_COLUMNS,
    PreparationSummary,
    PreparedDataset,
)
from ml_skew.training import TrainingConfig, train_regressor


def build_dataset(rows: int = 120) -> PreparedDataset:
    random = np.random.default_rng(42)

    pickup_hour = np.arange(rows) % 24
    pickup_day = np.arange(rows) % 7
    trip_distance = random.uniform(0.5, 20.0, rows)
    passenger_count = random.integers(1, 5, rows)

    is_weekend = (pickup_day >= 5).astype("int32")
    is_rush_hour = (
        ((pickup_hour >= 7) & (pickup_hour < 10)) | ((pickup_hour >= 16) & (pickup_hour < 19))
    ).astype("int32")

    features = pd.DataFrame(
        {
            "trip_distance_miles": trip_distance,
            "passenger_count": passenger_count,
            "pickup_location_id": random.integers(1, 264, rows),
            "dropoff_location_id": random.integers(1, 264, rows),
            "pickup_hour": pickup_hour,
            "pickup_day_of_week": pickup_day,
            "pickup_month": (np.arange(rows) % 12) + 1,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush_hour,
        },
        columns=FEATURE_COLUMNS,
    )

    fare_amount = 3.0 + (trip_distance * 2.8) + (passenger_count * 0.5) + (is_rush_hour * 1.5)

    target = pd.Series(
        fare_amount,
        name="fare_amount",
        dtype="float64",
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


def test_train_regressor_returns_model_and_metrics() -> None:
    dataset = build_dataset()

    result = train_regressor(
        dataset,
        TrainingConfig(
            n_estimators=100,
            learning_rate=0.05,
            min_child_samples=5,
        ),
    )

    assert result.training_rows == 96
    assert result.validation_rows == 24
    assert result.metrics.mean_absolute_error < 5.0
    assert result.metrics.root_mean_squared_error < 7.0
    assert result.metrics.r2_score > 0.8


def test_training_results_are_reproducible() -> None:
    dataset = build_dataset()
    config = TrainingConfig(
        n_estimators=50,
        min_child_samples=5,
        random_state=42,
    )

    first = train_regressor(dataset, config)
    second = train_regressor(dataset, config)

    assert first.metrics.as_dict() == pytest.approx(second.metrics.as_dict())


def test_small_dataset_is_rejected() -> None:
    dataset = build_dataset(rows=10)

    with pytest.raises(
        ValueError,
        match="at least 20 valid rows",
    ):
        train_regressor(dataset)
