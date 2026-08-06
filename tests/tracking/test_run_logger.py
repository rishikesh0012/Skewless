from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

import mlflow
from ml_skew.data.contracts import (
    FEATURE_COLUMNS,
    PreparationSummary,
    PreparedDataset,
)
from ml_skew.tracking import (
    TrackingSettings,
    log_training_run,
)
from ml_skew.training import TrainingConfig, train_regressor


def build_dataset(rows: int = 100) -> PreparedDataset:
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

    target = pd.Series(
        3.0 + (trip_distance * 2.8) + (passenger_count * 0.5) + (is_rush_hour * 1.5),
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


def test_training_run_is_logged(tmp_path: Path) -> None:
    database_path = tmp_path / "mlflow.db"
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()

    settings = TrackingSettings(
        tracking_uri=f"sqlite:///{database_path}",
        experiment_name="test-training-runs",
        registered_model_name="test-fare-regressor",
    )

    previous_tracking_uri = mlflow.get_tracking_uri()

    try:
        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.create_experiment(
            settings.experiment_name,
            artifact_location=artifact_directory.as_uri(),
        )

        dataset = build_dataset()
        config = TrainingConfig(
            n_estimators=20,
            learning_rate=0.1,
            min_child_samples=5,
        )
        training_result = train_regressor(dataset, config)

        logged_run = log_training_run(
            dataset=dataset,
            result=training_result,
            config=config,
            settings=settings,
            run_name="integration-test",
            dataset_name="synthetic-taxi-data",
            dataset_source=str(tmp_path / "source.parquet"),
            tags={"environment": "test"},
        )

        client = MlflowClient(tracking_uri=settings.tracking_uri)
        stored_run = client.get_run(logged_run.run_id)

        assert stored_run.info.experiment_id == logged_run.experiment_id
        assert stored_run.data.params["n_estimators"] == "20"
        assert stored_run.data.params["training_rows"] == "80"
        assert stored_run.data.tags["model_family"] == "lightgbm"
        assert stored_run.data.tags["environment"] == "test"
        assert "mean_absolute_error" in stored_run.data.metrics
        assert stored_run.inputs.dataset_inputs
        assert logged_run.model_uri
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
