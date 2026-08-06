from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import mlflow.data
import mlflow.lightgbm
from mlflow.models import infer_signature

import mlflow
from ml_skew.data.contracts import (
    TARGET_COLUMN,
    PreparedDataset,
)
from ml_skew.tracking.client import configure_tracking
from ml_skew.tracking.settings import TrackingSettings
from ml_skew.training.config import TrainingConfig
from ml_skew.training.train import TrainingResult


@dataclass(frozen=True, slots=True)
class LoggedTrainingRun:
    run_id: str
    experiment_id: str
    model_uri: str


def log_training_run(
    dataset: PreparedDataset,
    result: TrainingResult,
    config: TrainingConfig,
    *,
    settings: TrackingSettings | None = None,
    run_name: str,
    dataset_name: str,
    dataset_source: str,
    tags: Mapping[str, str] | None = None,
) -> LoggedTrainingRun:
    experiment = configure_tracking(settings)

    training_frame = dataset.features.astype("float64").copy()
    training_frame[TARGET_COLUMN] = dataset.target.astype("float64").to_numpy()

    tracked_dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
        training_frame,
        source=dataset_source,
        targets=TARGET_COLUMN,
        name=dataset_name,
    )

    input_example = dataset.features.head(5).astype("float64").copy()
    predictions = result.model.predict(input_example)
    signature = infer_signature(input_example, predictions)

    run_tags = {
        "project": "ml-skew",
        "model_family": "lightgbm",
        "task": "taxi-fare-regression",
    }
    run_tags.update(tags or {})

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        tags=run_tags,
    ) as run:
        mlflow.log_params(asdict(config))

        mlflow.log_params(
            {
                "rows_received": dataset.summary.rows_received,
                "rows_valid": dataset.summary.rows_valid,
                "rows_removed": dataset.summary.rows_removed,
                "training_rows": result.training_rows,
                "validation_rows": result.validation_rows,
                "feature_count": dataset.features.shape[1],
            }
        )

        mlflow.log_metrics(result.metrics.as_dict())
        mlflow.log_input(tracked_dataset, context="training")

        model_info = mlflow.lightgbm.log_model(
            lgb_model=result.model,
            name="model",
            input_example=input_example,
            signature=signature,
        )

    return LoggedTrainingRun(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        model_uri=model_info.model_uri,
    )
