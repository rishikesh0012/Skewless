from mlflow.entities import Experiment

import mlflow
from ml_skew.tracking.settings import TrackingSettings


def configure_tracking(
    settings: TrackingSettings | None = None,
) -> Experiment:
    resolved_settings = settings or TrackingSettings()

    mlflow.set_tracking_uri(resolved_settings.tracking_uri)

    return mlflow.set_experiment(resolved_settings.experiment_name)
