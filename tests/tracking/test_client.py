from pathlib import Path

import mlflow
from ml_skew.tracking import (
    TrackingSettings,
    configure_tracking,
)


def test_configure_tracking_creates_experiment(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "mlflow.db"

    settings = TrackingSettings(
        tracking_uri=f"sqlite:///{database_path}",
        experiment_name="test-experiment",
        registered_model_name="test-model",
    )

    previous_tracking_uri = mlflow.get_tracking_uri()

    try:
        experiment = configure_tracking(settings)

        assert mlflow.get_tracking_uri() == settings.tracking_uri
        assert experiment.name == "test-experiment"
        assert experiment.experiment_id
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)


def test_tracking_settings_read_environment_variables(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:6000",
    )
    monkeypatch.setenv(
        "MLFLOW_EXPERIMENT_NAME",
        "environment-experiment",
    )
    monkeypatch.setenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "environment-model",
    )

    settings = TrackingSettings()

    assert settings.tracking_uri == "http://localhost:6000"
    assert settings.experiment_name == "environment-experiment"
    assert settings.registered_model_name == "environment-model"
