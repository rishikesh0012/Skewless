from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrackingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_",
        case_sensitive=False,
        extra="ignore",
    )

    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "ml-skew-baseline"
    registered_model_name: str = "ml-skew-fare-regressor"

    @field_validator(
        "tracking_uri",
        "experiment_name",
        "registered_model_name",
    )
    @classmethod
    def require_value(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("MLflow configuration values cannot be empty")

        return cleaned
