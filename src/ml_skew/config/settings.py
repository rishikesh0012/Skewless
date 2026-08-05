from pydantic_settings import BaseSettings, SettingsConfigDict

from ml_skew.features.fault_injector import SkewMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    skew_mode: SkewMode = SkewMode.NONE
