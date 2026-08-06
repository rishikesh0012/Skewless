from __future__ import annotations

from dataclasses import dataclass

from mlflow import MlflowClient


@dataclass(frozen=True, slots=True)
class RegisteredModelAlias:
    model_name: str
    version: str
    alias: str
    model_uri: str
    run_id: str | None


def promote_model_version(
    *,
    model_name: str,
    version: int | str,
    alias: str = "champion",
    client: MlflowClient | None = None,
) -> RegisteredModelAlias:
    resolved_model_name = _require_value(
        model_name,
        field_name="model_name",
    )
    resolved_alias = _require_value(
        alias,
        field_name="alias",
    )
    resolved_version = _normalise_version(version)

    registry_client = client or MlflowClient()

    registry_client.set_registered_model_alias(
        name=resolved_model_name,
        alias=resolved_alias,
        version=resolved_version,
    )

    model_version = registry_client.get_model_version_by_alias(
        resolved_model_name,
        resolved_alias,
    )

    actual_version = str(model_version.version)

    if actual_version != resolved_version:
        raise RuntimeError(
            "Registered model alias did not resolve to the requested "
            f"version: expected {resolved_version}, received "
            f"{actual_version}"
        )

    return RegisteredModelAlias(
        model_name=resolved_model_name,
        version=actual_version,
        alias=resolved_alias,
        model_uri=(f"models:/{resolved_model_name}@{resolved_alias}"),
        run_id=model_version.run_id,
    )


def _require_value(value: str, *, field_name: str) -> str:
    resolved_value = value.strip()

    if not resolved_value:
        raise ValueError(f"{field_name} cannot be empty")

    return resolved_value


def _normalise_version(version: int | str) -> str:
    resolved_version = str(version).strip()

    if not resolved_version.isdigit() or int(resolved_version) < 1:
        raise ValueError("version must be a positive registered-model version")

    return resolved_version
