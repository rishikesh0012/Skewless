from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ml_skew.tracking import promote_model_version
from mlflow import MlflowClient


def test_promote_model_version_assigns_and_verifies_alias() -> None:
    client = Mock(spec=MlflowClient)
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="3",
        run_id="run-123",
    )

    result = promote_model_version(
        model_name="ml-skew-fare-regressor",
        version=3,
        alias="champion",
        client=client,
    )

    client.set_registered_model_alias.assert_called_once_with(
        name="ml-skew-fare-regressor",
        alias="champion",
        version="3",
    )
    client.get_model_version_by_alias.assert_called_once_with(
        "ml-skew-fare-regressor",
        "champion",
    )

    assert result.model_name == "ml-skew-fare-regressor"
    assert result.version == "3"
    assert result.alias == "champion"
    assert result.run_id == "run-123"
    assert result.model_uri == "models:/ml-skew-fare-regressor@champion"


@pytest.mark.parametrize(
    ("model_name", "version", "alias"),
    [
        ("", 1, "champion"),
        ("ml-skew-fare-regressor", 0, "champion"),
        ("ml-skew-fare-regressor", "invalid", "champion"),
        ("ml-skew-fare-regressor", 1, ""),
    ],
)
def test_promote_model_version_rejects_invalid_values(
    model_name: str,
    version: int | str,
    alias: str,
) -> None:
    with pytest.raises(ValueError):
        promote_model_version(
            model_name=model_name,
            version=version,
            alias=alias,
        )


def test_promote_model_version_detects_alias_mismatch() -> None:
    client = Mock(spec=MlflowClient)
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="2",
        run_id="run-456",
    )

    with pytest.raises(
        RuntimeError,
        match="expected 1, received 2",
    ):
        promote_model_version(
            model_name="ml-skew-fare-regressor",
            version=1,
            client=client,
        )
