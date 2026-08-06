import pytest

from ml_skew.training.config import TrainingConfig


def test_default_training_configuration() -> None:
    config = TrainingConfig()
    parameters = config.model_parameters()

    assert config.validation_size == 0.2
    assert config.random_state == 42
    assert parameters["objective"] == "regression_l1"
    assert parameters["n_estimators"] == 300
    assert parameters["learning_rate"] == 0.05


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_size", 0.0),
        ("validation_size", 1.0),
        ("n_estimators", 0),
        ("learning_rate", 0.0),
        ("num_leaves", 1),
        ("subsample", 1.1),
        ("colsample_bytree", 0.0),
        ("reg_alpha", -0.1),
        ("reg_lambda", -0.1),
    ],
)
def test_invalid_training_configuration_is_rejected(
    field: str,
    value: int | float,
) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})
