from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    validation_size: float = 0.2
    random_state: int = 42

    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20

    subsample: float = 0.9
    colsample_bytree: float = 0.9

    reg_alpha: float = 0.0
    reg_lambda: float = 0.1

    n_jobs: int = -1

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1")

        if self.n_estimators < 1:
            raise ValueError("n_estimators must be greater than zero")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero")

        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least 2")

        if self.min_child_samples < 1:
            raise ValueError("min_child_samples must be greater than zero")

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be between 0 and 1")

        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be between 0 and 1")

        if self.reg_alpha < 0.0:
            raise ValueError("reg_alpha cannot be negative")

        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda cannot be negative")

    def model_parameters(self) -> dict[str, int | float | str]:
        return {
            "objective": "regression_l1",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": -1,
        }
