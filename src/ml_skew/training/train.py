from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from ml_skew.data.contracts import PreparedDataset
from ml_skew.training.config import TrainingConfig
from ml_skew.training.evaluation import RegressionMetrics, evaluate_regression

MINIMUM_TRAINING_ROWS = 20


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: LGBMRegressor
    metrics: RegressionMetrics
    training_rows: int
    validation_rows: int


def train_regressor(
    dataset: PreparedDataset,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    training_config = config or TrainingConfig()

    if len(dataset.features) != len(dataset.target):
        raise ValueError("Features and target must contain the same number of rows")

    if len(dataset.features) < MINIMUM_TRAINING_ROWS:
        raise ValueError(f"Training requires at least {MINIMUM_TRAINING_ROWS} valid rows")

    split = train_test_split(
        dataset.features,
        dataset.target,
        test_size=training_config.validation_size,
        random_state=training_config.random_state,
        shuffle=True,
    )

    (
        training_features,
        validation_features,
        training_target,
        validation_target,
    ) = cast(
        "tuple[pd.DataFrame, pd.DataFrame, pd.Series[float], pd.Series[float]]",
        split,
    )

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=training_config.n_estimators,
        learning_rate=training_config.learning_rate,
        num_leaves=training_config.num_leaves,
        max_depth=training_config.max_depth,
        min_child_samples=training_config.min_child_samples,
        subsample=training_config.subsample,
        colsample_bytree=training_config.colsample_bytree,
        reg_alpha=training_config.reg_alpha,
        reg_lambda=training_config.reg_lambda,
        random_state=training_config.random_state,
        n_jobs=training_config.n_jobs,
        verbosity=-1,
    )
    model.fit(training_features, training_target)

    predictions = np.asarray(
        model.predict(validation_features),
        dtype=np.float64,
    )

    metrics = evaluate_regression(
        actual=validation_target.to_numpy(),
        predicted=predictions,
    )

    return TrainingResult(
        model=model,
        metrics=metrics,
        training_rows=len(training_features),
        validation_rows=len(validation_features),
    )
