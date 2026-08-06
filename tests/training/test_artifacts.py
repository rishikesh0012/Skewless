import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from ml_skew.training.artifacts import save_training_artifacts
from ml_skew.training.config import TrainingConfig
from ml_skew.training.evaluation import RegressionMetrics
from ml_skew.training.train import TrainingResult


def test_training_artifacts_are_saved(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "trip_distance_miles": [1.0, 2.0, 3.0, 4.0],
            "pickup_hour": [8, 12, 17, 21],
        }
    )
    target = pd.Series([6.0, 9.0, 13.0, 16.0])

    model = LGBMRegressor(
        n_estimators=2,
        min_child_samples=1,
        verbosity=-1,
    )
    model.fit(features, target)

    result = TrainingResult(
        model=model,
        metrics=RegressionMetrics(
            mean_absolute_error=2.05,
            root_mean_squared_error=6.70,
            r2_score=0.90,
        ),
        training_rows=80,
        validation_rows=20,
    )

    config = TrainingConfig(n_estimators=10)

    paths = save_training_artifacts(
        result=result,
        config=config,
        output_directory=tmp_path,
    )

    assert paths.model.exists()
    assert paths.metrics.exists()
    assert paths.metadata.exists()

    loaded_model = joblib.load(paths.model)
    assert isinstance(loaded_model, LGBMRegressor)

    metrics = json.loads(paths.metrics.read_text())
    assert metrics["mean_absolute_error"] == 2.05
    assert metrics["root_mean_squared_error"] == 6.70
    assert metrics["r2_score"] == 0.90

    metadata = json.loads(paths.metadata.read_text())
    assert metadata["model_type"] == "LGBMRegressor"
    assert metadata["training_rows"] == 80
    assert metadata["validation_rows"] == 20
    assert metadata["feature_names"] == [
        "trip_distance_miles",
        "pickup_hour",
    ]
    assert metadata["training_config"]["n_estimators"] == 10
