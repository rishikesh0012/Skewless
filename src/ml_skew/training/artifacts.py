from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib

from ml_skew.training.config import TrainingConfig
from ml_skew.training.train import TrainingResult


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    model: Path
    metrics: Path
    metadata: Path


def save_training_artifacts(
    result: TrainingResult,
    config: TrainingConfig,
    output_directory: Path,
) -> ArtifactPaths:
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = ArtifactPaths(
        model=output_directory / "model.joblib",
        metrics=output_directory / "metrics.json",
        metadata=output_directory / "metadata.json",
    )

    joblib.dump(result.model, paths.model)

    paths.metrics.write_text(
        json.dumps(
            result.metrics.as_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    metadata = {
        "model_type": type(result.model).__name__,
        "training_rows": result.training_rows,
        "validation_rows": result.validation_rows,
        "feature_names": list(result.model.feature_name_),
        "training_config": asdict(config),
    }

    paths.metadata.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return paths
