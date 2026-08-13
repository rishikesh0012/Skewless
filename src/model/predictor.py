from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, FeatureVector

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "fare_model.joblib"
MODEL_PATH_ENVIRONMENT_VARIABLE = "SKEWLESS_MODEL_PATH"


class PredictionModel(Protocol):
    def predict(self, data: pd.DataFrame) -> object: ...


class FarePredictor:
    def __init__(self, model_path: Path | None = None) -> None:
        configured_path = os.getenv(MODEL_PATH_ENVIRONMENT_VARIABLE)
        self.model_path = (
            Path(configured_path) if configured_path else model_path or DEFAULT_MODEL_PATH
        )
        self.model_path = self.model_path.expanduser().resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run `python -m model.train` first."
            )

        loaded = joblib.load(self.model_path)
        if not hasattr(loaded, "predict"):
            raise TypeError("The saved artifact does not expose a predict method")
        self._model: PredictionModel = loaded

    @property
    def model_name(self) -> str:
        return self.model_path.name

    @property
    def metadata(self) -> Mapping[str, object]:
        metadata_path = self.model_path.with_name("metadata.json")
        if not metadata_path.is_file():
            return {}
        loaded = json.loads(metadata_path.read_text())
        return loaded if isinstance(loaded, dict) else {}

    def predict(self, features: FeatureVector) -> float:
        frame = pd.DataFrame(
            [features.ordered_values()],
            columns=FEATURE_NAMES,
            dtype="float64",
        )
        predictions = np.asarray(self._model.predict(frame), dtype=np.float64).reshape(-1)

        if predictions.size != 1:
            raise RuntimeError("The fare model must return exactly one prediction")

        prediction = float(predictions[0])
        if not np.isfinite(prediction):
            raise RuntimeError("The fare model returned a non-finite prediction")
        return prediction
