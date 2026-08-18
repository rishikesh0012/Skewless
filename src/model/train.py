from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast
from zoneinfo import ZoneInfo

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from features import FEATURE_NAMES, TaxiTrip
from features.shared import transform_trip
from model.drift import compute_reference_stats, save_reference_stats
from model.schema import filter_valid_trips

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fare_model.joblib"
NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
MINIMUM_TRAINING_ROWS = 20
RANDOM_FOREST_N_ESTIMATORS = 200
SELECTION_METRIC = "root_mean_squared_error"
MLFLOW_EXPERIMENT_NAME = "skewless-fare-model"
DEFAULT_MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()

RAW_COLUMNS = (
    "tpep_pickup_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
)


class RegressorLike(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series) -> object: ...
    def predict(self, x: pd.DataFrame) -> object: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    validation_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20


@dataclass(frozen=True, slots=True)
class ModelResult:
    name: str
    model: RegressorLike
    metrics: dict[str, float]


def load_training_data(
    path: Path, row_limit: int | None = 100_000
) -> tuple[pd.DataFrame, pd.Series]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_parquet(path, columns=list(RAW_COLUMNS))
    if row_limit is not None:
        if row_limit < 1:
            raise ValueError("row_limit must be greater than zero")
        raw = raw.head(row_limit).copy()

    raw["tpep_pickup_datetime"] = pd.to_datetime(raw["tpep_pickup_datetime"], errors="coerce")
    for column in RAW_COLUMNS[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["passenger_count"] = raw["passenger_count"].fillna(1)

    raw = filter_valid_trips(raw).copy()
    if raw.empty:
        raise ValueError("No valid training rows remain after validation")

    feature_rows: list[dict[str, int | float]] = []
    rows = cast(
        "Iterator[tuple[object, object, object, object, object, object]]",
        raw.loc[:, RAW_COLUMNS].itertuples(index=False, name=None),
    )
    for pickup, passengers, distance, pickup_zone, dropoff_zone, _fare in rows:
        pickup_datetime = cast("pd.Timestamp", pickup).to_pydatetime()
        if pickup_datetime.tzinfo is None:
            pickup_datetime = pickup_datetime.replace(tzinfo=NEW_YORK_TIMEZONE)

        trip = TaxiTrip(
            pickup_datetime=pickup_datetime,
            passenger_count=int(cast("float", passengers)),
            trip_distance_miles=float(cast("float", distance)),
            pickup_location_id=int(cast("float", pickup_zone)),
            dropoff_location_id=int(cast("float", dropoff_zone)),
        )
        feature_rows.append(transform_trip(trip).model_dump())

    features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    target = raw["fare_amount"].astype("float64").reset_index(drop=True)
    return features, target


def _build_candidates(config: TrainingConfig) -> dict[str, RegressorLike]:
    return {
        "ridge": Ridge(random_state=config.random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            random_state=config.random_state,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            objective="regression_l1",
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def compare_models(
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig | None = None,
) -> list[ModelResult]:
    """Train each candidate regressor on the same split and score it with MAE, RMSE, and R2."""
    if len(features) != len(target):
        raise ValueError("Features and target must contain the same number of rows")
    if len(features) < MINIMUM_TRAINING_ROWS:
        raise ValueError(f"Training requires at least {MINIMUM_TRAINING_ROWS} valid rows")

    active_config = config or TrainingConfig()
    x_train, x_validation, y_train, y_validation = train_test_split(
        features,
        target,
        test_size=active_config.validation_size,
        random_state=active_config.random_state,
        shuffle=True,
    )

    results: list[ModelResult] = []
    for name, model in _build_candidates(active_config).items():
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_validation), dtype=np.float64)
        metrics = {
            "mean_absolute_error": float(mean_absolute_error(y_validation, predictions)),
            "root_mean_squared_error": float(root_mean_squared_error(y_validation, predictions)),
            "r2_score": float(r2_score(y_validation, predictions)),
        }
        results.append(ModelResult(name=name, model=model, metrics=metrics))

    return results


def select_best_model(results: list[ModelResult]) -> ModelResult:
    if not results:
        raise ValueError("results must contain at least one candidate")
    return min(results, key=lambda result: result.metrics[SELECTION_METRIC])


def save_comparison(
    results: list[ModelResult],
    selected_name: str,
    comparison_path: Path,
) -> None:
    ranked = sorted(results, key=lambda result: result.metrics[SELECTION_METRIC])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_metric": SELECTION_METRIC,
        "selected_model": selected_name,
        "candidates": [
            {
                "name": result.name,
                "model_type": type(result.model).__name__,
                "metrics": result.metrics,
            }
            for result in ranked
        ],
    }
    comparison_path.write_text(json.dumps(payload, indent=2) + "\n")


def log_comparison_to_mlflow(
    results: list[ModelResult],
    selected_name: str,
    training_rows: int,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> None:
    """Log one MLflow run per candidate (params + MAE/RMSE/R2), with the model
    artifact attached only to the selected candidate's run."""
    # MLflow >=3 disables the plain filesystem store ("./mlruns") by default and
    # asks for a database backend. We intentionally stay on the local file store.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    for result in results:
        with mlflow.start_run(run_name=result.name):
            mlflow.log_params({"candidate": result.name, **result.model.get_params()})
            mlflow.log_param("training_rows", training_rows)
            mlflow.log_metrics(result.metrics)
            mlflow.set_tag("selected", result.name == selected_name)
            if result.name == selected_name:
                mlflow.sklearn.log_model(result.model, name="model")


def save_model(
    model: RegressorLike,
    metrics: dict[str, float],
    model_path: Path,
    config: TrainingConfig,
    training_rows: int,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "project": "Skewless — Training-Serving Feature Parity",
        "model_type": type(model).__name__,
        "feature_names": list(FEATURE_NAMES),
        "training_rows": training_rows,
        "trained_at": datetime.now().astimezone().isoformat(),
        "training_config": asdict(config),
        "metrics": metrics,
    }
    model_path.with_name("metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Skewless taxi-fare model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--row-limit", type=int, default=100_000)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=DEFAULT_MLFLOW_TRACKING_URI)
    arguments = parser.parse_args()

    config = TrainingConfig()
    features, target = load_training_data(
        arguments.dataset.expanduser().resolve(), arguments.row_limit
    )

    results = compare_models(features, target, config)
    best = select_best_model(results)
    log_comparison_to_mlflow(
        results, best.name, len(features), tracking_uri=arguments.mlflow_tracking_uri
    )

    model_path = arguments.model_path.expanduser().resolve()
    save_model(best.model, best.metrics, model_path, config, len(features))
    save_comparison(results, best.name, model_path.with_name("model_comparison.json"))
    reference_stats = compute_reference_stats(features)
    save_reference_stats(reference_stats, model_path.with_name("reference_stats.json"))

    print(f"Training rows: {len(features):,}")
    print(f"{'model':<15}{'mae':>12}{'rmse':>12}{'r2':>12}")
    for result in sorted(results, key=lambda result: result.metrics[SELECTION_METRIC]):
        marker = " *" if result.name == best.name else ""
        print(
            f"{result.name:<15}"
            f"{result.metrics['mean_absolute_error']:>12.4f}"
            f"{result.metrics['root_mean_squared_error']:>12.4f}"
            f"{result.metrics['r2_score']:>12.4f}"
            f"{marker}"
        )
    print(f"Selected model ({SELECTION_METRIC}): {best.name}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
