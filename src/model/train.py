from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from features import FEATURE_NAMES, TaxiTrip
from features.shared import transform_trip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fare_model.joblib"
NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
MINIMUM_TRAINING_ROWS = 20

RAW_COLUMNS = (
    "tpep_pickup_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    validation_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20


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

    valid = (
        raw["tpep_pickup_datetime"].notna()
        & raw["trip_distance"].between(0.01, 100.0)
        & raw["fare_amount"].between(0.01, 500.0)
        & raw["passenger_count"].between(1, 8)
        & raw["PULocationID"].ge(1)
        & raw["DOLocationID"].ge(1)
    )
    raw = raw.loc[valid].copy()
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


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig | None = None,
) -> tuple[LGBMRegressor, dict[str, float]]:
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

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=active_config.n_estimators,
        learning_rate=active_config.learning_rate,
        num_leaves=active_config.num_leaves,
        min_child_samples=active_config.min_child_samples,
        random_state=active_config.random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    predictions = np.asarray(model.predict(x_validation), dtype=np.float64)
    metrics = {
        "mean_absolute_error": float(mean_absolute_error(y_validation, predictions)),
        "root_mean_squared_error": float(root_mean_squared_error(y_validation, predictions)),
        "r2_score": float(r2_score(y_validation, predictions)),
    }
    return model, metrics


def save_model(
    model: LGBMRegressor,
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
    arguments = parser.parse_args()

    config = TrainingConfig()
    features, target = load_training_data(
        arguments.dataset.expanduser().resolve(), arguments.row_limit
    )
    trained_model, metrics = train_model(features, target, config)
    save_model(
        trained_model, metrics, arguments.model_path.expanduser().resolve(), config, len(features)
    )

    print(f"Saved model: {arguments.model_path}")
    print(f"Training rows: {len(features):,}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
