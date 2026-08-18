from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from features import FEATURE_NAMES
from model.train import (
    MINIMUM_TRAINING_ROWS,
    MLFLOW_EXPERIMENT_NAME,
    RAW_COLUMNS,
    TrainingConfig,
    compare_models,
    load_training_data,
    log_comparison_to_mlflow,
    save_comparison,
    save_model,
    select_best_model,
)

CANDIDATE_NAMES = {"ridge", "random_forest", "lightgbm"}


def _synthetic_dataset(rows: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "trip_distance_miles": rng.uniform(0.5, 20.0, rows),
            "passenger_count": rng.integers(1, 5, rows),
            "pickup_location_id": rng.integers(1, 260, rows),
            "dropoff_location_id": rng.integers(1, 260, rows),
            "pickup_hour": rng.integers(0, 24, rows),
            "pickup_day_of_week": rng.integers(0, 7, rows),
            "pickup_month": rng.integers(1, 13, rows),
            "is_weekend": rng.integers(0, 2, rows),
            "is_rush_hour": rng.integers(0, 2, rows),
        }
    )[list(FEATURE_NAMES)]
    target = pd.Series(5.0 + features["trip_distance_miles"] * 2.5, name="fare_amount")
    return features, target


def test_compare_models_trains_all_three_candidates_with_full_metrics() -> None:
    features, target = _synthetic_dataset()

    results = compare_models(features, target, TrainingConfig(n_estimators=20))

    assert {result.name for result in results} == CANDIDATE_NAMES
    for result in results:
        assert set(result.metrics) == {
            "mean_absolute_error",
            "root_mean_squared_error",
            "r2_score",
        }
        assert result.metrics["mean_absolute_error"] >= 0
        assert result.metrics["root_mean_squared_error"] >= 0


def test_compare_models_rejects_too_few_rows() -> None:
    features, target = _synthetic_dataset(rows=MINIMUM_TRAINING_ROWS - 1)

    with pytest.raises(ValueError, match="at least"):
        compare_models(features, target)


def test_compare_models_rejects_mismatched_lengths() -> None:
    features, target = _synthetic_dataset()

    with pytest.raises(ValueError, match="same number of rows"):
        compare_models(features, target.iloc[:-1])


def test_select_best_model_picks_lowest_rmse() -> None:
    features, target = _synthetic_dataset()
    results = compare_models(features, target, TrainingConfig(n_estimators=20))

    best = select_best_model(results)

    assert best.metrics["root_mean_squared_error"] == min(
        result.metrics["root_mean_squared_error"] for result in results
    )


def test_save_comparison_writes_all_candidates_and_selected_model(tmp_path: Path) -> None:
    features, target = _synthetic_dataset()
    results = compare_models(features, target, TrainingConfig(n_estimators=20))
    best = select_best_model(results)
    output_path = tmp_path / "model_comparison.json"

    save_comparison(results, best.name, output_path)

    payload = json.loads(output_path.read_text())
    assert payload["selection_metric"] == "root_mean_squared_error"
    assert payload["selected_model"] == best.name
    assert {candidate["name"] for candidate in payload["candidates"]} == CANDIDATE_NAMES
    assert payload["candidates"][0]["name"] == best.name


def test_save_model_writes_metadata_compatible_with_predictor(tmp_path: Path) -> None:
    features, target = _synthetic_dataset()
    results = compare_models(features, target, TrainingConfig(n_estimators=20))
    best = select_best_model(results)
    model_path = tmp_path / "fare_model.joblib"

    save_model(best.model, best.metrics, model_path, TrainingConfig(), len(features))

    assert model_path.is_file()
    metadata = json.loads(model_path.with_name("metadata.json").read_text())
    assert metadata["model_type"] == type(best.model).__name__
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["metrics"] == best.metrics
    assert hasattr(best.model, "predict")


def test_log_comparison_to_mlflow_logs_a_run_per_candidate(tmp_path: Path) -> None:
    features, target = _synthetic_dataset()
    results = compare_models(features, target, TrainingConfig(n_estimators=20))
    best = select_best_model(results)
    tracking_uri = (tmp_path / "mlruns").as_uri()

    log_comparison_to_mlflow(results, best.name, len(features), tracking_uri=tracking_uri)

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert {run.data.params["candidate"] for run in runs} == CANDIDATE_NAMES

    selected_run = next(run for run in runs if run.data.params["candidate"] == best.name)
    assert selected_run.data.tags["selected"] == "True"
    assert selected_run.data.metrics["root_mean_squared_error"] == pytest.approx(
        best.metrics["root_mean_squared_error"]
    )
    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id='{selected_run.info.run_id}'",
    )
    assert {model.name for model in logged_models} == {"model"}


def _write_raw_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    frame["tpep_pickup_datetime"] = pd.to_datetime(frame["tpep_pickup_datetime"])
    frame.to_parquet(path)


def test_load_training_data_filters_invalid_rows_via_pandera(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trips.parquet"
    _write_raw_parquet(
        dataset_path,
        [
            {
                "tpep_pickup_datetime": "2024-01-08T13:30:00",
                "passenger_count": 2,
                "trip_distance": 4.5,
                "PULocationID": 132,
                "DOLocationID": 236,
                "fare_amount": 20.0,
            },
            {
                "tpep_pickup_datetime": "2024-01-08T14:00:00",
                "passenger_count": 2,
                "trip_distance": 4.5,
                "PULocationID": 132,
                "DOLocationID": 236,
                "fare_amount": -5.0,  # invalid: fare_amount must be > 0
            },
        ],
    )

    features, target = load_training_data(dataset_path, row_limit=None)

    assert len(features) == 1
    assert len(target) == 1


def test_load_training_data_raises_when_all_rows_invalid(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trips.parquet"
    _write_raw_parquet(
        dataset_path,
        [
            {
                "tpep_pickup_datetime": "2024-01-08T13:30:00",
                "passenger_count": 2,
                "trip_distance": 4.5,
                "PULocationID": 132,
                "DOLocationID": 236,
                "fare_amount": -5.0,
            },
        ],
    )

    with pytest.raises(ValueError, match="No valid training rows remain"):
        load_training_data(dataset_path, row_limit=None)
