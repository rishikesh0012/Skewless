from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from ml_skew.data.loader import load_trip_records
from ml_skew.data.preparation import prepare_training_data
from ml_skew.tracking import TrackingSettings, log_training_run
from ml_skew.training.artifacts import save_training_artifacts
from ml_skew.training.config import TrainingConfig
from ml_skew.training.train import train_regressor

DEFAULT_DATASET_PATH: Final = Path("data/raw/yellow_tripdata_2024-01.parquet")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/baseline")
DEFAULT_ROW_LIMIT: Final = 100_000
DEFAULT_RUN_NAME: Final = "baseline-real-taxi-2024-01"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Train the baseline taxi-fare model and log the run to MLflow.")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the NYC Yellow Taxi Parquet dataset.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory used for local model artifacts.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=DEFAULT_ROW_LIMIT,
        help="Maximum number of source rows to load.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Name displayed for the run in MLflow.",
    )
    return parser


def run_baseline(
    *,
    dataset_path: Path,
    output_directory: Path,
    row_limit: int,
    run_name: str,
) -> None:
    resolved_dataset_path = dataset_path.expanduser().resolve()
    resolved_output_directory = output_directory.expanduser().resolve()

    if not resolved_dataset_path.is_file():
        raise FileNotFoundError(f"Taxi dataset was not found: {resolved_dataset_path}")

    if row_limit <= 0:
        raise ValueError("row_limit must be greater than zero")

    frame = load_trip_records(
        resolved_dataset_path,
        row_limit=row_limit,
    )
    dataset = prepare_training_data(frame)

    config = TrainingConfig()
    result = train_regressor(dataset, config)

    save_training_artifacts(
        result=result,
        config=config,
        output_directory=resolved_output_directory,
    )

    logged_run = log_training_run(
        dataset=dataset,
        result=result,
        config=config,
        settings=TrackingSettings(),
        run_name=run_name,
        dataset_name="nyc-yellow-taxi-2024-01",
        dataset_source=resolved_dataset_path.as_uri(),
        tags={
            "environment": "local",
            "pipeline": "baseline",
            "dataset_version": "2024-01",
        },
    )

    print("\nBaseline training completed")
    print(f"Rows received:   {dataset.summary.rows_received:,}")
    print(f"Rows valid:      {dataset.summary.rows_valid:,}")
    print(f"Training rows:   {result.training_rows:,}")
    print(f"Validation rows: {result.validation_rows:,}")

    for metric_name, metric_value in result.metrics.as_dict().items():
        print(f"{metric_name}: {metric_value:.6f}")

    print(f"Artifacts:       {resolved_output_directory}")
    print(f"MLflow run ID:   {logged_run.run_id}")
    print(f"MLflow model:    {logged_run.model_uri}")


def main() -> None:
    arguments = build_parser().parse_args()

    run_baseline(
        dataset_path=arguments.dataset,
        output_directory=arguments.output_directory,
        row_limit=arguments.row_limit,
        run_name=arguments.run_name,
    )


if __name__ == "__main__":
    main()
