from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

from ml_skew.data.loader import load_trip_records
from ml_skew.data.preparation import prepare_training_data
from ml_skew.monitoring.drift_data import (
    MILES_TO_KILOMETRES,
    build_drift_datasets,
)
from ml_skew.monitoring.nannyml_drift import (
    UnivariateDriftReport,
    calculate_univariate_drift,
)

DEFAULT_DATASET_PATH: Final = Path("data/raw/yellow_tripdata_2024-01.parquet")
DEFAULT_OUTPUT_PATH: Final = Path("artifacts/monitoring/drift-report.json")
DEFAULT_SHIFTED_FEATURE: Final = "trip_distance_miles"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run statistical feature-drift analysis with NannyML.")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the NYC Yellow Taxi Parquet dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path used for the generated JSON drift report.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=2_000,
        help="Maximum number of source records to load.",
    )
    parser.add_argument(
        "--reference-rows",
        type=int,
        default=1_000,
        help="Number of validated reference-period rows.",
    )
    parser.add_argument(
        "--analysis-rows",
        type=int,
        default=500,
        help="Number of validated analysis-period rows.",
    )
    parser.add_argument(
        "--shifted-feature",
        default=DEFAULT_SHIFTED_FEATURE,
        help="Feature whose analysis distribution is shifted.",
    )
    parser.add_argument(
        "--shift-multiplier",
        type=float,
        default=MILES_TO_KILOMETRES,
        help="Multiplier used to create the shifted distribution.",
    )
    parser.add_argument(
        "--chunk-number",
        type=int,
        default=5,
        help="Number of analysis chunks.",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=0.1,
        help="Upper drift-alert threshold.",
    )
    return parser


def run_drift_analysis(
    *,
    dataset_path: Path,
    output_path: Path,
    row_limit: int,
    reference_rows: int,
    analysis_rows: int,
    shifted_feature: str,
    shift_multiplier: float,
    chunk_number: int,
    upper_threshold: float,
) -> UnivariateDriftReport:
    resolved_dataset_path = dataset_path.expanduser().resolve()
    resolved_output_path = output_path.expanduser().resolve()

    if not resolved_dataset_path.is_file():
        raise FileNotFoundError(f"Taxi dataset was not found: {resolved_dataset_path}")

    if row_limit <= 0:
        raise ValueError("row_limit must be greater than zero")

    frame = load_trip_records(
        resolved_dataset_path,
        row_limit=row_limit,
    )
    dataset = prepare_training_data(frame)

    drift_data = build_drift_datasets(
        dataset,
        reference_rows=reference_rows,
        analysis_rows=analysis_rows,
        shifted_feature=shifted_feature,
        shift_multiplier=shift_multiplier,
    )

    report = calculate_univariate_drift(
        reference=drift_data.reference,
        analysis=drift_data.analysis,
        feature_name=drift_data.shifted_feature,
        chunk_number=chunk_number,
        upper_threshold=upper_threshold,
    )

    payload = {
        "dataset": str(resolved_dataset_path),
        "rows_received": dataset.summary.rows_received,
        "rows_valid": dataset.summary.rows_valid,
        "reference_rows": len(drift_data.reference),
        "analysis_rows": len(drift_data.analysis),
        "shifted_feature": drift_data.shifted_feature,
        "shift_multiplier": drift_data.shift_multiplier,
        "method": report.method,
        "drift_detected": report.drift_detected,
        "alert_count": report.alert_count,
        "chunk_count": len(report.chunks),
        "max_drift_value": report.max_drift_value,
        "chunks": [asdict(chunk) for chunk in report.chunks],
    }

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    resolved_output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("\nStatistical drift analysis completed")
    print(f"Feature:          {report.feature_name}")
    print(f"Method:           {report.method}")
    print(f"Drift detected:   {report.drift_detected}")
    print(f"Alert count:      {report.alert_count}")
    print(f"Analysis chunks:  {len(report.chunks)}")
    print(f"Maximum value:    {report.max_drift_value:.6f}")
    print(f"Report:           {resolved_output_path}")

    return report


def main() -> None:
    arguments = build_parser().parse_args()

    run_drift_analysis(
        dataset_path=arguments.dataset,
        output_path=arguments.output,
        row_limit=arguments.row_limit,
        reference_rows=arguments.reference_rows,
        analysis_rows=arguments.analysis_rows,
        shifted_feature=arguments.shifted_feature,
        shift_multiplier=arguments.shift_multiplier,
        chunk_number=arguments.chunk_number,
        upper_threshold=arguments.upper_threshold,
    )


if __name__ == "__main__":
    main()
