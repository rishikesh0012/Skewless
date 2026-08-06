from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import pandas as pd

DEFAULT_DRIFT_METHOD = "kolmogorov_smirnov"


@dataclass(frozen=True, slots=True)
class DriftChunkResult:
    key: str
    chunk_index: int
    start_index: int
    end_index: int
    value: float
    upper_threshold: float
    alert: bool


@dataclass(frozen=True, slots=True)
class UnivariateDriftReport:
    feature_name: str
    method: str
    chunks: tuple[DriftChunkResult, ...]

    @property
    def drift_detected(self) -> bool:
        return any(chunk.alert for chunk in self.chunks)

    @property
    def alert_count(self) -> int:
        return sum(chunk.alert for chunk in self.chunks)

    @property
    def max_drift_value(self) -> float:
        return max(chunk.value for chunk in self.chunks)


def calculate_univariate_drift(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    *,
    feature_name: str,
    method: str = DEFAULT_DRIFT_METHOD,
    chunk_number: int = 5,
    upper_threshold: float = 0.1,
) -> UnivariateDriftReport:
    _validate_inputs(
        reference=reference,
        analysis=analysis,
        feature_name=feature_name,
        method=method,
        chunk_number=chunk_number,
        upper_threshold=upper_threshold,
    )

    nannyml = _load_nannyml()

    calculator = nannyml.UnivariateDriftCalculator(
        column_names=[feature_name],
        continuous_methods=[method],
        chunk_number=chunk_number,
        thresholds={method: nannyml.thresholds.ConstantThreshold(upper=upper_threshold)},
    )

    calculator.fit(reference)
    results = calculator.calculate(analysis)

    result_frame = results.filter(
        period="analysis",
        column_names=[feature_name],
        methods=[method],
    ).to_df()

    chunks = _extract_chunks(
        result_frame,
        feature_name=feature_name,
        method=method,
    )

    if not chunks:
        raise RuntimeError("NannyML returned no analysis chunks")

    return UnivariateDriftReport(
        feature_name=feature_name,
        method=method,
        chunks=chunks,
    )


def _extract_chunks(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    method: str,
) -> tuple[DriftChunkResult, ...]:
    value_column = (feature_name, method, "value")
    threshold_column = (
        feature_name,
        method,
        "upper_threshold",
    )
    alert_column = (feature_name, method, "alert")

    return tuple(
        DriftChunkResult(
            key=str(row[("chunk", "chunk", "key")]),
            chunk_index=int(row[("chunk", "chunk", "chunk_index")]),
            start_index=int(row[("chunk", "chunk", "start_index")]),
            end_index=int(row[("chunk", "chunk", "end_index")]),
            value=float(row[value_column]),
            upper_threshold=float(row[threshold_column]),
            alert=bool(row[alert_column]),
        )
        for _, row in frame.iterrows()
    )


def _validate_inputs(
    *,
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    feature_name: str,
    method: str,
    chunk_number: int,
    upper_threshold: float,
) -> None:
    if reference.empty:
        raise ValueError("reference data cannot be empty")

    if analysis.empty:
        raise ValueError("analysis data cannot be empty")

    if feature_name not in reference.columns:
        raise ValueError(f"Reference data is missing feature: {feature_name}")

    if feature_name not in analysis.columns:
        raise ValueError(f"Analysis data is missing feature: {feature_name}")

    if not method.strip():
        raise ValueError("method cannot be empty")

    if chunk_number <= 0:
        raise ValueError("chunk_number must be greater than zero")

    if not math.isfinite(upper_threshold) or upper_threshold <= 0:
        raise ValueError("upper_threshold must be a positive finite number")


def _load_nannyml() -> Any:
    try:
        return import_module("nannyml")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "NannyML is not installed. Activate .venv-monitoring before running drift analysis."
        ) from error
