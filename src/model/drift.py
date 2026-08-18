from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, FeatureVector

DEFAULT_BUFFER_SIZE = 500
MINIMUM_SAMPLES_FOR_DRIFT = 30
DRIFT_BIN_COUNT = 10
PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25
_EPSILON = 1e-6


class DriftStatus(StrEnum):
    UNAVAILABLE = "unavailable"
    INSUFFICIENT_DATA = "insufficient_data"
    STABLE = "stable"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"


@dataclass(frozen=True, slots=True)
class FeatureReferenceStats:
    mean: float
    std: float
    minimum: float
    maximum: float
    bin_edges: tuple[float, ...]
    bin_proportions: tuple[float, ...]


ReferenceStats = dict[str, FeatureReferenceStats]


@dataclass(frozen=True, slots=True)
class FeatureDriftResult:
    feature: str
    psi: float
    status: DriftStatus
    reference_mean: float
    current_mean: float


@dataclass(frozen=True, slots=True)
class DriftReport:
    status: DriftStatus
    sample_count: int
    reference_available: bool
    features: tuple[FeatureDriftResult, ...]


def _quantile_bin_edges(column: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(column, quantiles))
    if edges.size < 2:
        # Constant column (e.g. every training row has the same value): fall
        # back to a single bin that straddles it.
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_proportions(column: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(column, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(counts.shape[0])
    return cast("np.ndarray", counts / total)


def compute_reference_stats(features: pd.DataFrame, bins: int = DRIFT_BIN_COUNT) -> ReferenceStats:
    """Summarize each training feature's distribution for later drift checks."""
    stats: ReferenceStats = {}
    for name in FEATURE_NAMES:
        column = features[name].to_numpy(dtype="float64")
        edges = _quantile_bin_edges(column, bins)
        proportions = _bin_proportions(column, edges)
        stats[name] = FeatureReferenceStats(
            mean=float(np.mean(column)),
            std=float(np.std(column)),
            minimum=float(np.min(column)),
            maximum=float(np.max(column)),
            bin_edges=tuple(float(v) for v in edges),
            bin_proportions=tuple(float(v) for v in proportions),
        )
    return stats


def save_reference_stats(stats: ReferenceStats, path: Path) -> None:
    payload = {
        name: {
            "mean": feature_stats.mean,
            "std": feature_stats.std,
            "min": feature_stats.minimum,
            "max": feature_stats.maximum,
            "bin_edges": list(feature_stats.bin_edges),
            "bin_proportions": list(feature_stats.bin_proportions),
        }
        for name, feature_stats in stats.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_reference_stats(path: Path) -> ReferenceStats | None:
    """Returns None if no reference stats have been saved yet (e.g. a model
    trained before drift monitoring was added), rather than raising."""
    if not path.is_file():
        return None

    payload = json.loads(path.read_text())
    return {
        name: FeatureReferenceStats(
            mean=float(entry["mean"]),
            std=float(entry["std"]),
            minimum=float(entry["min"]),
            maximum=float(entry["max"]),
            bin_edges=tuple(float(v) for v in entry["bin_edges"]),
            bin_proportions=tuple(float(v) for v in entry["bin_proportions"]),
        )
        for name, entry in payload.items()
    }


def _status_for_psi(psi: float) -> DriftStatus:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return DriftStatus.SIGNIFICANT
    if psi >= PSI_MODERATE_THRESHOLD:
        return DriftStatus.MODERATE
    return DriftStatus.STABLE


def _overall_status(statuses: list[DriftStatus]) -> DriftStatus:
    if DriftStatus.SIGNIFICANT in statuses:
        return DriftStatus.SIGNIFICANT
    if DriftStatus.MODERATE in statuses:
        return DriftStatus.MODERATE
    return DriftStatus.STABLE


def compute_feature_psi(reference: FeatureReferenceStats, sample: np.ndarray) -> float:
    """Population Stability Index between a feature's reference distribution
    and a current sample, using the reference's own bin edges."""
    counts, _ = np.histogram(sample, bins=np.array(reference.bin_edges))
    total = counts.sum()
    if total == 0:
        return 0.0

    current_proportions = counts / total
    psi = 0.0
    for reference_p, current_p in zip(reference.bin_proportions, current_proportions, strict=True):
        ref = max(reference_p, _EPSILON)
        cur = max(float(current_p), _EPSILON)
        psi += (cur - ref) * math.log(cur / ref)
    return float(psi)


class DriftMonitor:
    """In-memory buffer of recently served feature vectors, compared against
    training-time reference statistics via Population Stability Index (PSI).

    Intentionally lightweight: no database, no persistence. The buffer resets
    on process restart, which is expected for a single-worker demo deployment.
    """

    def __init__(self, buffer_size: int = DEFAULT_BUFFER_SIZE) -> None:
        self._buffer: deque[FeatureVector] = deque(maxlen=buffer_size)

    def record(self, features: FeatureVector) -> None:
        self._buffer.append(features)

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    def compute_drift(self, reference: ReferenceStats | None) -> DriftReport:
        sample_count = len(self._buffer)

        if reference is None:
            return DriftReport(
                status=DriftStatus.UNAVAILABLE,
                sample_count=sample_count,
                reference_available=False,
                features=(),
            )

        if sample_count < MINIMUM_SAMPLES_FOR_DRIFT:
            return DriftReport(
                status=DriftStatus.INSUFFICIENT_DATA,
                sample_count=sample_count,
                reference_available=True,
                features=(),
            )

        frame = pd.DataFrame(
            [item.ordered_values() for item in self._buffer],
            columns=FEATURE_NAMES,
            dtype="float64",
        )
        results = []
        for name in FEATURE_NAMES:
            reference_stats = reference[name]
            sample = frame[name].to_numpy(dtype="float64")
            psi = compute_feature_psi(reference_stats, sample)
            results.append(
                FeatureDriftResult(
                    feature=name,
                    psi=psi,
                    status=_status_for_psi(psi),
                    reference_mean=reference_stats.mean,
                    current_mean=float(np.mean(sample)),
                )
            )

        return DriftReport(
            status=_overall_status([r.status for r in results]),
            sample_count=sample_count,
            reference_available=True,
            features=tuple(results),
        )
