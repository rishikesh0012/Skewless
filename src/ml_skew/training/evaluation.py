from __future__ import annotations

from dataclasses import asdict, dataclass

from numpy.typing import ArrayLike
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    mean_absolute_error: float
    root_mean_squared_error: float
    r2_score: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_regression(
    actual: ArrayLike,
    predicted: ArrayLike,
) -> RegressionMetrics:
    return RegressionMetrics(
        mean_absolute_error=float(mean_absolute_error(actual, predicted)),
        root_mean_squared_error=float(root_mean_squared_error(actual, predicted)),
        r2_score=float(r2_score(actual, predicted)),
    )
