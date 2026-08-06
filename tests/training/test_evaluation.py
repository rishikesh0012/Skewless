import numpy as np
import pytest

from ml_skew.training.evaluation import (
    RegressionMetrics,
    evaluate_regression,
)


def test_perfect_predictions_return_expected_metrics() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 20.0, 30.0])

    metrics = evaluate_regression(actual, predicted)

    assert metrics.mean_absolute_error == 0.0
    assert metrics.root_mean_squared_error == 0.0
    assert metrics.r2_score == 1.0


def test_regression_metrics_are_calculated_correctly() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])

    metrics = evaluate_regression(actual, predicted)

    assert metrics.mean_absolute_error == pytest.approx(7.0 / 3.0)
    assert metrics.root_mean_squared_error == pytest.approx(np.sqrt(17.0 / 3.0))
    assert metrics.r2_score == pytest.approx(0.915)


def test_metrics_can_be_serialized() -> None:
    metrics = RegressionMetrics(
        mean_absolute_error=2.1,
        root_mean_squared_error=3.4,
        r2_score=0.82,
    )

    assert metrics.as_dict() == {
        "mean_absolute_error": 2.1,
        "root_mean_squared_error": 3.4,
        "r2_score": 0.82,
    }
