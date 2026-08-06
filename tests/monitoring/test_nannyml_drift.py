import unittest
from importlib.util import find_spec

import pandas as pd

from ml_skew.monitoring.nannyml_drift import (
    calculate_univariate_drift,
)

NANNYML_AVAILABLE = find_spec("nannyml") is not None


def build_reference(rows: int = 1_000) -> pd.DataFrame:
    return pd.DataFrame(
        {"trip_distance_miles": [1.0 + ((index % 100) * 0.1) for index in range(rows)]}
    )


def build_analysis(
    rows: int = 500,
    *,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_distance_miles": [
                (1.0 + ((index % 100) * 0.1)) * multiplier for index in range(rows)
            ]
        }
    )


class NannyMLDriftIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        NANNYML_AVAILABLE,
        "NannyML monitoring environment is not active",
    )
    def test_distance_shift_triggers_all_chunks(self) -> None:
        report = calculate_univariate_drift(
            build_reference(),
            build_analysis(multiplier=1.609344),
            feature_name="trip_distance_miles",
            chunk_number=5,
            upper_threshold=0.1,
        )

        self.assertTrue(report.drift_detected)
        self.assertEqual(report.alert_count, 5)
        self.assertEqual(len(report.chunks), 5)
        self.assertAlmostEqual(
            report.max_drift_value,
            0.42,
            places=6,
        )
        self.assertTrue(all(chunk.alert for chunk in report.chunks))

    @unittest.skipUnless(
        NANNYML_AVAILABLE,
        "NannyML monitoring environment is not active",
    )
    def test_stable_distribution_does_not_trigger_alerts(
        self,
    ) -> None:
        report = calculate_univariate_drift(
            build_reference(),
            build_analysis(),
            feature_name="trip_distance_miles",
            chunk_number=5,
            upper_threshold=0.1,
        )

        self.assertFalse(report.drift_detected)
        self.assertEqual(report.alert_count, 0)
        self.assertEqual(len(report.chunks), 5)


class NannyMLDriftValidationTests(unittest.TestCase):
    def test_rejects_missing_feature(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "missing feature",
        ):
            calculate_univariate_drift(
                build_reference(),
                pd.DataFrame({"different_feature": [1.0]}),
                feature_name="trip_distance_miles",
            )

    def test_rejects_empty_reference_data(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "reference data cannot be empty",
        ):
            calculate_univariate_drift(
                pd.DataFrame(columns=["trip_distance_miles"]),
                build_analysis(),
                feature_name="trip_distance_miles",
            )

    def test_rejects_invalid_chunk_number(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "chunk_number must be greater than zero",
        ):
            calculate_univariate_drift(
                build_reference(),
                build_analysis(),
                feature_name="trip_distance_miles",
                chunk_number=0,
            )

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "positive finite number",
        ):
            calculate_univariate_drift(
                build_reference(),
                build_analysis(),
                feature_name="trip_distance_miles",
                upper_threshold=0,
            )


if __name__ == "__main__":
    unittest.main()
