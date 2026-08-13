import ast
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from features import TaxiTrip
from features.canonical import transform_trip as canonical_transform
from features.faults import MILES_TO_KILOMETRES, SkewMode
from features.online import transform_trip as online_transform
from features.shared import transform_trip as shared_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trip() -> TaxiTrip:
    return TaxiTrip(
        pickup_datetime=datetime(2024, 1, 8, 13, 30, tzinfo=ZoneInfo("UTC")),
        pickup_location_id=132,
        dropoff_location_id=236,
        passenger_count=None,
        trip_distance_miles=4.5,
    )


def test_all_paths_preserve_the_nine_feature_schema(trip: TaxiTrip) -> None:
    canonical = canonical_transform(trip)
    online = online_transform(trip)
    shared = shared_transform(trip)

    assert len(type(canonical).model_fields) == 9
    assert online == canonical == shared
    assert canonical.passenger_count == 1
    assert canonical.pickup_hour == 8
    assert canonical.is_rush_hour == 1


def test_distance_fault_changes_only_serving_distance(trip: TaxiTrip) -> None:
    canonical = canonical_transform(trip)
    online = online_transform(trip, SkewMode.DISTANCE_UNIT)

    assert online.trip_distance_miles == pytest.approx(
        canonical.trip_distance_miles * MILES_TO_KILOMETRES
    )
    assert online.model_dump(exclude={"trip_distance_miles"}) == canonical.model_dump(
        exclude={"trip_distance_miles"}
    )


def test_timezone_fault_uses_utc_for_online_time_features(trip: TaxiTrip) -> None:
    online = online_transform(trip, SkewMode.TIMEZONE)

    assert online.pickup_hour == 13
    assert online.pickup_day_of_week == 0
    assert online.pickup_month == 1
    assert online.is_weekend == 0
    assert online.is_rush_hour == 0


@pytest.mark.parametrize(
    ("pickup_datetime", "passenger_count", "distance"),
    [
        (datetime(2024, 1, 8, 11, 59, tzinfo=ZoneInfo("UTC")), None, 1.1234567),
        (datetime(2024, 1, 8, 12, 0, tzinfo=ZoneInfo("UTC")), 2, 4.5),
        (datetime(2024, 1, 8, 15, 0, tzinfo=ZoneInfo("UTC")), 4, 8.25),
        (datetime(2024, 1, 13, 5, 30, tzinfo=ZoneInfo("UTC")), 1, 2.0),
        (datetime(2024, 7, 8, 12, 30, tzinfo=ZoneInfo("UTC")), 3, 9.0),
    ],
)
def test_shared_matches_canonical_across_feature_edge_cases(
    pickup_datetime: datetime,
    passenger_count: int | None,
    distance: float,
) -> None:
    edge_trip = TaxiTrip(
        pickup_datetime=pickup_datetime,
        pickup_location_id=132,
        dropoff_location_id=236,
        passenger_count=passenger_count,
        trip_distance_miles=distance,
    )

    assert shared_transform(edge_trip) == canonical_transform(edge_trip)


def test_online_module_does_not_import_canonical_or_shared() -> None:
    source = (PROJECT_ROOT / "src" / "features" / "online.py").read_text()
    module = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "features.canonical" not in imported_modules
    assert "features.shared" not in imported_modules


def test_only_three_skew_modes_exist() -> None:
    assert [mode.value for mode in SkewMode] == ["none", "distance_unit", "timezone"]
