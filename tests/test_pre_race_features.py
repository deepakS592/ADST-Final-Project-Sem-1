from datetime import date

import numpy as np
import pandas as pd

import data_loader
import event_tyre_data
from data_loader import build_event_calendar, filter_laps_before_cutoff
from event_tyre_data import allocation_for_event, load_event_nomination
from backtesting import _actual_driver_strategy
from dynamic_dashboard_app import _driver_label
from strategy_engine import run_analysis, strategy_is_feasible, validate_stint_lengths
from weather_api import get_current_and_forecast_weather, weather_scenario_probabilities


def test_temporal_cutoff_excludes_race_day_and_later_rows():
    laps = pd.DataFrame({
        "EventDate": ["2024-03-01", "2024-03-02", "2024-03-03"],
        "Season": [2024, 2024, 2024],
        "Race": ["Earlier", "Target", "Later"],
        "LapNumber": [1, 1, 1],
    })
    filtered = filter_laps_before_cutoff(laps, "2024-03-02T00:00:00Z")
    assert filtered["Race"].tolist() == ["Earlier"]


def test_calendar_uses_official_override_and_separate_safe_cutoff(tmp_path, monkeypatch):
    override_path = tmp_path / "event_metadata_overrides.csv"
    pd.DataFrame([{
        "Season": 2026, "Round": 16, "Race": "Bahrain Grand Prix",
        "Circuit": "Sepang International Circuit", "Country": "Malaysia", "Location": "Kuala Lumpur",
        "EventDate": "2026-10-04", "RaceLaps": 56, "RaceLapsSource": "official F1 timetable",
        "RaceStartTimeUTC": "2026-10-04T07:00:00Z", "RaceStartTimeSource": "official timetable",
    }]).to_csv(override_path, index=False)
    monkeypatch.setattr(data_loader, "EVENT_METADATA_OVERRIDE_PATH", override_path)
    laps = pd.DataFrame({
        "Season": [2025], "Round": [22], "Race": ["Bahrain Grand Prix"], "OfficialEventName": ["Bahrain GP"],
        "Circuit": ["Sakhir"], "Country": ["Bahrain"], "Location": ["Sakhir"], "EventDate": ["2025-04-13"],
        "RaceLaps": [57], "LapNumber": [1], "Driver": ["VER"], "Team": ["Red Bull"],
    })
    statuses = pd.DataFrame([{
        "Season": 2026, "Round": 16, "Race": "Bahrain Grand Prix", "OfficialEventName": "Bahrain in Malaysia",
        "EventDate": "2026-10-04", "Status": "future/unrun", "Rows": 0,
    }])
    calendar = build_event_calendar(laps, statuses, today=date(2026, 9, 30))
    event = calendar.iloc[0]
    assert event["Circuit"] == "Sepang International Circuit"
    assert int(event["RaceLaps"]) == 56
    assert event["CutoffTimeUTC"] == "2026-10-04T07:00:00Z"
    assert event["DataCutoffTimeUTC"] == "2026-10-04T00:00:00Z"


def test_dry_compound_diversity_and_four_stint_feasibility():
    assert not strategy_is_feasible(["INTERMEDIATE", "SOFT"], "DRY")
    assert not strategy_is_feasible(["HARD", "HARD", "HARD"], "DRY")
    valid, reasons = validate_stint_lengths(
        ["SOFT", "MEDIUM", "HARD", "SOFT"], [15, 15, 15, 19], 64, "DRY"
    )
    assert valid, reasons


def test_official_nomination_requires_source_and_verified_status(tmp_path, monkeypatch):
    nomination_file = tmp_path / "nominations.csv"
    allocations_file = tmp_path / "allocations.csv"
    monkeypatch.setattr(event_tyre_data, "TYRE_ALLOCATIONS_PATH", allocations_file)
    pd.DataFrame(columns=["Season", "Round", "Race", "HardSets", "MediumSets", "SoftSets", "Source", "SourceURL", "VerifiedAt", "Status"]).to_csv(allocations_file, index=False)
    pd.DataFrame([{
        "Season": 2026, "Round": 16, "Race": "Bahrain Grand Prix", "HardCompound": "C2",
        "MediumCompound": "C3", "SoftCompound": "C4", "HardSets": "", "MediumSets": "", "SoftSets": "",
        "SprintWeekend": "false", "Source": "Pirelli", "SourceURL": "https://example.test/source",
        "VerifiedAt": "2026-08-27", "Status": "official",
    }]).to_csv(nomination_file, index=False)
    record = load_event_nomination(2026, "Bahrain Grand Prix", 16, path=nomination_file)
    assert record["available"] is True
    assert record["HardCompound"] == "C2"
    assert record["official_allocation_complete"] is False
    inventory, source = allocation_for_event(record)
    assert inventory["HARD"] == 2
    assert "model assumption" in source


def test_official_consecutive_range_maps_compound_roles_per_event(tmp_path, monkeypatch):
    nominations = tmp_path / "nominations.csv"
    allocations = tmp_path / "allocations.csv"
    monkeypatch.setattr(event_tyre_data, "TYRE_ALLOCATIONS_PATH", allocations)
    pd.DataFrame(columns=["Season", "Round", "Race", "HardSets", "MediumSets", "SoftSets", "Source", "SourceURL", "VerifiedAt", "Status"]).to_csv(allocations, index=False)
    pd.DataFrame([{
        "Season": 2026, "Round": 16, "Race": "Bahrain Grand Prix", "HardCompound": "",
        "MediumCompound": "", "SoftCompound": "", "NominatedRange": "C2-C4",
        "Source": "Pirelli Press Release", "SourceURL": "https://example.test/source",
        "VerifiedAt": "2026-08-27", "Status": "official range",
    }]).to_csv(nominations, index=False)
    record = load_event_nomination(2026, "Bahrain Grand Prix", 16, path=nominations)
    assert record["available"] is True
    assert record["role_mapping_available"] is True
    assert (record["HardCompound"], record["MediumCompound"], record["SoftCompound"]) == ("C2", "C3", "C4")
    assert record["role_mapping_source"].startswith("derived")


def test_forecast_variables_create_normalized_scenario_weights(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "FORECAST_CACHE_DIR", tmp_path / "forecast_cache")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "timezone": "Asia/Kuala_Lumpur",
                "current": {
                    "time": "2026-09-30T12:00", "temperature_2m": 30, "apparent_temperature": 34,
                    "precipitation": 0, "rain": 0, "weather_code": 2, "cloud_cover": 50,
                    "wind_speed_10m": 12, "wind_direction_10m": 120, "wind_gusts_10m": 18,
                    "relative_humidity_2m": 70,
                },
                "hourly": {
                    "time": ["2026-10-04T12:00", "2026-10-04T13:00", "2026-10-04T14:00", "2026-10-04T15:00"],
                    "temperature_2m": [30, 31, 31, 30], "apparent_temperature": [34, 35, 35, 34],
                    "precipitation_probability": [10, 60, 80, 30], "precipitation": [0, 0.6, 1.4, 0.1],
                    "rain": [0, 0.6, 1.4, 0.1], "weather_code": [2, 61, 63, 3],
                    "wind_speed_10m": [10, 11, 13, 12], "wind_direction_10m": [120, 125, 130, 120],
                    "wind_gusts_10m": [16, 18, 20, 18], "cloud_cover": [30, 90, 100, 60],
                    "relative_humidity_2m": [65, 85, 90, 75],
                },
            }

    response = get_current_and_forecast_weather(
        "Bahrain Grand Prix",
        {"date": "2026-10-04", "latitude": 3.112, "longitude": 101.7, "circuit": "Sepang", "country": "Malaysia"},
        request_get=lambda *args, **kwargs: Response(),
        forecast_days=7,
    )
    weights = weather_scenario_probabilities(response)["weights"]
    assert response["current"]["available"] is True
    assert response["forecast"]["available"] is True
    assert set(weights) == {"DRY", "DAMP", "WET"}
    assert np.isclose(sum(weights.values()), 1.0)


def test_pre_race_engine_requires_explicit_cutoff_and_asof_data():
    try:
        run_analysis("Future Grand Prix", "VER", analysis_mode="pre-race")
    except ValueError as error:
        assert "cutoff time" in str(error)
    else:
        raise AssertionError("Pre-race mode must reject calls without cutoff and as-of inputs.")


def test_dashboard_module_imports():
    import dynamic_dashboard_app  # noqa: F401


def test_backtest_actual_driver_strategy_counts_partial_laps():
    laps = pd.DataFrame({
        "Driver": ["ALB"] * 3,
        "LapNumber": [1, 2, 3],
        "LapTime": [100.0, 101.0, 102.0],
        "Stint": [1, 1, 2],
        "Compound": ["SOFT", "SOFT", "HARD"],
    })
    sequence, stops, actual_time, lap_count = _actual_driver_strategy(laps, "ALB")
    assert sequence == ["SOFT", "HARD"]
    assert stops == 1
    assert actual_time == 303.0
    assert lap_count == 3


def test_driver_label_uses_most_recent_historical_team():
    rows = pd.DataFrame({
        "Driver": ["ALB", "ALB"],
        "DriverName": ["Alexander Albon", "Alexander Albon"],
        "DriverNumber": ["23", "23"],
        "Team": ["Toro Rosso", "Williams"],
        "EventDate": ["2019-06-30", "2026-09-26"],
    })
    assert "Williams" in _driver_label("ALB", rows)
    pre_race_label = _driver_label("ALB", rows, include_team=False)
    assert "Alexander Albon" in pre_race_label
    assert "Williams" not in pre_race_label
