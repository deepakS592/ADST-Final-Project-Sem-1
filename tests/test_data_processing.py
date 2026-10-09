import json

import pandas as pd

from config import RACES
from data_preparation import clean_lap_data, load_lap_data
from strategy_engine import run_analysis
from weather_api import fallback_weather, get_historical_weather


def test_cleaning_normalizes_and_drops_invalid_and_duplicate_rows():
    frame = pd.DataFrame(
        [
            {"Race": " Bahrain Grand Prix ", "Driver": "ver", "Compound": "soft", "LapNumber": "1", "TyreLife": "1", "LapTime": "97.1"},
            {"Race": "Bahrain Grand Prix", "Driver": "VER", "Compound": "SOFT", "LapNumber": 1, "TyreLife": 1, "LapTime": 97.1},
            {"Race": "Bahrain Grand Prix", "Driver": "VER", "Compound": "SOFT", "LapNumber": 2, "TyreLife": 2, "LapTime": 20.0},
            {"Race": "Bahrain Grand Prix", "Driver": "VER", "Compound": "SOFT", "LapNumber": 3, "TyreLife": 0, "LapTime": 98.0},
            {"Race": "Bahrain Grand Prix", "Driver": "VER", "Compound": "ULTRA", "LapNumber": 4, "TyreLife": 4, "LapTime": 98.0},
        ]
    )
    cleaned = clean_lap_data(frame)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["Driver"] == "VER"
    assert cleaned.iloc[0]["Compound"] == "SOFT"


def test_supplied_historical_dataset_loads_without_synthetic_rows():
    laps = load_lap_data()
    assert len(laps) > 1000
    event_names = set(laps["Race"])
    expected_events = set(RACES) - {"Emilia-Romagna Grand Prix"}
    assert event_names.issuperset(expected_events)
    assert "Emilia Romagna Grand Prix" in event_names
    assert laps["Driver"].str.fullmatch(r"[A-Z0-9]{3}").all()


def test_open_meteo_failure_uses_labeled_fallback_without_fake_weather(monkeypatch):
    race, info = next(iter(RACES.items()))

    def raise_network_error(*args, **kwargs):
        raise OSError("offline test")

    weather = get_historical_weather(race, info, request_get=raise_network_error)
    assert weather["source"] == "fallback assumption"
    assert weather["condition"] == "DRY"
    assert weather["temperature"] is None
    assert weather["rain"] is None
    assert "offline test" in weather["reason"]


def test_open_meteo_uses_event_coordinates_when_present(monkeypatch, tmp_path):
    import config

    monkeypatch.setattr(config, "WEATHER_CACHE_DIR", tmp_path / "weather")
    info = {
        "season": 2024,
        "circuit": "Track A",
        "location": "City A",
        "country": "Country A",
        "date": "2024-03-01",
        "latitude": 10.0,
        "longitude": 20.0,
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "hourly": {
                    "time": ["2024-03-01T14:00"],
                    "temperature_2m": [25.0],
                    "precipitation": [0.0],
                    "rain": [0.0],
                    "weather_code": [0],
                    "wind_speed_10m": [5.0],
                    "wind_direction_10m": [180.0],
                }
            }

    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    weather = get_historical_weather("Track A Grand Prix", info, request_get=fake_get)
    assert weather["source"] == "Open-Meteo historical archive"
    assert weather["temperature"] == 25.0
    assert calls[0][1]["params"]["latitude"] == 10.0
    assert not any("geocoding" in url for url, _ in calls)


def test_backend_and_csv_outputs_are_generated(tmp_path):
    laps = load_lap_data()
    race = "Bahrain Grand Prix"
    driver = sorted(laps.loc[laps["Race"].eq(race), "Driver"].unique())[0]
    analysis = run_analysis(
        race,
        driver,
        season=2024,
        weather_override=fallback_weather(race, RACES[race], "offline integration test"),
        output_dir=tmp_path,
    )
    assert (tmp_path / "strategy_simulation_results.csv").exists()
    assert (tmp_path / "weather_analysis.csv").exists()
    backend_path = tmp_path / "backend_result.json"
    assert backend_path.exists()
    result = json.loads(backend_path.read_text(encoding="utf-8"))
    assert result["physical_constraints_satisfied"] is True
    assert result["recommended_strategy"]["EstimatedRaceTime"] > 0
    strategies = analysis["strategies"]
    assert strategies["PhysicalConstraintsSatisfied"].all()
    assert len(strategies) > 0


def test_missing_race_and_driver_errors_are_clear(monkeypatch):
    laps = load_lap_data()
    from strategy_engine import run_analysis as run

    monkeypatch.setattr("strategy_engine._ensure_training_assets", lambda: (laps, None, None, {}))
    try:
        run("Unavailable Grand Prix", "VER", season=2024, fetch_weather=False)
    except ValueError as error:
        assert "No historical laps" in str(error)
    else:
        raise AssertionError("Missing race should fail with a clear message.")

    try:
        run("Bahrain Grand Prix", "ZZZ", season=2024, fetch_weather=False)
    except ValueError as error:
        assert "No usable laps" in str(error)
    else:
        raise AssertionError("Missing driver should fail with a clear message.")


def test_missing_lap_data_returns_a_clear_error(monkeypatch):
    import pytest

    empty = pd.DataFrame(columns=["Season", "Race", "Driver"])
    monkeypatch.setattr("strategy_engine._ensure_training_assets", lambda: (empty, None, None, {}))
    with pytest.raises(ValueError, match="No historical laps"):
        run_analysis("Bahrain Grand Prix", "VER", season=2024, fetch_weather=False)


def test_models_are_created_when_supplied_data_is_sufficient():
    from data_preparation import build_pit_training_data
    from train_models import train_lap_time_model, train_pit_stop_model

    laps = load_lap_data()
    sample_season = int(laps["Season"].max())
    sample = laps[laps["Season"].eq(sample_season)].groupby("Race", group_keys=False).head(3000)
    lap_artifact, lap_metrics = train_lap_time_model(sample)
    pit_artifact, pit_metrics = train_pit_stop_model(build_pit_training_data(sample))
    assert lap_artifact is not None
    assert lap_metrics["MAE"] is not None
    assert lap_metrics["RMSE"] is not None
    assert lap_metrics["R2"] is not None
    assert pit_artifact is not None
    assert pit_metrics["training_rows"] >= 20
