from datetime import date, datetime, timezone

import numpy as np

from services import data_service
from services.data_service import classify_event_status, event_prediction_state
from strategy_engine import _simulation_record, generate_feasible_strategies, simulate_race_time


def test_event_status_tracks_the_race_weekend_and_completion():
    event = {"event_date": "2026-10-04", "season_status": "upcoming"}
    assert classify_event_status(event, date(2026, 10, 1)) == "UPCOMING"
    assert classify_event_status(event, date(2026, 10, 2)) == "CURRENT"
    assert classify_event_status(event, date(2026, 10, 3)) == "CURRENT"
    assert classify_event_status(event, date(2026, 10, 5)) == "COMPLETED"


def test_prediction_state_uses_race_start_and_configured_timezone(monkeypatch):
    monkeypatch.setenv("F1_APP_TIMEZONE", "Asia/Kolkata")
    event = {"event_date": "2027-03-14", "race_start_time_utc": "2027-03-14T09:00:00Z"}

    before_start = event_prediction_state(event, datetime(2027, 3, 14, 8, 59, tzinfo=timezone.utc))
    at_start = event_prediction_state(event, datetime(2027, 3, 14, 9, 0, tzinfo=timezone.utc))
    next_day = event_prediction_state(event, datetime(2027, 3, 15, 4, 0, tzinfo=timezone.utc))
    not_yet_race_weekend = event_prediction_state(
        {"event_date": "2027-03-20"}, datetime(2027, 3, 14, 8, 59, tzinfo=timezone.utc),
    )

    assert before_start["state"] == "PRE_RACE"
    assert before_start["pre_race_allowed"] is True
    assert before_start["timezone"] == "Asia/Kolkata"
    assert at_start["state"] == "IN_PROGRESS"
    assert at_start["pre_race_allowed"] is False
    assert next_day["state"] == "COMPLETED"
    assert not_yet_race_weekend["state"] == "UPCOMING"
    assert not_yet_race_weekend["pre_race_allowed"] is True


def test_prediction_state_uses_calendar_update_start_time(tmp_path, monkeypatch):
    calendar_path = tmp_path / "calendar.csv"
    calendar_path.write_text(
        "Season,Round,Race,RaceStartTimeUTC\n2027,1,Test Grand Prix,2027-03-14T09:00:00Z\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(data_service, "CALENDAR_UPDATES_PATH", calendar_path)
    data_service._calendar_race_start_times.cache_clear()
    event = {"season": 2027, "round": 1, "race": "Test Grand Prix", "event_date": "2027-03-14"}

    state = event_prediction_state(event, datetime(2027, 3, 14, 8, 59, tzinfo=timezone.utc))

    assert state["state"] == "PRE_RACE"
    assert state["race_start_time_utc"] == "2027-03-14T09:00:00Z"
    assert state["start_time_available"] is True
    data_service._calendar_race_start_times.cache_clear()


def test_prediction_state_conservatively_locks_race_day_without_start_time(monkeypatch):
    monkeypatch.setenv("F1_APP_TIMEZONE", "Asia/Kolkata")
    event = {"event_date": "2027-03-14", "season_status": "upcoming"}

    state = event_prediction_state(event, datetime(2027, 3, 14, 4, 0, tzinfo=timezone.utc))

    assert state["state"] == "IN_PROGRESS"
    assert state["pre_race_allowed"] is False
    assert state["start_time_available"] is False


def test_vectorized_strategy_score_matches_the_per_lap_simulator():
    distance = 44
    compounds = ("SOFT", "MEDIUM")
    surface = {
        compound: np.fromfunction(
            lambda age, lap: 95.0 + 2.0 * (compound == "MEDIUM") + .04 * age + .002 * lap,
            (64, distance + 1),
        )
        for compound in compounds
    }
    weather_adjustments = {compound: np.full_like(surface[compound], .003) for compound in compounds}
    lengths = (20, 24)
    scored = _simulation_record(
        race_name="Test Grand Prix", driver="VER", season=2026, circuit="Test Circuit", team="Test Team",
        race_distance=distance, tyres=compounds, stint_lengths=lengths, weather={"condition": "DRY"},
        surface=surface, weather_adjustments=weather_adjustments,
        degradation={"SOFT": .05, "MEDIUM": .04}, classifier=None, classifier_cache={},
    )
    reference = simulate_race_time(
        compounds, lengths, distance,
        lambda compound, age, lap: float(surface[compound][age, lap]),
    )
    assert np.isclose(scored["EstimatedRaceTime"], reference["estimated_race_time"])
    assert np.isclose(scored["PredictedLapTimeTotal"], reference["predicted_lap_time_total"])
    assert list(scored["PitLaps"]) == reference["pit_laps"]


def test_long_race_three_stop_search_is_bounded_and_still_scores_three_stops():
    distance = 71
    age_axis = np.arange(100)[:, None]
    surface = {
        compound: np.tile(90.0 + age_axis * .05, (1, distance + 1))
        for compound in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
    }
    weather_adjustments = {compound: np.zeros_like(values) for compound, values in surface.items()}
    plans = generate_feasible_strategies(
        race_name="Long Race", driver="VER", race_distance=distance, weather={"condition": "DRY"},
        surface=surface, weather_adjustments=weather_adjustments,
        degradation={"SOFT": .05, "MEDIUM": .04, "HARD": .03}, classifier=None,
    )
    assert len(plans) < 20_000
    assert plans["StrategyType"].eq("3-STOP").any()
    assert "adjacent-lap refinement" in plans.attrs["search_method"]
    assert plans["PhysicalConstraintsSatisfied"].all()
