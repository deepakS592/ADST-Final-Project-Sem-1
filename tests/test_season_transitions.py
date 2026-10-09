from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
import pandas as pd

from services import data_service, database
from services.database import connect, prediction_data_version
from startup_sync import _download_missing_race_rows


@pytest.fixture
def transition_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "season-transitions.sqlite")
    database.initialize_schema()
    data_service.clear_circuit_analysis_cache()
    data_service._calendar_race_start_times.cache_clear()
    yield
    data_service.clear_circuit_analysis_cache()


def _insert_calendar_event(season, round_number, race, event_date, *, observed=False):
    event_id = f"transition-{season}-{round_number}"
    with connect() as connection:
        connection.execute(
            "INSERT INTO events(event_id,season,round,race,circuit,country,location,event_date,season_status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (event_id, season, round_number, race, "Transition Circuit", "Testland", "Test City",
             event_date, "current/partial" if observed else "upcoming"),
        )
        if observed:
            connection.execute(
                "INSERT INTO laps(event_id,Season,Round,Race,Circuit,EventDate,Driver,Compound,LapNumber,TyreLife,LapTime) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, season, round_number, race, "Transition Circuit", event_date,
                 "TST", "SOFT", 1, 1, 90.0),
            )
    return event_id


def test_season_transition_sequence_uses_calendar_and_observations(transition_database):
    ongoing_season = 2040
    next_season = ongoing_season + 1
    _insert_calendar_event(ongoing_season, 1, "Observed opener", "2040-03-01", observed=True)
    _insert_calendar_event(ongoing_season, 2, "Season finale", "2040-11-10")
    first_id = _insert_calendar_event(next_season, 1, "Next season opener", "2041-03-07")
    second_id = _insert_calendar_event(next_season, 2, "Next season round two", "2041-03-21")

    # Current season stays primary while its official calendar has future races.
    active = data_service.season_catalog(date(2040, 6, 1))
    assert active["active_season"] == ongoing_season
    assert data_service.season_state(ongoing_season, date(2040, 6, 1))["state"] == "ACTIVE SEASON"

    # A selectable future calendar stays pre-season while the current season runs.
    preseason = data_service.selected_season_context(next_season, date(2040, 6, 1))
    assert preseason["state"] == "PRE-SEASON"
    assert preseason["label"] == f"{next_season} PRE-SEASON PREVIEW"
    assert f"The {ongoing_season} season is still in progress." in preseason["message"]
    assert f"No {next_season} race observations available yet." in preseason["message"]
    assert preseason["data_availability"]["calendar_data"] is True
    assert preseason["data_availability"]["current_season_observations"] == 0
    assert data_service.drivers_for_event(first_id) == []

    # After the last scheduled race date, the primary dashboard advances to the
    # first published event of the next season without creating any observations.
    after_finale = date(2040, 11, 11)
    assert data_service.season_state(ongoing_season, after_finale)["state"] == "SEASON COMPLETE"
    catalog = data_service.season_catalog(after_finale)
    assert catalog["active_season"] == next_season
    assert catalog["active_season_state"] == "PRE-SEASON"
    assert data_service.focus_event(after_finale)["event_id"] == first_id

    # Two days before the race is the event weekend; race day with unknown start
    # time is conservatively locked as in progress.
    weekend = date(2041, 3, 5)
    assert data_service.season_state(next_season, weekend)["state"] == "CURRENT RACE"
    race_now = datetime(2041, 3, 7, 15, 0, tzinfo=timezone.utc)
    assert data_service.season_state(next_season, date(2041, 3, 7), now=race_now)["state"] == "RACE IN PROGRESS"

    # Once the first event's date has passed, focus automatically moves to round two.
    after_first_race = date(2041, 3, 8)
    first_event = next(row for row in data_service.events_for_season(next_season) if row["event_id"] == first_id)
    assert data_service.event_prediction_state(first_event, datetime(2041, 3, 8, 9, 0, tzinfo=timezone.utc))["state"] == "COMPLETED"
    focus = data_service.focus_event(after_first_race)
    assert focus["event_id"] == second_id
    assert focus["season_context"]["state"] == "ACTIVE SEASON"


def test_cutoff_bundle_versions_change_only_after_new_observation_date(transition_database):
    with connect() as connection:
        connection.executemany(
            "INSERT INTO event_data_versions(event_date,cumulative_version) VALUES (?,?)",
            [("2040-03-01", "version-before-new-race"), ("2040-11-10", "version-after-new-race")],
        )
    assert prediction_data_version("2040-10-01T00:00:00Z") == "version-before-new-race"
    assert prediction_data_version("2040-12-01T00:00:00Z") == "version-after-new-race"


def test_historical_circuit_laps_are_not_mislabeled_as_calendar_only(transition_database):
    with connect() as connection:
        connection.execute(
            "INSERT INTO events(event_id,season,round,race,circuit,event_date,season_status) "
            "VALUES (?,?,?,?,?,?,?)",
            ("historic-event", 2039, 1, "Historic GP", "Historic Circuit", "2039-05-01", "historical"),
        )
        connection.execute(
            "INSERT INTO circuit_history_laps(event_id,season,round,race,circuit,event_date,driver,lap_number,lap_time,compound,source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("historic-event", 2039, 1, "Historic GP", "Historic Circuit", "2039-05-01", "TST", 1, 90.0, "UNKNOWN", "test fixture"),
        )
    event = data_service.events_for_season(2039)[0]
    state = data_service.season_state(2039, date(2040, 1, 1))

    assert event["data_kind"] == "HISTORICAL DATA"
    assert state["observed_races"] == 0
    assert state["historical_races"] == 1
    assert state["message"] is None


def test_startup_ingestion_loads_only_missing_completed_race_sessions(transition_database, tmp_path, monkeypatch):
    import startup_sync

    monkeypatch.setattr(startup_sync, "SEASON_DATA_DIR", tmp_path / "fastf1")
    calls = []

    class FakeSession:
        results = pd.DataFrame([{"DriverNumber": "1", "FullName": "Test Driver", "TeamName": "Test Team"}])
        laps = pd.DataFrame([{
            "Driver": "TST", "DriverNumber": "1", "LapTime": pd.Timedelta("90s"),
            "LapNumber": 1, "TyreLife": 1, "Compound": "SOFT", "Stint": 1,
        }])

        def load(self, **kwargs):
            assert kwargs == {"telemetry": False, "weather": False, "messages": False, "laps": True}

    schedule = pd.DataFrame([{
        "RoundNumber": 1, "EventName": "New Season Grand Prix", "OfficialEventName": "New Season GP",
        "Location": "Test Circuit", "Country": "Testland", "Session5": "Race",
        "Session5Date": "2041-03-07T12:00:00+00:00", "Session5DateUtc": "2041-03-07T12:00:00Z",
    }])
    fastf1 = SimpleNamespace(
        get_event_schedule=lambda season, include_testing=False: schedule,
        get_session=lambda season, round_number, session_type: (calls.append((season, round_number, session_type)) or FakeSession()),
    )
    event = {"event_id": "event-2041-1", "season": 2041, "round": 1, "race": "New Season Grand Prix",
             "event_date": "2041-03-07", "circuit": "Test Circuit"}

    laps, statuses = _download_missing_race_rows([event], date(2041, 3, 8), fastf1)

    assert calls == [(2041, 1, "R")]
    assert len(laps) == 1
    assert laps.iloc[0]["Race"] == event["race"]
    assert statuses == [{"event_id": event["event_id"], "status": "downloaded", "rows": 1}]
