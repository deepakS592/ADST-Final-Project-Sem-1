from __future__ import annotations

import pytest

import services.database as database
from services import data_service


@pytest.fixture
def race_database(tmp_path, monkeypatch):
    path = tmp_path / "race-history.sqlite"
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    database.initialize_schema()
    data_service.clear_circuit_analysis_cache()
    with database.connect() as connection:
        for year in range(2010, 2018):
            event_id = f"sepang-{year}"
            race = f"Sepang Race {year}"
            event_date = f"{year}-04-10"
            connection.execute(
                "INSERT INTO events(event_id,season,round,race,circuit,country,event_date,race_laps) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (event_id, year, 1, race, "Sepang International Circuit", "Malaysia", event_date, 4),
            )
            for lap, compound, stint, age in (
                (1, "SOFT", 1, 1), (2, "SOFT", 1, 2),
                (3, "MEDIUM", 2, 1), (4, "MEDIUM", 2, 2),
            ):
                connection.execute(
                    "INSERT INTO laps(event_id,Season,Round,Race,Circuit,EventDate,RaceLaps,Driver,"
                    "DriverName,DriverNumber,Team,Compound,LapNumber,TyreLife,Stint,LapTime) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (event_id, year, 1, race, "Sepang International Circuit", event_date, 4,
                     "VER", "Max Verstappen", "1", "Red Bull Racing", compound, lap, age, stint,
                     90.0 + age + year / 1000),
                )
            connection.execute(
                "INSERT INTO pit_transitions(season,round,race,circuit,driver,pit_lap,race_laps,previous_tyre,new_tyre) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (year, 1, race, "Sepang International Circuit", "VER", 2, 4, "SOFT", "MEDIUM"),
            )

        # Deliberately store a row on the cutoff date and one future row.
        for event_id, year, event_date in (
            ("cutoff-race", 2016, "2016-04-10"),
            ("future-race", 2028, "2028-04-10"),
        ):
            connection.execute(
                "INSERT INTO events(event_id,season,round,race,circuit,country,event_date,race_laps) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (event_id, year, 2, event_id, "Other Circuit", "Malaysia", event_date, 4),
            )
            connection.execute(
                "INSERT INTO laps(event_id,Season,Round,Race,Circuit,EventDate,RaceLaps,Driver,Compound,LapNumber,TyreLife,Stint,LapTime) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, year, 2, event_id, "Other Circuit", event_date, 4,
                 "VER", "SOFT", 1, 1, 1, 91.0),
            )

        # A future calendar event has metadata only; no race or driver observations.
        connection.execute(
            "INSERT INTO events(event_id,season,round,race,circuit,country,location,event_date,race_laps,season_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2027-sepang", 2027, 1, "Future Malaysian Grand Prix", "Sepang International Circuit",
             "Malaysia", "Kuala Lumpur", "2027-12-05", 56, "upcoming"),
        )
    data_service.clear_circuit_analysis_cache()
    yield path
    data_service.clear_circuit_analysis_cache()


def test_circuit_history_selects_six_latest_strictly_before_cutoff(race_database):
    result = data_service.circuit_analysis("Sepang International Circuit", before="2016-04-10")

    assert [race["season"] for race in result["races"]] == [2015, 2014, 2013, 2012, 2011, 2010]
    assert result["coverage"]["race_count"] == 6
    assert result["coverage"]["lap_count"] == 24
    assert result["coverage"]["stint_count"] == 12
    assert result["coverage"]["pit_transition_count"] == 6
    assert result["coverage"]["tyre_observation_count"] == 24
    assert result["coverage"]["latest_race_date"] == "2015-04-10"


def test_explicit_target_event_exclusion_and_fewer_than_six_are_reported(race_database):
    result = data_service.circuit_analysis(
        "Sepang", before="2018-01-01", exclude_event_id="sepang-2017",
    )
    assert [race["season"] for race in result["races"]] == [2016, 2015, 2014, 2013, 2012, 2011]
    assert "sepang-2017" not in result["event_ids_used"]

    with database.connect() as connection:
        for year in range(2020, 2023):
            event_id = f"short-{year}"
            connection.execute(
                "INSERT INTO events(event_id,season,round,race,circuit,event_date) VALUES (?,?,?,?,?,?)",
                (event_id, year, 1, f"Short Track {year}", "Short Track", f"{year}-03-01"),
            )
            connection.execute(
                "INSERT INTO laps(event_id,Season,Round,Race,Circuit,EventDate,Driver,Compound,LapNumber) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (event_id, year, 1, f"Short Track {year}", "Short Track", f"{year}-03-01", "VER", "SOFT", 1),
            )
    data_service.clear_circuit_analysis_cache()
    short = data_service.circuit_analysis("Short Track", before="2024-01-01")
    assert short["coverage"]["race_count"] == 3
    assert short["coverage"]["message"] == "Only 3 circuit races available"


def test_auxiliary_history_supplies_laps_pits_and_observed_stop_strategies_without_tyres(race_database):
    with database.connect() as connection:
        connection.execute("DELETE FROM pit_transitions WHERE circuit='Sepang International Circuit'")
        connection.execute("DELETE FROM laps WHERE Circuit='Sepang International Circuit'")
        connection.execute("DELETE FROM events WHERE circuit='Sepang International Circuit' AND season<2020")
        for year in range(2012, 2018):
            event_id = f"legacy-sepang-{year}"
            race = f"Malaysian Grand Prix {year}"
            event_date = f"{year}-03-25"
            connection.execute(
                "INSERT INTO events(event_id,season,round,race,circuit,country,event_date,race_laps) VALUES (?,?,?,?,?,?,?,?)",
                (event_id, year, 2, race, "Sepang International Circuit", "Malaysia", event_date, 56),
            )
            for driver, stint, lap in (("HAMILTON", 1, 1), ("VETTEL", 1, 1), ("VETTEL", 2, 2)):
                connection.execute(
                    "INSERT INTO circuit_history_laps(event_id,season,round,race,circuit,event_date,race_laps,driver,lap_number,lap_time,compound,stint,source) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (event_id, year, 2, race, "Sepang International Circuit", event_date, 56,
                     driver, lap, 100.0 + lap, "UNKNOWN", stint, "Jolpica F1 API"),
                )
            connection.execute(
                "INSERT INTO circuit_history_pit_stops(event_id,driver,pit_lap,stop_number,source) VALUES (?,?,?,?,?)",
                (event_id, "VETTEL", 20, 1, "Jolpica F1 API"),
            )

    data_service.clear_circuit_analysis_cache()
    result = data_service.circuit_analysis("Sepang", before="2026-10-04")

    assert [race["season"] for race in result["races"]] == [2017, 2016, 2015, 2014, 2013, 2012]
    assert result["coverage"]["lap_count"] == 18
    assert result["coverage"]["stint_count"] == 18
    assert result["coverage"]["pit_transition_count"] == 6
    assert result["coverage"]["tyre_observation_count"] == 0
    assert result["tyre_performance"] == []
    assert result["tyre_degradation"] == []
    assert result["strategy_distribution"]
    assert all(row["basis"] == "observed_pit_stop_count" for row in result["strategy_distribution"])
    assert result["strategy_distribution_source"].startswith("Observed pit-stop counts")


def test_imported_calendar_metadata_adds_future_season_without_entrants(race_database, tmp_path, monkeypatch):
    import calendar_import

    path = tmp_path / "calendar_updates.csv"
    monkeypatch.setattr(calendar_import, "CALENDAR_UPDATES_PATH", path)
    monkeypatch.setattr(data_service, "CALENDAR_UPDATES_PATH", path)
    monkeypatch.setattr(calendar_import, "_fastf1_rows", lambda year: [])

    class FakeClient:
        rows = [{
            "season": "2027", "round": "1", "raceName": "Australian Grand Prix",
            "date": "2027-03-14", "url": "https://example.test/race",
            "Circuit": {"circuitName": "Albert Park Grand Prix Circuit", "Location": {
                "locality": "Melbourne", "country": "Australia",
            }},
        }]

        def fetch_schedule(self, year):
            assert year == 2027
            return self.rows

    client = FakeClient()
    result = calendar_import.update_calendar([2027], client=client)
    assert result["upcoming_events_added_to_database"] == 1
    assert 2027 in data_service.seasons()
    event = data_service.events_for_season(2027)[0]
    assert event["event_status"] == "UPCOMING"
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM laps WHERE event_id=?", (event["event_id"],)).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM event_drivers WHERE event_id=?", (event["event_id"],)).fetchone()[0] == 0

    # A refreshed non-empty schedule replaces stale future metadata for that season.
    client.rows = [{**client.rows[0], "raceName": "Australian GP", "date": "2027-03-21"}]
    calendar_import.update_calendar([2027], client=client)
    refreshed = data_service.events_for_season(2027)
    assert [row["race"] for row in refreshed] == ["Australian GP"]
    assert refreshed[0]["event_date"] == "2027-03-21"

    # Empty sources preserve the last known calendar snapshot.
    client.rows = []
    empty_refresh = calendar_import.update_calendar([2027], client=client)
    assert empty_refresh["source_rows_fetched"] == 0
    assert [row["race"] for row in data_service.events_for_season(2027)] == ["Australian GP"]


def test_future_season_without_event_laps_uses_available_circuit_history_and_local_map(race_database):
    from app import app

    client = app.test_client()
    events = client.get("/api/events?season=2027&status=available")
    assert events.status_code == 200
    assert [event["event_id"] for event in events.json["events"]] == ["2027-sepang"]

    context = client.get("/api/event-context/2027-sepang")
    assert context.status_code == 200
    assert context.json["race_status"] == "UPCOMING"
    assert context.json["circuit_analysis"]["coverage"]["race_count"] == 6
    assert context.json["track"]["url"] == "/static/tracks/sepang.svg"
    assert context.json["analysis_availability"]["track_history"] is True
    assert context.json["driver_selection"]["confirmed"] is False
    assert context.json["drivers"]
    assert context.json["drivers"][0]["driver_number"] == "1"
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM laps WHERE event_id='2027-sepang'").fetchone()[0] == 0
    assert client.get(context.json["track"]["url"]).status_code == 200


def test_track_analysis_is_present_in_main_page_before_driver_selection(race_database):
    from app import app

    html = app.test_client().get("/").get_data(as_text=True)
    assert 'id="track-intelligence"' in html
    assert 'id="circuit-history-years"' in html
    assert 'id="driver-select"' in html
    assert '<section class="track-intelligence" id="track-intelligence"' in html


def test_unknown_event_season_filter_is_graceful(race_database):
    from app import app

    response = app.test_client().get("/api/events?season=2099&status=available")
    assert response.status_code == 200
    assert response.json["events"] == []
