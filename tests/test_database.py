from services.data_service import drivers_for_event, events_for_season, quality_report, seasons
from services.database import connect, table_count
from config import LAP_TIME_MODEL_PATH, PIT_STOP_MODEL_PATH
from train_models import load_model_artifact


def test_sqlite_schema_and_imported_historical_data():
    assert table_count("events") >= 160
    assert table_count("laps") >= 170_000
    assert table_count("pit_transitions") > 1_000
    assert table_count("tyre_performance") >= 3
    assert table_count("weather_observations") > 0
    assert seasons() == sorted(seasons(), reverse=True)


def test_event_and_driver_selectors_use_real_database_rows():
    events = []
    for year in seasons():
        events = [event for event in events_for_season(year) if event["driver_count"]]
        if events:
            break
    assert events
    event = events[0]
    drivers = drivers_for_event(event["event_id"])
    assert drivers
    with connect() as connection:
        actual = connection.execute("SELECT COUNT(DISTINCT Driver) FROM laps WHERE event_id=?", (event["event_id"],)).fetchone()[0]
    assert len(drivers) == actual


def test_upcoming_schedule_is_source_backed_and_driver_context_is_unconfirmed():
    from services.data_service import driver_options_for_event

    with connect() as connection:
        row = connection.execute("SELECT event_id FROM events WHERE season_status='upcoming' ORDER BY event_date LIMIT 1").fetchone()
    assert row is not None
    options = driver_options_for_event(row["event_id"])
    assert options["selection_type"] == "historical_candidates"
    assert options["confirmed"] is False
    assert options["drivers"]
    assert all(item["entry_status"].startswith("historical candidate") for item in options["drivers"])


def test_quality_checks_find_no_duplicate_event_driver_lap_keys():
    report = quality_report()
    assert report["duplicate_event_driver_lap_keys"] == 0
    assert report["tables"]["laps"] >= 170_000
    assert report["missing_weather_events"] > 0


def test_preserved_random_forest_artifacts_load():
    assert load_model_artifact(LAP_TIME_MODEL_PATH) is not None
    assert load_model_artifact(PIT_STOP_MODEL_PATH) is not None


def test_tyre_performance_schema_migrates_without_losing_rows(tmp_path, monkeypatch):
    import sqlite3

    import services.database as database

    legacy_path = tmp_path / "legacy.db"
    with sqlite3.connect(legacy_path) as connection:
        connection.execute(
            "CREATE TABLE tyre_performance ("
            "compound TEXT PRIMARY KEY, data_points INTEGER, compound_offset REAL, "
            "offset_source TEXT, degradation_rate REAL, degradation_source TEXT, "
            "stints_used INTEGER, r_squared REAL)"
        )
        connection.execute(
            "INSERT INTO tyre_performance(compound, data_points, degradation_rate) "
            "VALUES ('MEDIUM', 120, 0.04)"
        )

    monkeypatch.setattr(database, "DATABASE_PATH", legacy_path)
    database.initialize_schema()
    with database.connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(tyre_performance)")}
        row = connection.execute(
            "SELECT data_points, degradation_rate, degradation_std "
            "FROM tyre_performance WHERE compound='MEDIUM'"
        ).fetchone()
    assert "degradation_std" in columns
    assert tuple(row) == (120, 0.04, None)


def test_fresh_tyre_performance_schema_includes_degradation_std(tmp_path, monkeypatch):
    import services.database as database

    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "fresh.db")
    database.initialize_schema()
    with database.connect() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(tyre_performance)")}
    assert "degradation_std" in columns
