from datetime import date

import joblib
import pandas as pd
import pytest

import app as app_module
import pre_race_analysis
import startup_sync
from services import data_service, database
from services.database import connect, prediction_data_version


@pytest.fixture
def bundle_sync_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "bundle-lifecycle.sqlite")
    monkeypatch.setattr(pre_race_analysis, "PRE_RACE_MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(startup_sync, "_BUNDLE_STATES", {})
    monkeypatch.setattr(startup_sync, "_SYNC_STATE", {
        "sync_status": "idle", "last_successful_sync": None, "data_version": None,
        "new_events": [], "updated_events": [], "last_error": None,
    })
    database.initialize_schema()
    pre_race_analysis.clear_pre_race_caches()
    data_service.clear_circuit_analysis_cache()
    yield tmp_path
    pre_race_analysis.clear_pre_race_caches()
    data_service.clear_circuit_analysis_cache()


def _event(event_id, round_number, race, event_date, status="upcoming"):
    return {
        "event_id": event_id, "season": 2026, "round": round_number, "race": race,
        "circuit": race.removesuffix(" Grand Prix"), "country": "Testland", "location": "Test City",
        "event_date": event_date, "race_laps": 58, "season_status": status,
    }


def _insert_event(event):
    with connect() as connection:
        connection.execute(
            "INSERT INTO events(event_id,season,round,race,circuit,country,location,event_date,race_laps,season_status) "
            "VALUES (:event_id,:season,:round,:race,:circuit,:country,:location,:event_date,:race_laps,:season_status)",
            event,
        )


def _write_bundle(event_id, cutoff, version):
    path = pre_race_analysis._prediction_bundle_path(event_id, cutoff, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "bundle_version": pre_race_analysis.ASOF_BUNDLE_VERSION,
        "event_id": event_id, "cutoff_utc": cutoff, "data_version": version,
        "lap_artifact": {"model": "lap"}, "pit_artifact": {"model": "pit"},
    }, path)
    return path


def test_bundle_status_requires_current_metadata_and_reports_stale(bundle_sync_environment):
    cutoff = "2026-10-11T00:00:00Z"
    missing = pre_race_analysis.prediction_bundle_status("2026-17-d72842ae7c", cutoff, "new-data-version")
    assert missing["status"] == "MISSING"
    assert missing["available"] is False

    _write_bundle("2026-17-d72842ae7c", cutoff, "old-data-version")

    stale = pre_race_analysis.prediction_bundle_status("2026-17-d72842ae7c", cutoff, "new-data-version")
    assert stale["available"] is False
    assert stale["status"] == "STALE"
    assert "data version" in stale["reason"].lower()

    _write_bundle("2026-17-d72842ae7c", cutoff, "new-data-version")
    pre_race_analysis.clear_pre_race_caches()
    ready = pre_race_analysis.prediction_bundle_status("2026-17-d72842ae7c", cutoff, "new-data-version")
    assert ready["available"] is True
    assert ready["status"] == "READY"


def test_bundle_status_exposes_building_and_failure_reason(bundle_sync_environment):
    event_id, cutoff, version = "event-in-build", "2026-10-11T00:00:00Z", "current-data-version"
    startup_sync._record_bundle_state(event_id, cutoff, version, "BUILDING", stale=True)
    assert pre_race_analysis.prediction_bundle_status(event_id, cutoff, version)["status"] == "BUILDING"

    startup_sync._record_bundle_state(event_id, cutoff, version, "FAILED", reason="disk full", stale=True)
    failed = pre_race_analysis.prediction_bundle_status(event_id, cutoff, version)
    assert failed["status"] == "FAILED"
    assert failed["stale"] is True
    assert failed["reason"] == "disk full"


def test_bundle_writer_atomically_installs_verified_artifact(bundle_sync_environment, monkeypatch):
    event = _event("atomic-event", 17, "Singapore Grand Prix", "2026-10-11")
    cutoff = "2026-10-11T00:00:00Z"
    laps = pd.DataFrame([{
        "event_id": "observed-prior-race", "Season": 2026, "Race": "Prior Grand Prix",
        "EventDate": "2026-10-04", "Driver": "VER", "LapTime": 90.0,
    }])
    monkeypatch.setattr(pre_race_analysis, "_asof_models", lambda *args, **kwargs: ({"model": "lap"}, {"model": "pit"}, {}))
    monkeypatch.setattr(pre_race_analysis, "_performance_profile", lambda *args: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(pre_race_analysis, "_circuit_adjusted_performance", lambda laps, summary, *args, **kwargs: summary)
    monkeypatch.setattr(pre_race_analysis, "_circuit_adjusted_curve", lambda laps, curve, *args, **kwargs: curve)
    monkeypatch.setattr(pre_race_analysis, "_strategy_context", lambda *args, **kwargs: {"circuit_history": {"races": []}})

    artifact = pre_race_analysis.build_prediction_bundle(event, cutoff, data_version="atomic-version", laps=laps)

    assert artifact.exists()
    assert pre_race_analysis.prediction_bundle_status(event["event_id"], cutoff, "atomic-version")["status"] == "READY"
    assert not list(artifact.parent.glob(f".{artifact.name}.*.tmp"))


def test_startup_without_new_race_keeps_valid_bundle_and_reports_sync_status(bundle_sync_environment, monkeypatch):
    from pre_race_analysis import prediction_bundle_status

    event = _event("upcoming-1", 17, "Singapore Grand Prix", "2026-10-11")
    historical = _event("completed-1", 16, "Bahrain Grand Prix", "2026-10-04", "historical")
    _insert_event(event)
    _insert_event(historical)
    cutoff = "2026-10-11T00:00:00Z"
    artifact = _write_bundle(event["event_id"], cutoff, prediction_data_version(cutoff))
    historical_cutoff = "2026-10-04T00:00:00Z"
    old_artifact = _write_bundle(historical["event_id"], historical_cutoff, prediction_data_version(historical_cutoff))
    original_bytes = old_artifact.read_bytes()

    monkeypatch.setattr(startup_sync, "_missing_completed_calendar_events", lambda _today: [])
    monkeypatch.setattr(startup_sync, "_download_missing_race_rows", lambda *args: (pd.DataFrame(), []))
    monkeypatch.setattr(startup_sync, "_merge_into_existing_lap_source", lambda _laps: pytest.fail("no race data should be merged"))
    monkeypatch.setattr(pre_race_analysis, "build_prediction_bundle", lambda *args, **kwargs: pytest.fail("valid bundle should not be rebuilt"))

    result = startup_sync.run_startup_sync(today=date(2026, 10, 6), calendar_updater=lambda years: {"seasons": years})

    assert result["races_imported"] == 0
    assert result["bundles_rebuilt"] == 0
    assert artifact.exists()
    assert old_artifact.read_bytes() == original_bytes
    assert prediction_bundle_status(event["event_id"], cutoff)["status"] == "READY"
    snapshot = startup_sync.sync_status_snapshot()
    assert snapshot["sync_status"] == "complete"
    assert snapshot["last_successful_sync"]
    assert {"data_version", "new_events", "updated_events", "bundles_building", "bundles_ready", "bundles_stale", "bundles_failed"} <= snapshot.keys()


def test_new_race_version_rebuilds_only_later_bundles_and_failure_isolated(
    bundle_sync_environment, monkeypatch
):
    import data_preparation
    import services.prediction_service as prediction_service

    race = _event("2026-16-bahrain", 16, "Bahrain Grand Prix", "2026-10-04", "current/partial")
    completed = _event("2026-15-completed", 15, "Monza Grand Prix", "2026-09-13", "historical")
    futures = [
        _event("2026-17-d72842ae7c", 17, "Singapore Grand Prix", "2026-10-11"),
        _event("2026-18-austin", 18, "Austin Grand Prix", "2026-10-25"),
        _event("2026-19-mexico", 19, "Mexico City Grand Prix", "2026-11-01"),
        _event("2026-20-sao-paulo", 20, "São Paulo Grand Prix", "2026-11-08"),
    ]
    for item in [race, completed, *futures]:
        _insert_event(item)
    with connect() as connection:
        connection.execute("INSERT INTO event_data_versions(event_date,cumulative_version) VALUES (?,?)", ("2026-09-01", "before-race-version"))
    unaffected = _write_bundle(completed["event_id"], "2026-09-13T00:00:00Z", "before-race-version")
    unaffected_bytes = unaffected.read_bytes()
    old_future_artifacts = {
        item["event_id"]: _write_bundle(item["event_id"], f"{item['event_date']}T00:00:00Z", "before-race-version")
        for item in futures
    }
    assert pre_race_analysis.prediction_bundle_status(futures[0]["event_id"], "2026-10-11T00:00:00Z", "before-race-version")["status"] == "READY"

    downloaded = pd.DataFrame([{
        "Season": 2026, "Round": 16, "Race": "Bahrain Grand Prix", "EventDate": "2026-10-04",
        "Driver": "VER", "Compound": "SOFT", "LapNumber": 1, "LapTime": 90.0,
    }])
    monkeypatch.setattr(startup_sync, "_missing_completed_calendar_events", lambda _today: [race])
    monkeypatch.setattr(startup_sync, "_download_missing_race_rows", lambda *args: (downloaded, [{"event_id": race["event_id"], "status": "downloaded", "rows": 1}]))
    monkeypatch.setattr(startup_sync, "_merge_into_existing_lap_source", lambda _laps: 1)
    monkeypatch.setattr(data_preparation, "prepare_data", lambda **kwargs: {})

    def import_new_race(*, force=False, append_only=False):
        assert force and append_only
        with connect() as connection:
            connection.execute("INSERT INTO event_data_versions(event_date,cumulative_version) VALUES (?,?)", ("2026-10-04", "after-race-version"))
            connection.execute(
                "INSERT INTO laps(event_id,Season,Round,Race,EventDate,Driver,Compound,LapNumber,LapTime) VALUES (?,?,?,?,?,?,?,?,?)",
                (race["event_id"], 2026, 16, race["race"], "2026-10-04", "VER", "SOFT", 1, 90.0),
            )
        return {"imported": True, "laps": 1}

    monkeypatch.setattr(data_service, "import_project_data", import_new_race)
    monkeypatch.setattr(prediction_service, "clear_lap_cache", lambda: None)
    seen_stale = []
    status_impl = pre_race_analysis.prediction_bundle_status

    def track_stale(event_id, cutoff, data_version=None):
        status = status_impl(event_id, cutoff, data_version)
        if event_id == futures[0]["event_id"] and status.get("status") == "STALE":
            seen_stale.append((event_id, cutoff, data_version))
        return status

    monkeypatch.setattr(pre_race_analysis, "prediction_bundle_status", track_stale)
    built = []

    def fake_build(event, cutoff, *, data_version=None):
        event_id = str(event["event_id"])
        built.append(event_id)
        if event_id == futures[2]["event_id"]:
            raise RuntimeError("simulated Mexico model storage failure")
        return _write_bundle(event_id, cutoff, data_version)

    monkeypatch.setattr(pre_race_analysis, "build_prediction_bundle", fake_build)
    result = startup_sync.run_startup_sync(today=date(2026, 10, 6), calendar_updater=lambda years: {"seasons": years})

    assert result["races_imported"] == 1
    assert result["sqlite_import"]["imported"] is True
    assert prediction_data_version("2026-10-03T00:00:00Z") == "before-race-version"
    assert prediction_data_version("2026-10-11T00:00:00Z") == "after-race-version"
    assert seen_stale and seen_stale[0][0] == futures[0]["event_id"]
    assert built == [item["event_id"] for item in futures]
    assert unaffected.read_bytes() == unaffected_bytes
    assert pre_race_analysis.prediction_bundle_status(futures[0]["event_id"], "2026-10-11T00:00:00Z", "after-race-version")["status"] == "READY"
    assert pre_race_analysis.prediction_bundle_status(futures[1]["event_id"], "2026-10-25T00:00:00Z", "after-race-version")["status"] == "READY"
    assert pre_race_analysis.prediction_bundle_status(futures[3]["event_id"], "2026-11-08T00:00:00Z", "after-race-version")["status"] == "READY"
    failed = pre_race_analysis.prediction_bundle_status(futures[2]["event_id"], "2026-11-01T00:00:00Z", "after-race-version")
    assert failed["status"] == "FAILED"
    assert "Mexico model storage failure" in failed["reason"]
    assert all(path.exists() for path in old_future_artifacts.values())

    monkeypatch.setattr(app_module, "app_today", lambda: date(2026, 10, 6))
    client = app_module.create_app(initialize_data=False).test_client()
    context = client.get(f"/api/event-context/{futures[0]['event_id']}")
    assert context.status_code == 200
    assert context.json["model_bundle"]["status"] == "READY"
    assert context.json["model_bundle"]["available"] is True
    status = client.get("/api/sync-status")
    assert status.status_code == 200
    assert status.json["sync_status"] == "partial"
    assert status.json["new_events"] == [race["event_id"]]
    assert status.json["bundles_failed"] == 1
    assert status.json["bundles_ready"] >= 3

