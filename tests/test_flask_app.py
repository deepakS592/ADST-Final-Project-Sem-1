import app as app_module


def test_frontend_and_health_route_are_served_by_flask():
    client = app_module.app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"PREDICT PIT STRATEGY" in response.data
    assert client.get("/api/health").json["database"] == "SQLite"


def test_next_event_and_event_context_endpoints_expose_race_weekend_context():
    client = app_module.app.test_client()
    focus = client.get("/api/next-event")
    assert focus.status_code == 200
    event = focus.json["event"]
    assert event["event_status"] in {"CURRENT", "UPCOMING"}
    assert "forecast_available" in focus.json
    context = client.get(f"/api/event-context/{event['event_id']}")
    assert context.status_code == 200
    assert context.json["race_status"] in {"CURRENT", "UPCOMING"}
    assert context.json["prediction_state"]["state"] in {"UPCOMING", "PRE_RACE", "IN_PROGRESS", "COMPLETED"}
    assert context.json["model_bundle"]["status"] in {"READY", "STALE", "MISSING", "BUILDING", "FAILED", "NOT_REQUIRED"}
    assert context.json["race_distance"] == event["race_laps"]
    assert isinstance(context.json["forecast"]["available"], bool)
    bundle_status = client.get(f"/api/bundle-status/{event['event_id']}")
    assert bundle_status.status_code == 200
    assert bundle_status.json["status"] == context.json["model_bundle"]["status"]
    state = client.get(f"/api/event-state/{event['event_id']}")
    assert state.status_code == 200
    assert state.json["state"] == context.json["prediction_state"]["state"]


def test_sync_status_api_exposes_incremental_sync_and_bundle_counts():
    payload = app_module.app.test_client().get("/api/sync-status")
    assert payload.status_code == 200
    assert {
        "sync_status", "last_successful_sync", "data_version", "new_events", "updated_events",
        "bundles_building", "bundles_ready", "bundles_stale", "bundles_failed",
    } <= payload.json.keys()


def test_race_and_driver_api_uses_sqlite_records():
    client = app_module.app.test_client()
    season = client.get("/api/seasons").json["seasons"][0]
    events = client.get(f"/api/events?season={season}").json["events"]
    assert events
    event = events[0]
    drivers = client.get(f"/api/drivers?event_id={event['event_id']}").json["drivers"]
    assert drivers
    assert all(item["driver_code"] for item in drivers)


def test_upcoming_driver_choices_are_labeled_unconfirmed_history():
    client = app_module.app.test_client()
    events = client.get("/api/events?season=2026").json["events"]
    upcoming = next(event for event in events if event["season_status"] == "upcoming")
    choices = client.get(f"/api/drivers?event_id={upcoming['event_id']}").json
    assert choices["selection_type"] == "historical_candidates"
    assert choices["confirmed"] is False
    assert choices["drivers"]
    assert all("unconfirmed" in item["entry_status"] for item in choices["drivers"])


def test_upcoming_event_cannot_be_backtested():
    client = app_module.app.test_client()
    events = client.get("/api/events?season=2026").json["events"]
    upcoming = next(event for event in events if event["season_status"] == "upcoming")
    response = client.post("/api/backtest", json={"event_id": upcoming["event_id"], "driver": "VER"})
    assert response.status_code == 400
    assert "completed races" in response.json["error"]


def test_invalid_prediction_input_is_rejected():
    client = app_module.app.test_client()
    response = client.post("/api/predict", json={"event_id": "", "driver": ""})
    assert response.status_code == 400


def test_nonparticipant_cannot_be_predicted(monkeypatch):
    import services.prediction_service as prediction_service

    client = app_module.app.test_client()
    year = client.get("/api/seasons").json["seasons"][0]
    event = client.get(f"/api/events?season={year}").json["events"][0]
    monkeypatch.setattr(app_module, "event_prediction_state", lambda *args: {"state": "PRE_RACE"})
    monkeypatch.setattr(prediction_service, "classify_event_status", lambda *args: "COMPLETED")
    monkeypatch.setattr(prediction_service, "event_driver", lambda *args: None)
    monkeypatch.setattr(prediction_service, "_all_laps", lambda: __import__("pandas").DataFrame([{"event_id": "different-event", "Driver": "VER"}]))
    response = client.post("/api/predict", json={"event_id": event["event_id"], "driver": "VER", "fetch_weather": False})
    assert response.status_code == 400
    assert "did not participate" in response.json["error"]


def test_prediction_endpoint_returns_visualization_payload(monkeypatch):
    client = app_module.app.test_client()
    payload = {
        "event": {"race": "Test Grand Prix", "season": 2024},
        "result": {"analysis_mode": "PRE-RACE", "expected_strategy": {"strategy": "MEDIUM → HARD", "pit_laps": [24], "stint_lengths": [24, 29]}, "race_distance": 53, "confidence_score": 61, "model_metrics": {}},
        "timeline": [{"compound": "MEDIUM", "start_lap": 1, "end_lap": 24}],
        "strategy_support": [], "historical_track": {}, "tyre_performance": [], "historical_weather": {},
    }
    monkeypatch.setattr(app_module, "predict_strategy", lambda *args, **kwargs: payload)
    response = client.post("/api/predict", json={"event_id": "any-id", "driver": "VER"})
    assert response.status_code == 200
    assert response.json["result"]["expected_strategy"]["pit_laps"] == [24]
    assert response.json["timeline"]


def test_prediction_endpoint_locks_in_progress_and_completed_events(monkeypatch):
    import pytest

    client = app_module.app.test_client()
    monkeypatch.setattr(app_module, "get_event", lambda event_id: {"event_id": event_id})
    monkeypatch.setattr(app_module, "predict_strategy", lambda *args, **kwargs: pytest.fail("locked event reached predictor"))
    for state, message in (
        ("IN_PROGRESS", "RACE IN PROGRESS — PRE-RACE PREDICTION LOCKED"),
        ("COMPLETED", "RACE COMPLETED — PRE-RACE MODE CLOSED"),
    ):
        monkeypatch.setattr(app_module, "event_prediction_state", lambda event, state=state: {"state": state})
        response = client.post("/api/predict", json={"event_id": "locked-event", "driver": "VER"})
        assert response.status_code == 409
        assert response.json["error"] == message


def test_prediction_endpoint_allows_ready_upcoming_and_pre_race_events(monkeypatch):
    client = app_module.app.test_client()
    calls = []
    monkeypatch.setattr(app_module, "get_event", lambda event_id: {"event_id": event_id})
    monkeypatch.setattr(app_module, "predict_strategy", lambda event_id, driver, **kwargs: calls.append((event_id, driver)) or {"strategy": "MEDIUM → HARD"})
    for state in ("UPCOMING", "PRE_RACE"):
        monkeypatch.setattr(app_module, "event_prediction_state", lambda event, state=state: {"state": state, "pre_race_allowed": True})
        response = client.post("/api/predict", json={"event_id": "ready-event", "driver": "VER"})
        assert response.status_code == 200
        assert response.json["strategy"] == "MEDIUM → HARD"
    assert calls == [("ready-event", "VER"), ("ready-event", "VER")]


def test_backtest_endpoint_rejects_missing_selection():
    response = app_module.app.test_client().post("/api/backtest", json={})
    assert response.status_code == 400
