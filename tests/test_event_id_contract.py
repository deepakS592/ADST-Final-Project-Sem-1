from __future__ import annotations

import app as app_module
from scripts.train_model import resolve_event
from services.database import connect


def test_every_event_api_id_exists_in_sqlite_and_resolves_for_training():
    client = app_module.app.test_client()
    event_ids = []

    for endpoint in ("/api/next-event", "/api/focus-event"):
        response = client.get(endpoint)
        assert response.status_code == 200
        event_ids.append(response.json["event"]["event_id"])

    seasons_response = client.get("/api/seasons")
    assert seasons_response.status_code == 200
    for season in seasons_response.json["seasons"]:
        response = client.get(f"/api/events?season={season}&status=all")
        assert response.status_code == 200
        event_ids.extend(event["event_id"] for event in response.json["events"])

    assert event_ids
    with connect() as connection:
        sqlite_ids = {row[0] for row in connection.execute("SELECT event_id FROM events")}

    assert set(event_ids) <= sqlite_ids
    for event_id in set(event_ids):
        training_event = resolve_event(event_id)
        assert training_event is not None
        assert training_event["event_id"] == event_id

