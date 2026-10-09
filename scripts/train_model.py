from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from config import LAP_TIME_MODEL_PATH, MODEL_METRICS_PATH, PIT_STOP_MODEL_PATH, ensure_project_directories
from data_preparation import build_pit_training_data
from services.data_service import app_today, classify_event_status
from services.database import connect, get_event, laps_frame, prediction_data_version
from train_models import train_lap_time_model, train_pit_stop_model


def resolve_event(event_id: str) -> dict | None:
    """Resolve the exact canonical event_id shared by SQLite and the event APIs."""
    return get_event(str(event_id))


def _missing_event_message(event_id: str) -> str:
    message = f"SQLite has no event with id {event_id!r}."
    parts = str(event_id).split("-", 2)
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
        return message
    with connect() as connection:
        matches = connection.execute(
            "SELECT event_id, race FROM events WHERE season=? AND round=? ORDER BY event_id",
            (int(parts[0]), int(parts[1])),
        ).fetchall()
    if len(matches) == 1:
        canonical_id, race = matches[0]
        return f"{message} The canonical ID for {race} (season {parts[0]}, round {parts[1]}) is {canonical_id!r}; rerun with --asof-event {canonical_id}."
    if matches:
        choices = ", ".join(f"{row['race']}: {row['event_id']}" for row in matches)
        return f"{message} SQLite has these event IDs for season {parts[0]}, round {parts[1]}: {choices}."
    return message


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train full-snapshot models or build a cutoff-safe event bundle offline.")
    parser.add_argument("--asof-event", help="Build pre-race models and historical features for a SQLite event_id.")
    parser.add_argument("--asof-upcoming", action="store_true", help="Build bundles for every current or upcoming event in SQLite.")
    parser.add_argument("--cutoff", help="Override the conservative UTC cutoff (ISO date/time) for --asof-event.")
    args = parser.parse_args()
    ensure_project_directories()
    if args.asof_event or args.asof_upcoming:
        if args.asof_upcoming:
            with connect() as connection:
                events = [dict(row) for row in connection.execute("SELECT * FROM events ORDER BY event_date, season, round")]
            targets = [event for event in events if classify_event_status(event, app_today()) in {"CURRENT", "UPCOMING"}]
            if not targets:
                raise RuntimeError("SQLite contains no source-backed current or upcoming event.")
            from pre_race_analysis import build_prediction_bundle

            built = []
            for event in targets:
                status = classify_event_status(event, app_today())
                event_date = str(event.get("event_date") or "")[:10]
                cutoff = f"{app_today().isoformat()}T00:00:00Z" if status == "CURRENT" else f"{event_date}T00:00:00Z"
                version = prediction_data_version(cutoff)
                path = build_prediction_bundle(event, cutoff, data_version=version)
                built.append({"event_id": event["event_id"], "race": event["race"], "cutoff_utc": cutoff, "artifact": str(path)})
            print(json.dumps({"training_mode": "offline cutoff-safe artifacts", "built_count": len(built), "artifacts": built}, indent=2))
            return 0

        event = resolve_event(args.asof_event)
        if not event:
            raise RuntimeError(_missing_event_message(args.asof_event))
        status = classify_event_status(event, app_today())
        event_date = str(event.get("event_date") or "")[:10]
        if not event_date:
            raise RuntimeError("The selected event has no source-backed date.")
        cutoff = args.cutoff or (
            f"{app_today().isoformat()}T00:00:00Z" if status == "CURRENT" else f"{event_date}T00:00:00Z"
        )
        from pre_race_analysis import build_prediction_bundle

        version = prediction_data_version(cutoff)
        path = build_prediction_bundle(event, cutoff, data_version=version)
        print(json.dumps({
            "event_id": event["event_id"],
            "race": event["race"],
            "cutoff_utc": cutoff,
            "data_version": version,
            "artifact": str(path),
            "training_mode": "offline cutoff-safe artifact build",
        }, indent=2))
        return 0

    laps = laps_frame()
    if laps.empty:
        raise RuntimeError("SQLite contains no historical lap data. Run scripts/import_data.py first.")
    pit_training = build_pit_training_data(laps)
    lap_artifact, lap_metrics = train_lap_time_model(laps)
    pit_artifact, pit_metrics = train_pit_stop_model(pit_training)
    if lap_artifact:
        joblib.dump(lap_artifact, LAP_TIME_MODEL_PATH)
    if pit_artifact:
        joblib.dump(pit_artifact, PIT_STOP_MODEL_PATH)
    metrics = {"lap_time": lap_metrics, "pit_stop": pit_metrics, "source": "SQLite laps table", "training_rows": int(len(laps))}
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
