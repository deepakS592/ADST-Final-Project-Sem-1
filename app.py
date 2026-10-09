"""Flask entry point for the SQLite-backed F1 pit-stop strategy application."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from services.backtesting_service import backtest_event
from services.data_service import app_today, circuit_analysis, classify_event_status, completed_events_for_season, driver_options_for_event, event_prediction_state, events_for_season, focus_event, import_project_data, quality_report, season_catalog, selected_season_context, seasons, track_asset_for_circuit
from services.database import connect, get_event, initialize_schema, table_count
from services.prediction_service import PredictionArtifactUnavailable, clear_lap_cache, predict_strategy
from services.weather_service import cached_weather, historical_weather, live_weather, refresh_historical_weather, store_weather
from startup_sync import schedule_startup_sync, sync_status_snapshot

PROJECT_ROOT = Path(__file__).resolve().parent


def create_app(*, initialize_data: bool = True) -> Flask:
    application = Flask(__name__)
    application.config.update(JSON_SORT_KEYS=False, MAX_CONTENT_LENGTH=64 * 1024)
    initialize_schema()
    if initialize_data and table_count("laps") == 0:
        import_project_data()
    if initialize_data:
        schedule_startup_sync()

    @application.get("/")
    def index():
        return render_template("index.html")

    @application.get("/api/health")
    def health():
        return jsonify({"status": "ok", "database": "SQLite", "lap_rows": table_count("laps")})

    @application.get("/api/sync-status")
    def api_sync_status():
        return jsonify(_json_safe(sync_status_snapshot()))

    def event_bundle_status(event: dict, prediction_state: dict, cutoff_date: str) -> dict:
        if str(prediction_state.get("state", "")).upper() in {"IN_PROGRESS", "COMPLETED"}:
            return {
                "status": "NOT_REQUIRED", "available": False, "event_id": str(event["event_id"]),
                "cutoff_utc": f"{cutoff_date}T00:00:00Z", "data_version": None,
                "reason": "Pre-race prediction is locked for this race state.", "instruction": None,
            }
        from pre_race_analysis import prediction_bundle_status
        return prediction_bundle_status(str(event["event_id"]), f"{cutoff_date}T00:00:00Z")

    @application.get("/api/next-event")
    @application.get("/api/focus-event")
    def api_focus_event():
        event = focus_event()
        if not event:
            return jsonify({"error": "No current or upcoming event is in the local schedule. Refresh the schedule data."}), 503
        choices = driver_options_for_event(str(event["event_id"]))
        with connect() as connection:
            row = connection.execute("SELECT value FROM app_metadata WHERE key='imported_at_utc'").fetchone()
            forecast_row = connection.execute(
                "SELECT payload_json, retrieved_at FROM weather_snapshots WHERE event_id=? AND kind='forecast'",
                (str(event["event_id"]),),
            ).fetchone()
        forecast_available = False
        forecast_updated = None
        if forecast_row:
            forecast_updated = forecast_row["retrieved_at"]
            try:
                forecast_available = bool(json.loads(forecast_row["payload_json"]).get("available"))
            except (TypeError, json.JSONDecodeError):
                pass
        return jsonify(_json_safe({
            "event": event,
            "season_context": event.get("season_context"),
            "drivers": choices["drivers"],
            "driver_selection": {"type": choices["selection_type"], "confirmed": choices["confirmed"]},
            "planning_date": event.get("planning_date"),
            "historical_data_updated_at_utc": row[0] if row else None,
            "forecast_available": forecast_available,
            "forecast_updated_at_utc": forecast_updated,
        }))

    @application.get("/api/event-context/<event_id>")
    def api_event_context(event_id: str):
        event = get_event(event_id)
        if not event:
            return jsonify({"error": "Race not found."}), 404
        status = classify_event_status(event, app_today())
        cutoff_date = app_today().isoformat() if status == "CURRENT" else str(event.get("event_date") or "")[:10]
        if not cutoff_date:
            cutoff_date = app_today().isoformat()
        prediction_state = event_prediction_state(event)
        event = {**event, "race_start_time_utc": prediction_state["race_start_time_utc"]}
        season_context = selected_season_context(int(event["season"]))
        choices = driver_options_for_event(event_id)
        forecast = {"available": False, "retrieved_at": None, "source": None}
        with connect() as connection:
            row = connection.execute(
                "SELECT retrieved_at, source, payload_json FROM weather_snapshots "
                "WHERE event_id=? AND kind='forecast' ORDER BY retrieved_at DESC LIMIT 1",
                (str(event_id),),
            ).fetchone()
        if row:
            try:
                package = json.loads(row["payload_json"])
                forecast = {
                    "available": bool(package.get("available")),
                    "retrieved_at": row["retrieved_at"],
                    "source": row["source"],
                }
            except (TypeError, json.JSONDecodeError):
                pass
        circuit = str(event.get("circuit") or "Unknown")
        history = circuit_analysis(circuit, before=cutoff_date, exclude_event_id=event_id)
        bundle_status = event_bundle_status(event, prediction_state, cutoff_date)
        cached = cached_weather(event_id)
        track = {
            "circuit": circuit,
            "country": event.get("country"),
            "location": event.get("location"),
            "race_laps": event.get("race_laps"),
            "length_km": event.get("track_length_km"),
            "turns": event.get("turns"),
            **track_asset_for_circuit(circuit),
        }
        return jsonify(_json_safe({
            "event": {**event, "event_status": status},
            "season_context": season_context,
            "data_availability": season_context.get("data_availability", {}),
            "race_status": status,
            "prediction_state": prediction_state,
            "race_distance": event.get("race_laps"),
            "race_date": event.get("event_date"),
            "forecast": forecast,
            "cutoff_date": cutoff_date,
            "track": track,
            "circuit_analysis": history,
            "cached_weather": cached,
            "model_bundle": bundle_status,
            "analysis_availability": {
                "track_history": int(history.get("coverage", {}).get("race_count", 0)) > 0,
                "driver_prediction": bool(bundle_status.get("available")),
                "driver_entries_confirmed": bool(choices["confirmed"]),
            },
            "driver_selection": {"type": choices["selection_type"], "confirmed": choices["confirmed"]},
            "drivers": choices["drivers"],
        }))

    @application.get("/api/bundle-status/<event_id>")
    def api_bundle_status(event_id: str):
        event = get_event(event_id)
        if not event:
            return jsonify({"error": "Race not found."}), 404
        status = classify_event_status(event, app_today())
        cutoff_date = app_today().isoformat() if status == "CURRENT" else str(event.get("event_date") or "")[:10]
        if not cutoff_date:
            cutoff_date = app_today().isoformat()
        prediction_state = event_prediction_state(event)
        return jsonify(_json_safe(event_bundle_status(event, prediction_state, cutoff_date)))

    @application.get("/api/seasons")
    def api_seasons():
        catalog = season_catalog()
        if not catalog["seasons"]:
            return jsonify({"error": "No source-backed race calendar is loaded yet. Refresh the calendar first."}), 503
        return jsonify(catalog)

    @application.get("/api/event-state/<event_id>")
    def api_event_state(event_id: str):
        event = get_event(event_id)
        if not event:
            return jsonify({"error": "Race not found."}), 404
        return jsonify(event_prediction_state(event))

    @application.get("/api/events")
    def api_events():
        season_value = request.args.get("season", "")
        if not season_value.isdigit():
            return jsonify({"error": "Choose a valid season."}), 400
        events = events_for_season(int(season_value))
        status_filter = request.args.get("status", "").strip().upper()
        if status_filter in {"UPCOMING", "FUTURE"}:
            events = [event for event in events if event["event_status"] == "UPCOMING"]
        elif status_filter in {"AVAILABLE", "CURRENT_UPCOMING"}:
            events = [event for event in events if event["event_status"] in {"CURRENT", "UPCOMING"}]
        elif status_filter and status_filter not in {"ALL", "ANY"}:
            return jsonify({"error": "Status must be upcoming, available, or all."}), 400
        return jsonify({"season": int(season_value), "season_context": selected_season_context(int(season_value)), "events": events})

    @application.get("/api/historical-events")
    def api_historical_events():
        season_value = request.args.get("season", "")
        if not season_value.isdigit():
            return jsonify({"error": "Choose a valid season."}), 400
        return jsonify({"events": completed_events_for_season(int(season_value))})

    @application.get("/api/drivers")
    def api_drivers():
        event_id = request.args.get("event_id", "").strip()
        if not event_id:
            return jsonify({"error": "Choose a race before selecting a driver."}), 400
        return jsonify(_json_safe(driver_options_for_event(event_id)))

    @application.post("/api/predict")
    def api_predict():
        body = request.get_json(silent=True) or {}
        event_id, driver = str(body.get("event_id", "")).strip(), str(body.get("driver", "")).strip()
        if not event_id or not driver:
            return jsonify({"error": "Select a race and a participating driver."}), 400
        event = get_event(event_id)
        if event:
            state = event_prediction_state(event)["state"]
            if state == "IN_PROGRESS":
                return jsonify({"error": "RACE IN PROGRESS — PRE-RACE PREDICTION LOCKED", "prediction_state": state}), 409
            if state == "COMPLETED":
                return jsonify({"error": "RACE COMPLETED — PRE-RACE MODE CLOSED", "prediction_state": state}), 409
        try:
            prediction = predict_strategy(
                event_id,
                driver,
                fetch_weather=body.get("fetch_weather", True),
                refresh_weather=bool(body.get("refresh_weather", False)),
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except PredictionArtifactUnavailable as error:
            return jsonify({"error": str(error), "artifact_ready": False}), 503
        except Exception as error:
            request_id = uuid.uuid4().hex[:10]
            application.logger.exception("Pit-stop prediction failed")
            return jsonify({
                "error": "The analysis stopped before returning a strategy. Retry once; if it continues, refresh the included data.",
                "request_id": request_id,
                "detail": str(error)[:240],
            }), 500
        return jsonify(_json_safe(prediction))

    @application.post("/api/backtest")
    def api_backtest():
        body = request.get_json(silent=True) or {}
        event_id, driver = str(body.get("event_id", "")).strip(), str(body.get("driver", "")).strip()
        if not event_id or not driver:
            return jsonify({"error": "Select a completed race and a participating driver."}), 400
        try:
            return jsonify(_json_safe(backtest_event(event_id, driver)))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except PredictionArtifactUnavailable as error:
            return jsonify({"error": str(error)}), 503
        except Exception:
            application.logger.exception("Backtest failed")
            return jsonify({"error": "The backtest could not be completed for this race and driver."}), 500

    @application.get("/api/weather/<event_id>")
    def api_weather(event_id: str):
        from services.database import get_event
        event = get_event(event_id)
        if not event:
            return jsonify({"error": "Race not found."}), 404
        force_refresh = request.args.get("refresh", "0").casefold() in {"1", "true", "yes"}
        package = live_weather(event, force_refresh=force_refresh)
        store_weather(event_id, package)
        stored_history = historical_weather(event_id)
        include_historical = request.args.get("history", "0").casefold() in {"1", "true", "yes"}
        if include_historical and not stored_history.get("available"):
            try:
                refresh_historical_weather(event)
                stored_history = historical_weather(event_id)
            except Exception:
                pass
        return jsonify(_json_safe({"live": package, "historical": stored_history, "retrieved_at_utc": package.get("retrieved_at_utc"), "cache_status": package.get("cache_status")}))

    @application.get("/api/quality")
    def api_quality():
        return jsonify(_json_safe(quality_report()))

    @application.post("/api/update")
    def api_update():
        body = request.get_json(silent=True) or {}
        try:
            result = import_project_data(force=bool(body.get("force", True)))
            clear_lap_cache()
            return jsonify(_json_safe(result))
        except Exception:
            application.logger.exception("Local data refresh failed")
            return jsonify({"error": "SQLite could not be refreshed from the included local datasets."}), 500

    @application.get("/api/model-metrics")
    def api_model_metrics():
        metrics_path = PROJECT_ROOT / "data" / "processed" / "model_metrics.json"
        if not metrics_path.exists():
            return jsonify({"metrics": {}, "reason": "No saved model metrics are available."})
        try:
            return jsonify(json.loads(metrics_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return jsonify({"metrics": {}, "reason": "Saved model metrics could not be read."})

    return application


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
