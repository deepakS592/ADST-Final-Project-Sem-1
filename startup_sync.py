"""Best-effort startup refresh for official calendars and newly completed races."""

from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import date, datetime, timezone

import pandas as pd

from config import CLEAN_LAP_DATA_PATH, RAW_LAP_DATA_PATH, SEASON_DATA_DIR
from services.data_service import app_today, clear_circuit_analysis_cache, _event_id
from services.database import connect, prediction_data_version

LOGGER = logging.getLogger(__name__)
_START_LOCK = threading.Lock()
_STARTED = False
_STATE_LOCK = threading.RLock()
_SYNC_STATE = {
    "sync_status": "idle",
    "last_successful_sync": None,
    "data_version": None,
    "new_events": [],
    "updated_events": [],
    "last_error": None,
}
_BUNDLE_STATES: dict[tuple[str, str, str], dict] = {}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_data_version() -> str:
    return prediction_data_version("9999-12-31T23:59:59Z")


def _record_bundle_state(event_id: str, cutoff: str, version: str, status: str, **details) -> dict:
    key = (str(event_id), str(cutoff), str(version))
    entry = {"event_id": key[0], "cutoff_utc": key[1], "data_version": key[2], "status": status.upper(), "updated_at": _now_utc(), **details}
    with _STATE_LOCK:
        for old_key in [candidate for candidate in _BUNDLE_STATES if candidate[:2] == key[:2] and candidate != key]:
            _BUNDLE_STATES.pop(old_key, None)
        _BUNDLE_STATES[key] = entry
    return dict(entry)


def bundle_lifecycle_status(event_id: str, cutoff: str, version: str) -> dict | None:
    with _STATE_LOCK:
        entry = _BUNDLE_STATES.get((str(event_id), str(cutoff), str(version)))
        return dict(entry) if entry else None


def sync_status_snapshot() -> dict:
    with _STATE_LOCK:
        states = [dict(value) for value in _BUNDLE_STATES.values()]
        result = dict(_SYNC_STATE)
    result.update({
        "data_version": result.get("data_version") or _latest_data_version(),
        "bundles_building": sum(item["status"] == "BUILDING" for item in states),
        "bundles_ready": sum(item["status"] == "READY" for item in states),
        "bundles_stale": sum(item["status"] == "STALE" or (item["status"] == "FAILED" and item.get("stale", False)) for item in states),
        "bundles_failed": sum(item["status"] == "FAILED" for item in states),
        "bundle_events": states,
    })
    return result


def _begin_sync() -> None:
    with _STATE_LOCK:
        _SYNC_STATE.update({
            "sync_status": "running", "data_version": _latest_data_version(),
            "new_events": [], "updated_events": [], "last_error": None,
        })


def _record_import_progress(new_events: list[str], updated_events: list[str] | None = None) -> None:
    with _STATE_LOCK:
        _SYNC_STATE.update({
            "data_version": _latest_data_version(),
            "new_events": list(new_events),
            "updated_events": list(updated_events or []),
        })


def _finish_sync(result: dict, *, error: str | None = None) -> None:
    failed_bundles = any(item.get("status") == "FAILED" for item in result.get("bundle_results", []))
    calendar_result = result.get("calendar") or {}
    failed_sources = bool(calendar_result.get("error")) or any(
        item.get("status") in {"unavailable", "schedule-missing", "no-usable-race-laps"}
        for item in result.get("race_status", [])
    )
    status = "failed" if error else "partial" if failed_bundles or failed_sources else "complete"
    with _STATE_LOCK:
        _SYNC_STATE.update({
            "sync_status": status,
            "last_successful_sync": _now_utc() if status == "complete" else _SYNC_STATE.get("last_successful_sync"),
            "data_version": _latest_data_version(),
            "new_events": list(result.get("new_events", [])),
            "updated_events": list(result.get("updated_events", [])),
            "last_error": str(error)[:500] if error else None,
        })


def _missing_completed_calendar_events(today: date) -> list[dict]:
    """Return recent scheduled races that have passed but have no observed laps."""
    first_year = max(1950, today.year - 1)
    with connect() as connection:
        rows = connection.execute(
            "SELECT e.* FROM events e WHERE e.season BETWEEN ? AND ? AND e.event_date<? "
            "AND NOT EXISTS (SELECT 1 FROM laps l WHERE l.event_id=e.event_id) "
            "AND LOWER(e.season_status) IN ('upcoming','current/scheduled','calendar-only') "
            "ORDER BY e.event_date,e.season,e.round",
            (first_year, today.year, today.isoformat()),
        ).fetchall()
    return [dict(row) for row in rows]


def _download_missing_race_rows(events: list[dict], today: date, fastf1=None) -> tuple[pd.DataFrame, list[dict]]:
    if not events:
        return pd.DataFrame(), []
    from data_loader import _event_metadata, _extract_laps, _load_fastf1, _status_row

    fastf1 = fastf1 or _load_fastf1()
    by_season: dict[int, list[dict]] = {}
    for event in events:
        by_season.setdefault(int(event["season"]), []).append(event)
    frames: list[pd.DataFrame] = []
    statuses: list[dict] = []
    for season, season_events in sorted(by_season.items()):
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        rows_by_round = {
            int(pd.to_numeric(row.get("RoundNumber"), errors="coerce")): row
            for _, row in schedule.iterrows()
            if pd.notna(pd.to_numeric(row.get("RoundNumber"), errors="coerce"))
        }
        cache_dir = SEASON_DATA_DIR / str(season)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for event in season_events:
            round_number = int(event["round"])
            scheduled = rows_by_round.get(round_number)
            if scheduled is None:
                statuses.append({"event_id": event["event_id"], "status": "schedule-missing"})
                continue
            meta = _event_metadata(season, scheduled)
            if not meta.get("EventDate") or date.fromisoformat(meta["EventDate"]) >= today:
                continue
            cache_file = cache_dir / f"round_{round_number:02d}_race_laps.csv"
            try:
                if cache_file.exists():
                    laps = pd.read_csv(cache_file, low_memory=False)
                else:
                    session = fastf1.get_session(season, round_number, "R")
                    session.load(telemetry=False, weather=False, messages=False, laps=True)
                    laps = _extract_laps(session, meta)
                    laps = laps.dropna(subset=["LapTime", "LapNumber", "Driver", "Compound"])
                    if not laps.empty:
                        laps["RaceLaps"] = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        laps.to_csv(cache_file, index=False, encoding="utf-8")
                if laps.empty:
                    statuses.append({"event_id": event["event_id"], "status": "no-usable-race-laps"})
                    continue
                # Keep the canonical SQLite identity in step with the season/round/race contract.
                laps["Season"] = season
                laps["Round"] = round_number
                laps["Race"] = str(event["race"])
                laps["EventName"] = laps.get("EventName", str(event["race"]))
                laps["OfficialEventName"] = laps.get("OfficialEventName", event.get("official_event_name") or event["race"])
                laps["Circuit"] = laps.get("Circuit", event.get("circuit") or "Unknown")
                laps["Country"] = laps.get("Country", event.get("country") or "Unknown")
                laps["Location"] = laps.get("Location", event.get("location") or event.get("circuit") or "Unknown")
                laps["EventDate"] = str(event["event_date"])[:10]
                frames.append(laps)
                statuses.append({"event_id": event["event_id"], "status": "downloaded", "rows": len(laps)})
            except Exception as error:
                LOGGER.warning("Startup sync could not load %s round %s: %s", season, round_number, error)
                statuses.append({"event_id": event["event_id"], "status": "unavailable", "reason": str(error)[:300]})
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), statuses


def _merge_into_existing_lap_source(new_laps: pd.DataFrame) -> int:
    if new_laps.empty:
        return 0
    source = RAW_LAP_DATA_PATH if RAW_LAP_DATA_PATH.exists() else CLEAN_LAP_DATA_PATH
    if not source.exists():
        raise FileNotFoundError("No existing lap snapshot is available to extend safely.")
    existing = pd.read_csv(source, low_memory=False)
    merged = pd.concat([existing, new_laps], ignore_index=True, sort=False)
    keys = ["Season", "Round", "Race", "Driver", "LapNumber"]
    merged = merged.drop_duplicates(keys, keep="last")
    RAW_LAP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_LAP_DATA_PATH.with_suffix(".startup-sync.tmp.csv")
    merged.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(RAW_LAP_DATA_PATH)
    return int(len(merged) - len(existing))


def _build_missing_future_bundles(today: date, *, after_date: str | None = None) -> list[dict]:
    """Build only missing or stale cutoff bundles whose dates can include current data."""
    from pre_race_analysis import build_prediction_bundle, clear_pre_race_caches, prediction_bundle_status

    query = (
        "SELECT * FROM events WHERE event_date>=? "
        "AND LOWER(season_status) IN ('upcoming','current/scheduled')"
    )
    params: list[str] = [today.isoformat()]
    if after_date:
        query += " AND event_date>?"
        params.append(str(after_date)[:10])
    query += " ORDER BY event_date,season,round"
    with connect() as connection:
        targets = [dict(row) for row in connection.execute(query, params)]
    results = []
    for event in targets:
        cutoff = f"{str(event['event_date'])[:10]}T00:00:00Z"
        version = prediction_data_version(cutoff)
        event_id = str(event["event_id"])
        current = prediction_bundle_status(event_id, cutoff, version)
        if current.get("status") == "READY":
            _record_bundle_state(event_id, cutoff, version, "READY", reason=None, stale=False)
            continue
        was_stale = current.get("status") in {"STALE", "FAILED"} or bool(current.get("stale"))
        _record_bundle_state(
            event_id, cutoff, version, "STALE" if was_stale else "MISSING",
            reason=current.get("reason"), stale=was_stale,
        )
        _record_bundle_state(event_id, cutoff, version, "BUILDING", reason=None, stale=was_stale)
        try:
            artifact = build_prediction_bundle(event, cutoff, data_version=version)
            clear_pre_race_caches()
            verified = prediction_bundle_status(event_id, cutoff, version)
            if verified.get("status") != "READY" or not verified.get("available"):
                raise RuntimeError(verified.get("reason") or "Built bundle failed metadata validation.")
            _record_bundle_state(event_id, cutoff, version, "READY", reason=None, stale=False, artifact=str(artifact))
            results.append({"event_id": event_id, "status": "READY", "artifact": str(artifact)})
        except Exception as error:
            reason = str(error)[:500] or error.__class__.__name__
            _record_bundle_state(event_id, cutoff, version, "FAILED", reason=reason, stale=was_stale)
            LOGGER.exception("Startup bundle build failed for %s", event_id)
            results.append({"event_id": event_id, "status": "FAILED", "reason": reason, "stale": was_stale})
    return results


def run_startup_sync(*, today: date | None = None, calendar_updater=None, fastf1_loader=None) -> dict:
    """Refresh the current/next calendar and import newly completed Race sessions.

    This is intentionally an offline preprocessing task. Predictions only read
    the versioned bundle and precomputed summaries produced here.
    """
    current_day = today or app_today()
    _begin_sync()
    try:
        updater = calendar_updater
        try:
            if updater is None:
                from calendar_import import update_calendar
                updater = update_calendar
            calendar_result = updater([current_day.year, current_day.year + 1])
        except Exception as error:
            LOGGER.warning("Official calendar startup refresh was unavailable: %s", error)
            calendar_result = {"error": str(error)[:300]}

        missing = _missing_completed_calendar_events(current_day)
        # The injection seam is used by tests; production uses FastF1's local cache.
        fastf1 = fastf1_loader() if fastf1_loader is not None else None
        laps, statuses = _download_missing_race_rows(missing, current_day, fastf1)
        imported: dict = {}
        added_rows = 0
        if not laps.empty:
            # The data source and SQLite transaction commit before any cutoff
            # fingerprints are read or affected bundles are marked stale.
            added_rows = _merge_into_existing_lap_source(laps)
            from data_preparation import prepare_data
            from services.data_service import import_project_data, clear_circuit_analysis_cache
            from services.prediction_service import clear_lap_cache

            prepare_data(raw_path=RAW_LAP_DATA_PATH)
            imported = import_project_data(force=True, append_only=True)
            clear_lap_cache()
            clear_circuit_analysis_cache()
            from pre_race_analysis import clear_pre_race_caches
            clear_pre_race_caches()
            imported_event_ids = sorted({str(item["event_id"]) for item in statuses if item.get("status") == "downloaded"})
            _record_import_progress(imported_event_ids)
            latest_race_day = max(str(value)[:10] for value in laps["EventDate"].dropna())
            bundle_results = _build_missing_future_bundles(current_day, after_date=latest_race_day)
            clear_circuit_analysis_cache()
        else:
            bundle_results = _build_missing_future_bundles(current_day)

        ready_bundles = [item for item in bundle_results if item.get("status") == "READY"]
        new_events = sorted({str(item["event_id"]) for item in statuses if item.get("status") == "downloaded"})
        result = {
            "calendar": calendar_result,
            "races_imported": int(laps[["Season", "Round", "Race"]].drop_duplicates().shape[0]) if not laps.empty else 0,
            "lap_rows_added": added_rows,
            "sqlite_import": imported,
            "bundles_rebuilt": len(ready_bundles),
            "rebuilt_bundles": ready_bundles,
            "bundle_results": bundle_results,
            "race_status": statuses,
            "new_events": new_events,
            "updated_events": [],
        }
        _finish_sync(result)
        result.update(sync_status_snapshot())
        return result
    except Exception as error:
        LOGGER.exception("Startup sync failed; the existing local snapshot remains available")
        _finish_sync({}, error=str(error))
        raise


def schedule_startup_sync() -> bool:
    """Start one best-effort daemon sync per app process, outside test imports."""
    global _STARTED
    if os.getenv("F1_STARTUP_SYNC", "1").strip().casefold() in {"0", "false", "no", "off"}:
        return False
    # Tests exercise run_startup_sync with a fake source; importing app under
    # pytest must not initiate network requests or mutate the packaged database.
    if "pytest" in sys.modules:
        return False
    with _START_LOCK:
        if _STARTED:
            return False
        _STARTED = True
    def worker() -> None:
        try:
            LOGGER.info("Running race-calendar and completed-race startup sync")
            run_startup_sync()
        except Exception:
            LOGGER.exception("Startup data sync stopped; the existing local snapshot remains available")
    threading.Thread(target=worker, name="f1-startup-sync", daemon=True).start()
    return True
