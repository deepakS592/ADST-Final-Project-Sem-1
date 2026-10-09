"""Import the supplied historical snapshot and query the SQLite catalog."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import CALENDAR_UPDATES_PATH, HISTORICAL_CIRCUIT_LAPS_PATH, HISTORICAL_CIRCUIT_PIT_STOPS_PATH
from services.database import PROJECT_ROOT, connect, initialize_schema


COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
EVENT_KEY_COLUMNS = ["Season", "Round", "Race"]
LAP_COLUMNS = [
    "event_id", "Season", "Round", "Race", "EventName", "OfficialEventName", "Circuit",
    "Country", "Location", "EventDate", "RaceLaps", "Driver", "DriverName", "DriverNumber",
    "Team", "Compound", "LapNumber", "TyreLife", "Stint", "LapTime", "IsAccurate", "Deleted",
    "TrackStatus", "FreshTyre", "PitInTime", "PitOutTime", "Position",
]


def _event_id(season: int, round_number: int, race: str) -> str:
    digest = hashlib.sha1(f"{season}|{round_number}|{race}".encode("utf-8")).hexdigest()[:10]
    return f"{season}-{round_number:02d}-{digest}"


def _cutoff_data_versions(
    laps: pd.DataFrame,
    *,
    existing_versions: list[tuple[str, str]] | None = None,
    append_only: bool = False,
) -> list[tuple[str, str]]:
    """Build cumulative date fingerprints as part of the offline import."""
    if laps.empty or not {"event_id", "EventDate"}.issubset(laps.columns):
        return []
    columns = [column for column in LAP_COLUMNS if column in laps.columns]
    frame = laps.copy()
    frame["EventDate"] = pd.to_datetime(frame["EventDate"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    frame = frame[frame["EventDate"].notna()].copy()
    frame = frame.sort_values([column for column in ("EventDate", "event_id", "Driver", "LapNumber") if column in frame])
    day_groups = list(frame.groupby("EventDate", sort=True))
    previous = dict(existing_versions or [])
    if append_only and previous:
        current_days = {str(day) for day, _group in day_groups}
        new_days = sorted(current_days - set(previous))
        old_days = sorted(previous)
        # Preserve preexisting bundle versions when startup only appends later
        # race dates. A backfill inside the old range takes the full rebuild path.
        if old_days and set(old_days).issubset(current_days) and new_days and new_days[0] > old_days[-1]:
            cumulative = previous[old_days[-1]]
            versions = sorted(previous.items())
            day_groups = [(day, group) for day, group in day_groups if str(day) in set(new_days)]
        else:
            cumulative = hashlib.sha256(b"f1-lap-data-v1").hexdigest()
            versions = []
    else:
        cumulative = hashlib.sha256(b"f1-lap-data-v1").hexdigest()
        versions = []
    for day, group in day_groups:
        row_hash = pd.util.hash_pandas_object(group[columns], index=False).values.tobytes()
        day_hash = hashlib.sha256(row_hash).hexdigest()
        cumulative = hashlib.sha256(f"{cumulative}|{day}|{day_hash}".encode("utf-8")).hexdigest()
        versions.append((str(day), cumulative))
    return versions


def _source_path() -> Path:
    preferred = PROJECT_ROOT / "data" / "processed" / "all_seasons_laps_clean.csv"
    fallback = PROJECT_ROOT / "data" / "raw" / "f1_laps_2019_2026.csv"
    for path in (preferred, fallback):
        if path.exists():
            return path
    raise FileNotFoundError("No local F1 lap dataset was found in data/processed or data/raw.")


def quality_checks(frame: pd.DataFrame) -> list[dict]:
    issues: list[dict] = []
    expected = set(LAP_COLUMNS) - {"event_id", "FreshTyre", "PitInTime", "PitOutTime", "Position"}
    missing_columns = sorted(expected - set(frame.columns))
    if missing_columns:
        raise ValueError("Lap source is missing required columns: " + ", ".join(missing_columns))
    checks = [
        ("missing_driver", frame["Driver"].isna() | frame["Driver"].astype(str).str.strip().eq("")),
        ("missing_race", frame["Race"].isna() | frame["Race"].astype(str).str.strip().eq("")),
        ("missing_lap_number", pd.to_numeric(frame["LapNumber"], errors="coerce").isna()),
        ("invalid_lap_time", pd.to_numeric(frame["LapTime"], errors="coerce").le(0)),
        ("unsupported_compound", ~frame["Compound"].astype(str).str.upper().isin(COMPOUNDS)),
    ]
    for code, mask in checks:
        count = int(mask.fillna(True).sum())
        if count:
            issues.append({"issue_code": code, "issue_count": count, "details": "Observed in the imported source; rows are retained for traceability."})
    duplicate_mask = frame.duplicated(["Season", "Round", "Race", "Driver", "LapNumber"], keep="first")
    if duplicate_mask.any():
        issues.append({
            "issue_code": "duplicate_event_driver_lap",
            "issue_count": int(duplicate_mask.sum()),
            "details": "Repeated event, driver, and lap keys are excluded from the application lap table.",
        })
    return issues


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()


def _history_frames(base_laps: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load source-backed older race observations for circuit analysis only."""
    history = _read_csv_if_present(HISTORICAL_CIRCUIT_LAPS_PATH)
    pit_stops = _read_csv_if_present(HISTORICAL_CIRCUIT_PIT_STOPS_PATH)
    if history.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    required = {"Season", "Round", "Race", "Circuit", "EventDate", "Driver", "LapNumber", "LapTime"}
    if required - set(history.columns):
        return pd.DataFrame(), pd.DataFrame(), []
    defaults = {
        "Country": "Unknown", "Location": "Unknown", "RaceLaps": None,
        "Position": None, "Compound": "UNKNOWN", "TyreLife": None, "Stint": None,
        "Source": "Jolpica F1 API",
    }
    for column, value in defaults.items():
        if column not in history:
            history[column] = value
    history["Season"] = pd.to_numeric(history["Season"], errors="coerce")
    history["Round"] = pd.to_numeric(history["Round"], errors="coerce")
    history["LapNumber"] = pd.to_numeric(history["LapNumber"], errors="coerce")
    history["LapTime"] = pd.to_numeric(history["LapTime"], errors="coerce")
    history["RaceLaps"] = pd.to_numeric(history["RaceLaps"], errors="coerce")
    history["Position"] = pd.to_numeric(history["Position"], errors="coerce")
    history["TyreLife"] = pd.to_numeric(history["TyreLife"], errors="coerce")
    history["Stint"] = pd.to_numeric(history["Stint"], errors="coerce")
    history = history.dropna(subset=["Season", "Round", "LapNumber", "LapTime", "EventDate"])
    history = history[history["LapTime"].gt(0)].copy()
    history["Season"] = history["Season"].astype(int)
    history["Round"] = history["Round"].astype(int)
    history["LapNumber"] = history["LapNumber"].astype(int)
    history["Stint"] = history["Stint"].astype("Int64")
    history["Compound"] = history["Compound"].fillna("UNKNOWN").astype(str).str.upper()
    history["Driver"] = history["Driver"].fillna("").astype(str).str.strip().str.upper()
    history = history[history["Driver"].ne("")]

    existing_keys = set()
    if base_laps is not None and not base_laps.empty:
        existing_keys = {
            (int(row.Season), int(row.Round), str(row.Race))
            for row in base_laps[["Season", "Round", "Race"]].drop_duplicates().itertuples(index=False)
        }
    else:
        with connect() as connection:
            existing_keys = {
                (int(row[0]), int(row[1]), str(row[2]))
                for row in connection.execute("SELECT DISTINCT Season,Round,Race FROM laps")
            }
    history = history[
        ~history.apply(lambda row: (int(row.Season), int(row.Round), str(row.Race)) in existing_keys, axis=1)
    ].copy()
    if history.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    history["event_id"] = history.apply(lambda row: _event_id(int(row.Season), int(row.Round), str(row.Race)), axis=1)
    event_rows = []
    for keys, group in history.groupby(EVENT_KEY_COLUMNS, dropna=False, sort=True):
        season, round_number, race = int(keys[0]), int(keys[1]), str(keys[2])
        first = group.sort_values("LapNumber").iloc[0]
        distance = pd.to_numeric(group["RaceLaps"], errors="coerce").dropna()
        event_rows.append({
            "event_id": _event_id(season, round_number, race), "season": season,
            "round": round_number, "race": race, "event_name": race,
            "official_event_name": race, "circuit": str(first["Circuit"]),
            "country": str(first["Country"]), "location": str(first["Location"]),
            "event_date": str(first["EventDate"])[:10],
            "race_laps": int(distance.max()) if not distance.empty else None,
            "season_status": "historical",
        })

    lap_columns = [
        "event_id", "Season", "Round", "Race", "Circuit", "EventDate", "RaceLaps",
        "Driver", "LapNumber", "LapTime", "Position", "Compound", "TyreLife", "Stint", "Source",
    ]
    history = history.reindex(columns=lap_columns).rename(columns={
        "Season": "season", "Round": "round", "Race": "race", "Circuit": "circuit",
        "EventDate": "event_date", "RaceLaps": "race_laps", "Driver": "driver",
        "LapNumber": "lap_number", "LapTime": "lap_time", "Position": "position",
        "Compound": "compound", "TyreLife": "tyre_life", "Stint": "stint", "Source": "source",
    })
    if not pit_stops.empty:
        for column in ("Season", "Round", "Race", "Driver", "PitLap"):
            if column not in pit_stops:
                pit_stops[column] = None
        for column, value in (("StopNumber", None), ("DurationSeconds", None), ("Source", "Jolpica F1 API")):
            if column not in pit_stops:
                pit_stops[column] = value
        pit_stops["Season"] = pd.to_numeric(pit_stops["Season"], errors="coerce")
        pit_stops["Round"] = pd.to_numeric(pit_stops["Round"], errors="coerce")
        pit_stops["PitLap"] = pd.to_numeric(pit_stops["PitLap"], errors="coerce")
        pit_stops["StopNumber"] = pd.to_numeric(pit_stops["StopNumber"], errors="coerce")
        pit_stops["DurationSeconds"] = pd.to_numeric(pit_stops["DurationSeconds"], errors="coerce")
        pit_stops = pit_stops.dropna(subset=["Season", "Round", "Race", "Driver", "PitLap"])
        pit_stops["Season"] = pit_stops["Season"].astype(int)
        pit_stops["Round"] = pit_stops["Round"].astype(int)
        pit_stops["PitLap"] = pit_stops["PitLap"].astype(int)
        pit_stops = pit_stops[
            ~pit_stops.apply(lambda row: (int(row.Season), int(row.Round), str(row.Race)) in existing_keys, axis=1)
        ].copy()
        pit_stops["event_id"] = pit_stops.apply(lambda row: _event_id(int(row.Season), int(row.Round), str(row.Race)), axis=1)
        pit_stops["Driver"] = pit_stops["Driver"].fillna("").astype(str).str.strip().str.upper()
        pit_stops = pit_stops[pit_stops["Driver"].ne("")]
        pit_stops = pit_stops[["event_id", "Driver", "PitLap", "StopNumber", "DurationSeconds", "Source"]].rename(columns={
            "Driver": "driver", "PitLap": "pit_lap", "StopNumber": "stop_number",
            "DurationSeconds": "duration_seconds", "Source": "source",
        })
    else:
        pit_stops = pd.DataFrame(columns=["event_id", "driver", "pit_lap", "stop_number", "duration_seconds", "source"])
    return history, pit_stops, event_rows


def sync_calendar_metadata(connection=None, *, refreshed_seasons: list[int] | None = None) -> int:
    """Import upcoming calendar rows as event metadata without adding entrants or laps."""
    _calendar_race_start_times.cache_clear()
    calendar = _read_csv_if_present(CALENDAR_UPDATES_PATH)
    if calendar.empty:
        return 0
    expected = {"Season", "Round", "Race", "Circuit", "EventDate"}
    if expected - set(calendar.columns):
        return 0
    own_connection = connection is None
    context = connect() if own_connection else None
    conn = context if own_connection else connection
    inserted = 0
    try:
        seasons = sorted({int(year) for year in (refreshed_seasons or [])})
        if seasons:
            placeholders = ",".join("?" for _ in seasons)
            conn.execute(
                f"DELETE FROM events WHERE season IN ({placeholders}) "
                "AND LOWER(season_status) IN ('upcoming','current/scheduled','calendar-only') "
                "AND NOT EXISTS (SELECT 1 FROM laps WHERE laps.event_id=events.event_id) "
                "AND NOT EXISTS (SELECT 1 FROM event_drivers WHERE event_drivers.event_id=events.event_id) "
                "AND NOT EXISTS (SELECT 1 FROM weather_observations WHERE weather_observations.event_id=events.event_id) "
                "AND NOT EXISTS (SELECT 1 FROM weather_snapshots WHERE weather_snapshots.event_id=events.event_id) "
                "AND NOT EXISTS (SELECT 1 FROM circuit_history_laps WHERE circuit_history_laps.event_id=events.event_id) "
                "AND NOT EXISTS (SELECT 1 FROM circuit_history_pit_stops WHERE circuit_history_pit_stops.event_id=events.event_id)",
                seasons,
            )
        for row in calendar.to_dict(orient="records"):
            try:
                season, round_number = int(row["Season"]), int(row["Round"])
                race = str(row["Race"]).strip()
                event_date = str(row["EventDate"])[:10]
                race_day = date.fromisoformat(event_date)
            except (TypeError, ValueError, KeyError):
                continue
            if not race:
                continue
            event_id = _event_id(season, round_number, race)
            race_laps = pd.to_numeric(pd.Series([row.get("RaceLaps")]), errors="coerce").iloc[0]
            cursor = conn.execute(
                "INSERT OR IGNORE INTO events(event_id,season,round,race,event_name,official_event_name,circuit,country,location,event_date,race_laps,season_status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, season, round_number, race, race,
                 str(row.get("OfficialEventName") or race), str(row.get("Circuit") or "Unknown"),
                 str(row.get("Country") or "Unknown"), str(row.get("Location") or row.get("Circuit") or "Unknown"),
                 event_date, int(race_laps) if pd.notna(race_laps) and race_laps > 0 else None,
                 "upcoming" if race_day > app_today() else "current/scheduled" if race_day == app_today() else "calendar-only"),
            )
            inserted += int(cursor.rowcount > 0)
        if own_connection:
            conn.commit()
    finally:
        if own_connection:
            conn.close()
    return inserted


def sync_circuit_history_data(connection=None, *, base_laps: pd.DataFrame | None = None) -> dict[str, int]:
    """Refresh auxiliary history tables without changing model-training lap data."""
    history, pit_stops, event_rows = _history_frames(base_laps)
    own_connection = connection is None
    conn = connect() if own_connection else connection
    try:
        old_ids = [row[0] for row in conn.execute("SELECT DISTINCT event_id FROM circuit_history_laps")]
        conn.execute("DELETE FROM circuit_history_pit_stops")
        conn.execute("DELETE FROM circuit_history_laps")
        for event_id in old_ids:
            conn.execute(
                "DELETE FROM events WHERE event_id=? AND NOT EXISTS (SELECT 1 FROM laps WHERE laps.event_id=events.event_id)",
                (event_id,),
            )
        if event_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO events(event_id,season,round,race,event_name,official_event_name,circuit,country,location,event_date,race_laps,season_status) "
                "VALUES (:event_id,:season,:round,:race,:event_name,:official_event_name,:circuit,:country,:location,:event_date,:race_laps,:season_status)",
                event_rows,
            )
        if not history.empty:
            history.where(pd.notna(history), None).to_sql("circuit_history_laps", conn, if_exists="append", index=False, chunksize=5000)
        if not pit_stops.empty:
            pit_stops.where(pd.notna(pit_stops), None).to_sql("circuit_history_pit_stops", conn, if_exists="append", index=False, chunksize=1000)
        if own_connection:
            conn.commit()
        return {"races": len(event_rows), "laps": len(history), "pit_stops": len(pit_stops)}
    finally:
        if own_connection:
            conn.close()


def import_project_data(*, force: bool = False, append_only: bool = False) -> dict:
    """Create or refresh all analytical tables from the included real-data snapshot."""
    initialize_schema()
    source = _source_path()
    processed = PROJECT_ROOT / "data" / "processed"
    laps = pd.read_csv(source, low_memory=False)
    issues = quality_checks(laps)
    if "Round" not in laps:
        laps["Round"] = 0
    for column in LAP_COLUMNS:
        if column not in laps:
            laps[column] = None
    laps["Season"] = pd.to_numeric(laps["Season"], errors="coerce").astype("Int64")
    laps["Round"] = pd.to_numeric(laps["Round"], errors="coerce").fillna(0).astype(int)
    laps["RaceLaps"] = pd.to_numeric(laps["RaceLaps"], errors="coerce")
    laps["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce")
    laps["LapTime"] = pd.to_numeric(laps["LapTime"], errors="coerce")
    laps["TyreLife"] = pd.to_numeric(laps["TyreLife"], errors="coerce")
    laps["Stint"] = pd.to_numeric(laps["Stint"], errors="coerce")
    laps["Position"] = pd.to_numeric(laps["Position"], errors="coerce")
    laps = laps.drop_duplicates(["Season", "Round", "Race", "Driver", "LapNumber"], keep="first").copy()
    for column in ("Race", "Driver", "Compound", "Circuit"):
        laps[column] = laps[column].fillna("").astype(str).str.strip()
    laps["Driver"] = laps["Driver"].str.upper()
    laps["Compound"] = laps["Compound"].str.upper()
    event_rows = []
    event_ids: dict[tuple[int, int, str], str] = {}
    for keys, group in laps.groupby(EVENT_KEY_COLUMNS, dropna=False, sort=True):
        season, round_number, race = int(keys[0]), int(keys[1]), str(keys[2])
        event_id = _event_id(season, round_number, race)
        event_ids[(season, round_number, race)] = event_id
        first = group.iloc[0]
        observed_laps = pd.to_numeric(group["RaceLaps"], errors="coerce").dropna()
        event_rows.append({
            "event_id": event_id,
            "season": season,
            "round": round_number,
            "race": race,
            "event_name": first.get("EventName"),
            "official_event_name": first.get("OfficialEventName"),
            "circuit": first.get("Circuit"),
            "country": first.get("Country"),
            "location": first.get("Location"),
            "event_date": str(first.get("EventDate", ""))[:10],
            "race_laps": int(observed_laps.max()) if not observed_laps.empty else None,
            "season_status": "current/partial" if season == datetime.now(timezone.utc).year else "historical",
        })

    # Upcoming races are included only when the supplied source schedule marks
    # them as unrun. They have no event_drivers rows until the source records entries.
    status_path = processed / "download_status.csv"
    if status_path.exists():
        try:
            from data_loader import build_event_calendar

            schedule = pd.read_csv(status_path, low_memory=False)
            upcoming = build_event_calendar(laps, schedule)
            for row in upcoming.to_dict(orient="records"):
                if str(row.get("EventStatus", "")).upper() not in {"CURRENT", "UPCOMING"}:
                    continue
                season = int(row["Season"])
                round_number = int(row.get("Round") or 0)
                race = str(row["Race"])
                key = (season, round_number, race)
                if key in event_ids:
                    continue
                event_id = _event_id(season, round_number, race)
                event_ids[key] = event_id
                distance = pd.to_numeric(pd.Series([row.get("RaceLaps")]), errors="coerce").iloc[0]
                event_rows.append({
                    "event_id": event_id, "season": season, "round": round_number, "race": race,
                    "event_name": race, "official_event_name": row.get("OfficialEventName"),
                    "circuit": row.get("Circuit"), "country": row.get("Country"), "location": row.get("Location"),
                    "event_date": str(row.get("EventDate", ""))[:10],
                    "race_laps": int(distance) if pd.notna(distance) and distance > 0 else None,
                    "season_status": "upcoming" if str(row.get("EventStatus", "")).upper() == "UPCOMING" else "current/scheduled",
                })
        except Exception as error:
            issues.append({"issue_code": "upcoming_schedule_import", "issue_count": 1, "details": f"Schedule rows were not imported: {error}"})
    laps["event_id"] = [event_ids[(int(row.Season), int(row.Round), str(row.Race))] for row in laps.itertuples()]
    laps = laps[LAP_COLUMNS]

    processed = PROJECT_ROOT / "data" / "processed"
    pit_path = processed / "pit_strategy_training_data.csv"
    tyre_path = processed / "tyre_performance.csv"
    degradation_path = processed / "tyre_degradation.csv"
    weather_path = processed / "weather_analysis.csv"
    pit = pd.read_csv(pit_path) if pit_path.exists() else pd.DataFrame()
    tyre = pd.read_csv(tyre_path) if tyre_path.exists() else pd.DataFrame()
    degradation = pd.read_csv(degradation_path) if degradation_path.exists() else pd.DataFrame()
    weather = pd.read_csv(weather_path) if weather_path.exists() else pd.DataFrame()
    observed_event_keys = {
        (int(season), str(race)) for season, race in laps[["Season", "Race"]].drop_duplicates().itertuples(index=False, name=None)
    }
    weather_event_keys = set()
    if not weather.empty:
        weather_event_keys = {
            (int(row["Season"]), str(row["Race"]))
            for row in weather.to_dict(orient="records")
            if pd.notna(row.get("Season")) and str(row.get("Race", "")).strip()
        }
    missing_weather_events = len(observed_event_keys - weather_event_keys)
    if missing_weather_events:
        issues.append({
            "issue_code": "missing_weather_events",
            "issue_count": missing_weather_events,
            "details": "Historical hourly weather is not loaded for every lap event. Missing event weather is fetched on demand when the provider has it.",
        })

    driver_rows = (
        laps.groupby(["event_id", "Driver"], as_index=False)
        .agg(driver_name=("DriverName", "first"), driver_number=("DriverNumber", "first"), team=("Team", "last"))
        .rename(columns={"Driver": "driver_code"})
    )

    with connect() as connection:
        existing = int(connection.execute("SELECT COUNT(*) FROM laps").fetchone()[0])
        previous_versions = [
            (str(row[0]), str(row[1]))
            for row in connection.execute("SELECT event_date,cumulative_version FROM event_data_versions ORDER BY event_date")
        ]
        previous_lap_marker = connection.execute("SELECT value FROM app_metadata WHERE key='lap_rows'").fetchone()
        previous_lap_count = int(previous_lap_marker[0]) if previous_lap_marker and str(previous_lap_marker[0]).isdigit() else existing
        if existing and not force:
            calendar_added = sync_calendar_metadata(connection)
            history_counts = sync_circuit_history_data(connection)
            return {
                "imported": False, "reason": "SQLite already contains imported laps.",
                "events": int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
                "laps": existing, "calendar_events_added": calendar_added,
                **{key: value for key, value in history_counts.items() if key != "laps"},
                "circuit_history_laps": history_counts["laps"],
            }
        weather_columns = ["event_id", "season", "race", "circuit", "country", "observed_at", "temperature", "precipitation", "rain", "precipitation_probability", "wind_speed", "wind_direction", "weather_code", "weather", "condition", "impact", "source"]
        snapshot_columns = ["event_id", "kind", "retrieved_at", "source", "payload_json"]
        previous_weather = [dict(row) for row in connection.execute("SELECT " + ", ".join(weather_columns) + " FROM weather_observations")]
        previous_snapshots = [dict(row) for row in connection.execute("SELECT " + ", ".join(snapshot_columns) + " FROM weather_snapshots")]
        for table in ("weather_snapshots", "weather_observations", "historical_feature_cache", "event_data_versions", "circuit_history_pit_stops", "circuit_history_laps", "pit_transitions", "tyre_performance", "tyre_degradation", "event_drivers", "laps", "events"):
            connection.execute(f"DELETE FROM {table}")
        pd.DataFrame(event_rows).to_sql("events", connection, if_exists="append", index=False, chunksize=1000)
        laps.to_sql("laps", connection, if_exists="append", index=False, chunksize=5000)
        connection.executemany(
            "INSERT INTO event_data_versions(event_date,cumulative_version) VALUES (?,?)",
            _cutoff_data_versions(
                laps,
                existing_versions=previous_versions,
                append_only=bool(append_only and previous_versions and len(laps) > previous_lap_count),
            ),
        )
        calendar_added = sync_calendar_metadata(connection)
        history_counts = sync_circuit_history_data(connection, base_laps=laps)
        driver_rows.to_sql("event_drivers", connection, if_exists="append", index=False, chunksize=5000)
        if not pit.empty:
            pit.rename(columns={
                "Season": "season", "Round": "round", "Race": "race", "Circuit": "circuit", "Team": "team",
                "Driver": "driver", "PitLap": "pit_lap", "CurrentLap": "current_lap", "RaceLaps": "race_laps",
                "RaceProgress": "race_progress", "PreviousTyre": "previous_tyre", "PreviousTyreAge": "previous_tyre_age",
                "NewTyre": "new_tyre", "AverageLapTimeBefore": "average_lap_time_before", "AverageLapTimeAfter": "average_lap_time_after",
            }).to_sql("pit_transitions", connection, if_exists="append", index=False, chunksize=2000)
        if not tyre.empty:
            tyre.rename(columns={
                "Compound": "compound", "DataPoints": "data_points",
                "CompoundOffset": "compound_offset", "OffsetSource": "offset_source",
                "DegradationRate": "degradation_rate", "DegradationSource": "degradation_source",
                "StintsUsed": "stints_used", "DegradationStd": "degradation_std",
                "R2": "r_squared",
            }).to_sql("tyre_performance", connection, if_exists="append", index=False)
        if not degradation.empty:
            degradation.rename(columns={"Compound": "compound", "TyreLife": "tyre_life", "LapTime": "lap_time"}).to_sql("tyre_degradation", connection, if_exists="append", index=False, chunksize=1000)
        if not weather.empty:
            weather_rows = []
            for row in weather.to_dict(orient="records"):
                season = int(row["Season"]) if pd.notna(row.get("Season")) else None
                race = str(row.get("Race", ""))
                candidates = [record for record in event_rows if record["season"] == season and record["race"] == race]
                event_id = candidates[0]["event_id"] if candidates else None
                observed_at = str(row.get("Time") or row.get("Date") or "")
                weather_rows.append({
                    "event_id": event_id, "season": season, "race": race, "circuit": row.get("Circuit"),
                    "country": row.get("Country"), "observed_at": observed_at,
                    "temperature": row.get("Temperature"), "precipitation": row.get("Precipitation"),
                    "rain": row.get("Rain"), "precipitation_probability": row.get("PrecipitationProbability"),
                    "wind_speed": row.get("WindSpeed"), "wind_direction": row.get("WindDirection"),
                    "weather_code": row.get("WeatherCode"), "weather": row.get("Weather"),
                    "condition": row.get("Condition"), "impact": row.get("Impact"), "source": row.get("Source"),
                })
            pd.DataFrame(weather_rows).to_sql("weather_observations", connection, if_exists="append", index=False, chunksize=2000)
        valid_event_ids = {row[0] for row in connection.execute("SELECT event_id FROM events")}
        current_weather_keys = {
            (row[0], row[1], row[2])
            for row in connection.execute("SELECT event_id, observed_at, source FROM weather_observations")
        }
        restore_weather = []
        for row in previous_weather:
            key = (row["event_id"], row["observed_at"], row["source"])
            if row["event_id"] in valid_event_ids and key not in current_weather_keys:
                restore_weather.append(tuple(row[column] for column in weather_columns))
                current_weather_keys.add(key)
        if restore_weather:
            placeholders = ", ".join("?" for _ in weather_columns)
            connection.executemany(
                f"INSERT INTO weather_observations({', '.join(weather_columns)}) VALUES ({placeholders})",
                restore_weather,
            )
        valid_snapshots = [row for row in previous_snapshots if row["event_id"] in valid_event_ids]
        if valid_snapshots:
            placeholders = ", ".join("?" for _ in snapshot_columns)
            connection.executemany(
                f"INSERT INTO weather_snapshots({', '.join(snapshot_columns)}) VALUES ({placeholders})",
                [tuple(row[column] for column in snapshot_columns) for row in valid_snapshots],
            )
        now = datetime.now(timezone.utc).isoformat()
        connection.execute("DELETE FROM data_quality_issues")
        for issue in issues:
            connection.execute(
                "INSERT INTO data_quality_issues(checked_at, issue_code, issue_count, details) VALUES (?, ?, ?, ?)",
                (now, issue["issue_code"], issue["issue_count"], issue["details"]),
            )
        connection.executemany("INSERT OR REPLACE INTO app_metadata(key, value) VALUES (?, ?)", [
            ("source_path", str(source)), ("imported_at_utc", now), ("lap_rows", str(len(laps))),
        ])
    clear_circuit_analysis_cache()
    return {
        "imported": True, "source": str(source), "events": len(event_rows) + calendar_added + history_counts["races"],
        "laps": len(laps), "drivers": len(driver_rows), "pit_transitions": len(pit),
        "weather_rows": len(weather), "calendar_events_added": calendar_added,
        **{key: value for key, value in history_counts.items() if key != "laps"},
        "circuit_history_laps": history_counts["laps"], "quality_issues": issues,
    }


def seasons() -> list[int]:
    with connect() as connection:
        return [int(row[0]) for row in connection.execute("SELECT DISTINCT season FROM events ORDER BY season DESC")]


def events_for_season(season: int) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT e.*, (SELECT COUNT(*) FROM event_drivers d WHERE d.event_id=e.event_id) AS driver_count, "
            "(SELECT COUNT(*) FROM laps l WHERE l.event_id=e.event_id) AS observed_lap_count, "
            "(SELECT COUNT(*) FROM circuit_history_laps h WHERE h.event_id=e.event_id) AS historical_lap_count "
            "FROM events e WHERE season=? ORDER BY event_date, round, race",
            (int(season),),
        ).fetchall()
    today = app_today()
    events = [dict(row) for row in rows]
    for event in events:
        event["race_start_time_utc"] = event_race_start_time_utc(event)
        event["event_status"] = classify_event_status(event, today)
        event["has_race_observations"] = int(event.get("observed_lap_count") or 0) > 0
        event["has_historical_observations"] = int(event.get("historical_lap_count") or 0) > 0
        event["data_kind"] = (
            "CURRENT-SEASON OBSERVATIONS" if event["has_race_observations"]
            else "HISTORICAL DATA" if event["has_historical_observations"]
            else "CALENDAR DATA"
        )
    return events


def upcoming_events_for_season(season: int, on_date: date | None = None) -> list[dict]:
    """Return every current or future event in a season, in race-date order."""
    today = app_today(on_date)
    return [event for event in events_for_season(season) if classify_event_status(event, today) in {"CURRENT", "UPCOMING"}]


def current_event(season: int | None = None, on_date: date | None = None) -> dict | None:
    today = app_today(on_date)
    candidates = events_for_season(season) if season is not None else [event for year in seasons() for event in events_for_season(year)]
    return next((event for event in candidates if classify_event_status(event, today) == "CURRENT"), None)


def next_event(season: int | None = None, on_date: date | None = None) -> dict | None:
    today = app_today(on_date)
    candidates = events_for_season(season) if season is not None else [event for year in seasons() for event in events_for_season(year)]
    return next((event for event in candidates if classify_event_status(event, today) == "UPCOMING"), None)


def app_today(on_date: date | None = None) -> date:
    """Return today's date in the configured race-planning timezone."""
    if on_date is not None:
        return on_date
    return datetime.now(app_timezone()).date()


def app_timezone() -> ZoneInfo:
    """Return the configured timezone used to interpret race dates and current time."""
    zone_name = os.getenv("F1_APP_TIMEZONE", "Asia/Kolkata")
    try:
        return ZoneInfo(zone_name)
    except Exception:
        return ZoneInfo("UTC")


@lru_cache(maxsize=1)
def _calendar_race_start_times() -> dict[tuple[int, int, str], str]:
    """Read scheduled UTC race starts from the existing calendar-update source."""
    calendar = _read_csv_if_present(CALENDAR_UPDATES_PATH)
    required = {"Season", "Round", "Race", "RaceStartTimeUTC"}
    if calendar.empty or required - set(calendar.columns):
        return {}
    starts: dict[tuple[int, int, str], str] = {}
    for row in calendar.to_dict(orient="records"):
        try:
            key = (int(row["Season"]), int(row["Round"]), str(row["Race"]).strip())
            value = str(row.get("RaceStartTimeUTC") or "").strip()
            timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        except (TypeError, ValueError):
            continue
        if key[2] and pd.notna(timestamp):
            starts[key] = timestamp.isoformat().replace("+00:00", "Z")
    return starts


def event_race_start_time_utc(event: dict) -> str | None:
    """Resolve source-backed race start time from an event row or calendar update."""
    value = event.get("race_start_time_utc") or event.get("RaceStartTimeUTC")
    if not value:
        try:
            key = (int(event["season"]), int(event["round"]), str(event["race"]).strip())
        except (KeyError, TypeError, ValueError):
            return None
        value = _calendar_race_start_times().get(key)
    if not value:
        return None
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp.isoformat().replace("+00:00", "Z") if pd.notna(timestamp) else None


def event_prediction_state(event: dict, now: datetime | None = None) -> dict:
    """Classify a selected event's pre-race window using schedule time and app timezone."""
    zone = app_timezone()
    current = now or datetime.now(zone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=zone)
    current_local = current.astimezone(zone)
    try:
        race_day = date.fromisoformat(str(event.get("event_date", ""))[:10])
    except ValueError:
        phase = "UPCOMING" if event.get("season_status") in {"upcoming", "current/scheduled"} else "COMPLETED"
        return {
            "state": phase, "pre_race_allowed": phase == "UPCOMING",
            "race_start_time_utc": event_race_start_time_utc(event), "timezone": zone.key,
            "start_time_available": False,
        }

    race_start_time = event_race_start_time_utc(event)
    race_start = pd.to_datetime(race_start_time, utc=True, errors="coerce") if race_start_time else pd.NaT
    if pd.notna(race_start):
        if current.astimezone(timezone.utc) < race_start.to_pydatetime():
            phase = "PRE_RACE" if current_local.date() >= race_day - timedelta(days=2) else "UPCOMING"
        elif current_local.date() <= race_day:
            phase = "IN_PROGRESS"
        else:
            phase = "COMPLETED"
    elif current_local.date() < race_day - timedelta(days=2):
        phase = "UPCOMING"
    elif current_local.date() < race_day:
        phase = "PRE_RACE"
    elif current_local.date() == race_day:
        # With no published start time, block from race-day midnight rather than
        # risk presenting a pre-race forecast after the race has begun.
        phase = "IN_PROGRESS"
    else:
        phase = "COMPLETED"

    return {
        "state": phase,
        "pre_race_allowed": phase in {"UPCOMING", "PRE_RACE"},
        "race_start_time_utc": race_start_time,
        "timezone": zone.key,
        "start_time_available": bool(race_start_time),
    }


def classify_event_status(event: dict, today: date | None = None) -> str:
    """Classify an event dynamically from its race date and weekend window."""
    current_day = app_today(today)
    try:
        race_day = date.fromisoformat(str(event.get("event_date", ""))[:10])
    except ValueError:
        return "UPCOMING" if event.get("season_status") in {"upcoming", "current/scheduled"} else "COMPLETED"
    weekend_start = race_day - timedelta(days=2)
    if weekend_start <= current_day <= race_day:
        return "CURRENT"
    if race_day > current_day:
        return "UPCOMING"
    return "COMPLETED"


def season_state(season: int, on_date: date | None = None, now: datetime | None = None) -> dict:
    """Summarize one calendar season without treating schedule rows as results."""
    today = app_today(on_date)
    events = events_for_season(int(season))
    if not events:
        return {"season": int(season), "state": "UNAVAILABLE", "label": "CALENDAR UNAVAILABLE",
                "calendar_events": 0, "observed_races": 0, "message": "No source-backed calendar rows are available."}
    current = [row for row in events if classify_event_status(row, today) == "CURRENT"]
    upcoming = [row for row in events if classify_event_status(row, today) == "UPCOMING"]
    observed_races = sum(bool(row.get("has_race_observations")) for row in events)
    historical_races = sum(bool(row.get("has_historical_observations")) for row in events)
    phase = None
    if current:
        phase = event_prediction_state(current[0], now=now)["state"]
        state = "RACE IN PROGRESS" if phase == "IN_PROGRESS" else "CURRENT RACE"
    elif upcoming:
        progressed = observed_races > 0 or any(
            date.fromisoformat(str(row["event_date"])[:10]) < today for row in events if row.get("event_date")
        )
        state = "ACTIVE SEASON" if progressed else "PRE-SEASON"
    else:
        state = "SEASON COMPLETE"
    message = None
    if observed_races == 0 and historical_races == 0:
        message = f"No {int(season)} race observations available yet. Predictions use historical circuit, driver, tyre and weather information."
    with connect() as connection:
        has_historical_laps = (
            connection.execute("SELECT 1 FROM laps LIMIT 1").fetchone() is not None
            or connection.execute("SELECT 1 FROM circuit_history_laps LIMIT 1").fetchone() is not None
        )
    return {
        "season": int(season), "state": state, "label": state,
        "calendar_events": len(events), "observed_races": int(observed_races),
        "historical_races": int(historical_races),
        "current_event_id": current[0]["event_id"] if current else None,
        "next_event_id": upcoming[0]["event_id"] if upcoming else None,
        "race_phase": phase, "message": message,
        "data_availability": {
            "calendar_data": True,
            "historical_data": has_historical_laps,
            "current_season_observations": int(observed_races),
            "future_season_pre_race_context": state == "PRE-SEASON",
        },
    }


def season_catalog(on_date: date | None = None, now: datetime | None = None) -> dict:
    """Return all calendar seasons and the primary season selected by schedule state."""
    years = seasons()
    states = [season_state(year, on_date, now) for year in years]
    current = [item for item in states if item["state"] in {"CURRENT RACE", "RACE IN PROGRESS"}]
    active = [item for item in states if item["state"] == "ACTIVE SEASON"]
    preseason = [item for item in states if item["state"] == "PRE-SEASON"]
    complete = [item for item in states if item["state"] == "SEASON COMPLETE"]
    if current:
        primary = current[0]
    elif active:
        primary = max(active, key=lambda row: row["season"])
    elif preseason:
        # When several future calendars have been published, show the nearest
        # one after the last completed season.
        primary = min(preseason, key=lambda row: row["season"])
    elif complete:
        primary = max(complete, key=lambda row: row["season"])
    else:
        primary = None
    return {
        "seasons": years,
        "active_season": primary["season"] if primary else None,
        "active_season_state": primary["state"] if primary else "UNAVAILABLE",
        "season_states": states,
    }


def selected_season_context(season: int, on_date: date | None = None, now: datetime | None = None) -> dict:
    """Explain whether a user-selected season is active, complete, or pre-season."""
    catalog = season_catalog(on_date, now)
    selected = next((item for item in catalog["season_states"] if item["season"] == int(season)), None)
    if selected is None:
        return {"season": int(season), "state": "UNAVAILABLE", "label": "CALENDAR UNAVAILABLE",
                "message": "No source-backed calendar rows are available for this season."}
    context = dict(selected)
    if selected["state"] == "PRE-SEASON":
        active = next((item for item in catalog["season_states"]
                       if item["season"] == catalog["active_season"] and item["state"] not in {"PRE-SEASON", "SEASON COMPLETE"}), None)
        lead = f"The {active['season']} season is still in progress. " if active else ""
        context["label"] = f"{int(season)} PRE-SEASON PREVIEW"
        context["message"] = (
            f"{lead}No {int(season)} race observations available yet. "
            "Predictions use historical circuit, driver, tyre and weather information."
        )
    return context


def focus_event(on_date: date | None = None, now: datetime | None = None) -> dict | None:
    """Return the focus event for the active season, including pre-season calendars."""
    today = app_today(on_date)
    catalog = season_catalog(today, now)
    selected_season = catalog["active_season"]
    if selected_season is None:
        return None
    events = events_for_season(selected_season)
    current = [event for event in events if classify_event_status(event, today) == "CURRENT"]
    upcoming = [event for event in events if classify_event_status(event, today) == "UPCOMING"]
    if current:
        selected = current[0]
    elif upcoming:
        selected = upcoming[0]
    elif events:
        selected = events[-1]
    else:
        return None
    race_day = date.fromisoformat(str(selected["event_date"])[:10])
    selected["race_start_time_utc"] = event_race_start_time_utc(selected)
    selected["event_status"] = classify_event_status(selected, today)
    selected["weekend_start"] = (race_day - timedelta(days=2)).isoformat()
    selected["weekend_end"] = race_day.isoformat()
    selected["planning_date"] = today.isoformat()
    selected["race_phase"] = event_prediction_state(selected, now=now)["state"]
    selected["season_context"] = selected_season_context(selected_season, today, now)
    return selected


def completed_events_for_season(season: int, on_date: date | None = None) -> list[dict]:
    """Return only completed events for the separate historical workflow."""
    today = app_today(on_date)
    return [event for event in events_for_season(season) if classify_event_status(event, today) == "COMPLETED"]


def drivers_for_event(event_id: str) -> list[dict]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM event_drivers WHERE event_id=? ORDER BY driver_code", (event_id,)).fetchall()
    return [dict(row) for row in rows]


def driver_options_for_event(event_id: str) -> dict:
    """Return actual event participants or explicitly unconfirmed historic candidates."""
    actual = drivers_for_event(event_id)
    if actual:
        return {"drivers": actual, "selection_type": "observed_participants", "confirmed": True}
    with connect() as connection:
        event_row = connection.execute("SELECT season, event_date, season_status FROM events WHERE event_id=?", (event_id,)).fetchone()
        if not event_row:
            return {"drivers": [], "selection_type": "unavailable", "confirmed": False}
        if event_row["season_status"] not in {"upcoming", "current/scheduled"}:
            return {"drivers": [], "selection_type": "unavailable", "confirmed": False}
        season = int(event_row["season"])
        cutoff = str(event_row["event_date"])
        has_season_history = connection.execute(
            "SELECT 1 FROM laps WHERE Season=? AND EventDate < ? LIMIT 1", (season, cutoff),
        ).fetchone() is not None
        where = "EventDate < ? AND Driver IS NOT NULL AND TRIM(Driver) <> ''"
        params: list[object] = [cutoff]
        if has_season_history:
            where += " AND Season=?"
            params.append(season)
        query = (
            "SELECT Driver AS driver_code, DriverName AS driver_name, DriverNumber AS driver_number, Team AS team "
            "FROM (SELECT Driver, DriverName, DriverNumber, Team, ROW_NUMBER() OVER (PARTITION BY Driver ORDER BY EventDate DESC, Season DESC, LapNumber DESC) AS rn "
            f"FROM laps WHERE {where}) WHERE rn=1 ORDER BY Driver"
        )
        candidates = [dict(row) for row in connection.execute(query, params)]
    for row in candidates:
        row["entry_status"] = "historical candidate; event entry unconfirmed"
    return {"drivers": candidates, "selection_type": "historical_candidates", "confirmed": False}


TRACK_ASSET_ALIASES = {
    "sepang": "sepang", "sepang international circuit": "sepang",
    "marina bay": "singapore", "marina bay street circuit": "singapore", "singapore": "singapore",
    "austin": "cota", "circuit of the americas": "cota",
    "mexico city": "mexico_city", "autodromo hermanos rodriguez": "mexico_city",
    "autódromo hermanos rodríguez": "mexico_city", "autodromo hermanos rodríguez": "mexico_city",
    "sao paulo": "interlagos", "são paulo": "interlagos", "interlagos": "interlagos",
    "las vegas": "las_vegas", "lusail": "lusail", "yas island": "yas_marina",
    "yas marina": "yas_marina", "yas marina circuit": "yas_marina", "baku": "baku", "baku city circuit": "baku", "monza": "monza", "madrid": "madrid",
    "zandvoort": "zandvoort", "budapest": "hungaroring", "spa-francorchamps": "spa",
    "circuit de spa-francorchamps": "spa", "silverstone": "silverstone", "silverstone circuit": "silverstone",
    "spielberg": "red_bull_ring", "red bull ring": "red_bull_ring", "barcelona": "barcelona",
    "circuit de barcelona-catalunya": "barcelona",
    "monaco": "monaco", "monte carlo": "monaco", "circuit de monaco": "monaco",
    "montréal": "montreal", "montreal": "montreal", "circuit gilles-villeneuve": "montreal",
    "miami": "miami", "miami gardens": "miami", "miami international autodrome": "miami",
    "suzuka": "suzuka", "suzuka international racing course": "suzuka",
    "shanghai": "shanghai", "shanghai international circuit": "shanghai",
    "melbourne": "albert_park", "albert park": "albert_park", "albert park grand prix circuit": "albert_park",
    "sakhir": "bahrain", "bahrain": "bahrain", "bahrain international circuit": "bahrain",
    "jeddah": "jeddah", "jeddah corniche circuit": "jeddah", "lusail international circuit": "lusail",
    "yas marina circuit": "yas_marina", "imola": "imola",
    "autodromo enzo e dino ferrari": "imola", "autodromo enzo e dino ferrari circuit": "imola",
    "le castellet": "paul_ricard", "paul ricard": "paul_ricard",
    "istanbul": "istanbul", "istanbul park": "istanbul", "sochi": "sochi", "sochi autodrom": "sochi",
    "portimão": "portimao", "portimao": "portimao", "algarve international circuit": "portimao",
    "nürburgring": "nurburgring", "nurburgring": "nurburgring",
    "mugello": "mugello", "autodromo internazionale del mugello": "mugello",
    "hockenheim": "hockenheim", "hockenheimring": "hockenheim", "sakhir international circuit": "bahrain",
}


def track_asset_for_circuit(circuit: str | None) -> dict:
    """Resolve a known circuit to a bundled, local SVG without guessing geometry."""
    label = str(circuit or "").strip()
    normalized = re.sub(r"\s+", " ", label.casefold())
    slug = TRACK_ASSET_ALIASES.get(normalized)
    if not slug:
        return {"slug": None, "url": None, "available": False, "label": label or None}
    asset_path = PROJECT_ROOT / "static" / "tracks" / f"{slug}.svg"
    if not asset_path.is_file():
        return {"slug": None, "url": None, "available": False, "label": label or None}
    return {"slug": slug, "url": f"/static/tracks/{slug}.svg", "available": True, "label": label or None}


def circuit_identity_key(circuit: str | None) -> str:
    normalized = re.sub(r"\s+", " ", str(circuit or "").strip().casefold())
    return TRACK_ASSET_ALIASES.get(normalized, normalized)


@lru_cache(maxsize=128)
def _circuit_analysis_cached(circuit: str, before: str, exclude_event_id: str | None) -> dict:
    """Analyze only the six most recent completed circuit events before cutoff."""
    from collections import Counter

    if not circuit or circuit.casefold() == "unknown":
        return _empty_circuit_analysis(circuit, before)
    identity = circuit_identity_key(circuit)
    circuit_names = sorted({str(circuit).strip().casefold()} | {name for name, slug in TRACK_ASSET_ALIASES.items() if slug == identity})
    with connect() as connection:
        exclusion = " AND e.event_id<>?" if exclude_event_id else ""
        names_sql = ",".join("?" for _ in circuit_names)
        params: list[object] = [*circuit_names, before[:10]]
        if exclude_event_id:
            params.append(str(exclude_event_id))
        events = [dict(row) for row in connection.execute(
            "SELECT e.event_id,e.season,e.round,e.race,e.event_date,e.circuit,e.country,e.race_laps "
            f"FROM events e WHERE LOWER(TRIM(e.circuit)) IN ({names_sql}) AND e.event_date<?" + exclusion + " "
            "AND (EXISTS (SELECT 1 FROM laps l WHERE l.event_id=e.event_id) "
            "OR EXISTS (SELECT 1 FROM circuit_history_laps h WHERE h.event_id=e.event_id)) "
            "ORDER BY e.event_date DESC,e.season DESC,e.round DESC LIMIT 6",
            params,
        )]
        if not events:
            return _empty_circuit_analysis(circuit, before)
        ids = [str(event["event_id"]) for event in events]
        marks = ",".join("?" for _ in ids)
        scoped_params = tuple(ids)
        performance = [dict(row) for row in connection.execute(
            "SELECT l.compound,ROUND(AVG(l.tyre_life),2) mean_tyre_age,"
            "ROUND(AVG(l.lap_time),3) mean_lap_time,ROUND(MIN(l.lap_time),3) best_lap_time,"
            "COUNT(*) sample_laps,COUNT(DISTINCT l.event_id) event_count FROM ("
            "SELECT event_id,UPPER(Compound) compound,TyreLife tyre_life,LapTime lap_time FROM laps "
            "UNION ALL SELECT event_id,UPPER(compound),tyre_life,lap_time FROM circuit_history_laps) l "
            f"WHERE l.event_id IN ({marks}) AND l.lap_time>0 AND l.tyre_life IS NOT NULL "
            "AND l.compound IN ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET') "
            "GROUP BY l.compound ORDER BY sample_laps DESC", scoped_params,
        )]
        degradation = [dict(row) for row in connection.execute(
            "WITH source_laps AS (SELECT event_id,Driver driver,UPPER(Compound) compound,TyreLife tyre_life,LapTime lap_time FROM laps "
            "UNION ALL SELECT event_id,driver,UPPER(compound),tyre_life,lap_time FROM circuit_history_laps), "
            "normalized AS (SELECT l.event_id,l.driver,l.compound,CAST(ROUND(l.tyre_life) AS INTEGER) tyre_age,"
            "l.lap_time-AVG(l.lap_time) OVER (PARTITION BY l.event_id,l.driver) delta_seconds "
            f"FROM source_laps l WHERE l.event_id IN ({marks}) AND l.lap_time>0 AND l.tyre_life IS NOT NULL "
            "AND l.compound IN ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET')) "
            "SELECT compound,tyre_age,ROUND(AVG(delta_seconds),4) delta_seconds,COUNT(*) sample_laps,"
            "COUNT(DISTINCT event_id) event_count FROM normalized WHERE tyre_age>0 "
            "GROUP BY compound,tyre_age HAVING COUNT(*)>=4 ORDER BY compound,tyre_age",
            scoped_params,
        )]
        pit_rows = [dict(row) for row in connection.execute(
            "SELECT p.pit_lap,p.race_laps,p.driver,p.previous_tyre,p.new_tyre,e.event_id "
            "FROM pit_transitions p JOIN events e ON e.season=p.season AND e.round=p.round AND e.race=p.race "
            f"WHERE e.event_id IN ({marks}) AND p.pit_lap IS NOT NULL AND p.race_laps>0 "
            "UNION ALL SELECT p.pit_lap,e.race_laps,p.driver,NULL,NULL,p.event_id "
            "FROM circuit_history_pit_stops p JOIN events e ON e.event_id=p.event_id "
            f"WHERE p.event_id IN ({marks}) AND p.pit_lap IS NOT NULL AND e.race_laps>0 "
            "ORDER BY event_id,driver,pit_lap",
            (*scoped_params, *scoped_params),
        )]
        stint_rows = [tuple(row) for row in connection.execute(
            "SELECT event_id,driver,stint,MIN(lap_number),compound FROM ("
            "SELECT event_id,Driver driver,Stint stint,LapNumber lap_number,UPPER(Compound) compound FROM laps "
            "UNION ALL SELECT event_id,driver,stint,lap_number,UPPER(compound) FROM circuit_history_laps) "
            f"WHERE event_id IN ({marks}) AND stint IS NOT NULL "
            "AND compound IN ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET') "
            "GROUP BY event_id,driver,stint,compound ORDER BY event_id,driver,MIN(lap_number)",
            scoped_params,
        )]
        driver_event_rows = [tuple(row) for row in connection.execute(
            "SELECT DISTINCT event_id,driver FROM ("
            "SELECT event_id,Driver driver FROM laps "
            "UNION ALL SELECT event_id,driver FROM circuit_history_laps) "
            f"WHERE event_id IN ({marks}) AND driver IS NOT NULL AND driver!=''",
            scoped_params,
        )]
        coverage_row = connection.execute(
            "SELECT COUNT(*) lap_count,COUNT(DISTINCT CASE WHEN stint IS NOT NULL THEN "
            "event_id||':'||driver||':'||CAST(stint AS TEXT) END) stint_count,"
            "SUM(CASE WHEN compound IN ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET') AND tyre_life IS NOT NULL THEN 1 ELSE 0 END) tyre_observation_count "
            "FROM (SELECT event_id,Driver driver,Stint stint,UPPER(Compound) compound,TyreLife tyre_life FROM laps "
            "UNION ALL SELECT event_id,driver,stint,UPPER(compound),tyre_life FROM circuit_history_laps) "
            f"WHERE event_id IN ({marks})",
            scoped_params,
        ).fetchone()

        event_detail_rows = []
        for event in events:
            event_id = str(event["event_id"])
            event_detail_rows.append({
                "event_id": event_id,
                "lap_count": int(connection.execute(
                    "SELECT (SELECT COUNT(*) FROM laps WHERE event_id=?) + (SELECT COUNT(*) FROM circuit_history_laps WHERE event_id=?)",
                    (event_id, event_id),
                ).fetchone()[0]),
                "pit_count": int(connection.execute(
                    "SELECT (SELECT COUNT(*) FROM pit_transitions p WHERE p.season=? AND p.round=? AND p.race=?) + "
                    "(SELECT COUNT(*) FROM circuit_history_pit_stops p WHERE p.event_id=?)",
                    (event["season"], event["round"], event["race"], event_id),
                ).fetchone()[0]),
                "tyre_count": int(connection.execute(
                    "SELECT (SELECT COUNT(*) FROM laps WHERE event_id=? AND UPPER(Compound) IN ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET') AND TyreLife IS NOT NULL) + "
                    "(SELECT COUNT(*) FROM circuit_history_laps WHERE event_id=? AND UPPER(compound) IN ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET') AND tyre_life IS NOT NULL)",
                    (event_id, event_id),
                ).fetchone()[0]),
                "has_fastf1": bool(connection.execute("SELECT 1 FROM laps WHERE event_id=? LIMIT 1", (event_id,)).fetchone()),
            })

    sequences: Counter[str] = Counter()
    current_key = None
    current_sequence: list[str] = []
    for event_id, driver, _stint, _first_lap, compound in stint_rows:
        key = (event_id, driver)
        if current_key is not None and key != current_key:
            if len(current_sequence) >= 2:
                sequences[" → ".join(current_sequence)] += 1
            current_sequence = []
        current_key = key
        if compound:
            current_sequence.append(str(compound).upper())
    if len(current_sequence) >= 2:
        sequences[" → ".join(current_sequence)] += 1

    pit_laps = [int(row["pit_lap"]) for row in pit_rows if row.get("pit_lap") is not None]
    pit_progress = []
    pit_bands: Counter[str] = Counter()
    for row in pit_rows:
        try:
            progress = max(0.0, min(.999, float(row["pit_lap"]) / float(row["race_laps"])))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        pit_progress.append(progress)
        low = int(progress * 10) * 10
        pit_bands[f"{low}–{low + 10}%"] += 1
    total_sequences = sum(sequences.values())
    strategy_distribution_source = "Observed tyre-compound stint sequences"
    if total_sequences:
        strategy_distribution = [
            {"strategy": name, "count": count, "share": round(count / total_sequences, 4), "basis": "compound_sequence"}
            for name, count in sequences.most_common(8)
        ]
    else:
        driver_events = {(str(event_id), str(driver)) for event_id, driver in driver_event_rows}
        race_seasons = {str(row["event_id"]): int(row["season"]) for row in events}
        stop_counts: Counter[tuple[str, str]] = Counter(
            (str(row["event_id"]), str(row["driver"])) for row in pit_rows
        )
        observed_pit_strategies: Counter[int] = Counter()
        for event_id, driver in driver_events:
            if race_seasons.get(event_id, 0) >= 2011:
                observed_pit_strategies[int(stop_counts[(event_id, driver)])] += 1
        total_strategies = sum(observed_pit_strategies.values())
        strategy_distribution = [
            {"strategy": f"{stops} stop" if stops == 1 else f"{stops} stops", "count": count,
             "share": round(count / total_strategies, 4), "basis": "observed_pit_stop_count"}
            for stops, count in sorted(observed_pit_strategies.items())
        ]
        strategy_distribution_source = (
            "Observed pit-stop counts; tyre compound sequences are unavailable in the historical source"
            if strategy_distribution else "No source-backed strategy observations for this circuit sample"
        )
    pit_distribution = [
        {"band": band, "count": count, "share": round(count / len(pit_progress), 4) if pit_progress else 0.0}
        for band, count in sorted(pit_bands.items(), key=lambda pair: int(pair[0].split("–")[0]))
    ]
    event_details = {row["event_id"]: row for row in event_detail_rows}
    event_summaries = []
    for row in events:
        detail = event_details[str(row["event_id"])]
        event_summaries.append({
            "event_id": row["event_id"], "season": int(row["season"]), "round": int(row["round"]),
            "race": row["race"], "event_date": row["event_date"], "circuit": row["circuit"],
            "country": row["country"], "race_laps": row["race_laps"],
            "lap_count": detail["lap_count"], "pit_transition_count": detail["pit_count"],
            "tyre_observation_count": detail["tyre_count"],
            "lap_source": "FastF1" if detail["has_fastf1"] else "Jolpica F1 lap history",
        })
    years = [str(row["season"]) for row in event_summaries]
    coverage = {
        "race_count": len(events),
        "races_requested": 6,
        "lap_count": int(coverage_row["lap_count"] or 0),
        "stint_count": int(coverage_row["stint_count"] or 0),
        "pit_transition_count": len(pit_laps),
        "tyre_observation_count": int(coverage_row["tyre_observation_count"] or 0),
        "earliest_race_date": min((str(row["event_date"]) for row in events), default=None),
        "latest_race_date": max((str(row["event_date"]) for row in events), default=None),
        "race_years": years,
        "message": f"Only {len(events)} circuit races available" if len(events) < 6 else "Six most recent completed circuit races",
    }
    return {
        "circuit": circuit,
        "cutoff_date": before[:10],
        "event_ids_used": ids,
        "races": event_summaries,
        "coverage": coverage,
        "history_label": " • ".join(years) if years else "No completed circuit races in the supplied data before this cutoff",
        "tyre_performance": performance,
        "tyre_degradation": degradation,
        "pit_transition_count": len(pit_laps),
        "median_pit_lap": float(pd.Series(pit_laps).median()) if pit_laps else None,
        "median_pit_progress": float(pd.Series(pit_progress).median()) if pit_progress else None,
        "pit_window_distribution": pit_distribution,
        "common_pit_lap_bands": _lap_bands(pit_laps),
        "common_next_compounds": dict(Counter(str(row["new_tyre"]) for row in pit_rows if row.get("new_tyre")).most_common(5)),
        "strategy_distribution": strategy_distribution,
        "strategy_distribution_source": strategy_distribution_source,
        "tyre_data_source": "Known compound and tyre-age lap observations only; historical records without tyre telemetry are excluded.",
    }


def _empty_circuit_analysis(circuit: str, before: str) -> dict:
    return {
        "circuit": circuit, "cutoff_date": before[:10], "event_ids_used": [], "races": [],
        "coverage": {"race_count": 0, "races_requested": 6, "lap_count": 0, "stint_count": 0,
                     "pit_transition_count": 0, "tyre_observation_count": 0,
                     "earliest_race_date": None, "latest_race_date": None, "race_years": [],
                     "message": "Only 0 completed circuit races available"},
        "history_label": "No completed circuit races in the supplied data before this cutoff",
        "tyre_performance": [], "tyre_degradation": [], "pit_transition_count": 0,
        "median_pit_lap": None, "median_pit_progress": None, "pit_window_distribution": [],
        "common_pit_lap_bands": [], "common_next_compounds": {}, "strategy_distribution": [],
        "strategy_distribution_source": "No source-backed strategy observations for this circuit sample",
        "tyre_data_source": "Known compound and tyre-age lap observations only; historical records without tyre telemetry are excluded.",
    }


def circuit_analysis(circuit: str, *, before: str | None = None, exclude_event_id: str | None = None) -> dict:
    """Cached six-race circuit analysis with strict event-date and target-event exclusion."""
    cutoff = str(before or app_today().isoformat())[:10]
    return _circuit_analysis_cached(str(circuit or "Unknown"), cutoff, str(exclude_event_id) if exclude_event_id else None)


def clear_circuit_analysis_cache() -> None:
    _circuit_analysis_cached.cache_clear()


def tyre_performance_for_circuit(circuit: str, *, before: str | None = None, exclude_event_id: str | None = None) -> list[dict]:
    return circuit_analysis(circuit, before=before, exclude_event_id=exclude_event_id)["tyre_performance"]


def tyre_degradation_for_circuit(circuit: str, *, before: str | None = None, exclude_event_id: str | None = None) -> list[dict]:
    """Return driver-race-normalized degradation from the selected six circuit events."""
    return circuit_analysis(circuit, before=before, exclude_event_id=exclude_event_id)["tyre_degradation"]


def historical_summary(circuit: str, before: str | None = None, exclude_event_id: str | None = None) -> dict:
    analysis = circuit_analysis(circuit, before=before, exclude_event_id=exclude_event_id)
    return {key: analysis[key] for key in (
        "pit_transition_count", "median_pit_lap", "common_pit_lap_bands", "common_next_compounds",
        "pit_window_distribution", "strategy_distribution", "strategy_distribution_source",
        "tyre_data_source", "coverage", "races", "history_label",
    )}


def _lap_bands(laps: list[int]) -> list[dict]:
    if not laps:
        return []
    bands = pd.cut(pd.Series(laps), bins=[0, 15, 25, 35, 50, 100], labels=["1–15", "16–25", "26–35", "36–50", "51+"])
    counts = bands.value_counts(sort=False)
    return [{"band": str(key), "count": int(value)} for key, value in counts.items() if value]


def quality_report() -> dict:
    with connect() as connection:
        totals = {table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in (
            "events", "laps", "event_drivers", "pit_transitions", "weather_observations",
            "circuit_history_laps", "circuit_history_pit_stops",
        )}
        issues = [dict(row) for row in connection.execute("SELECT * FROM data_quality_issues WHERE issue_code <> 'missing_weather_events' ORDER BY issue_count DESC")]
        missing_weather = int(connection.execute(
            "SELECT COUNT(*) FROM events e WHERE EXISTS (SELECT 1 FROM laps l WHERE l.event_id=e.event_id) "
            "AND NOT EXISTS (SELECT 1 FROM weather_observations w WHERE w.event_id=e.event_id)"
        ).fetchone()[0])
        if missing_weather:
            issues.append({"issue_code": "missing_weather_events", "issue_count": missing_weather, "details": "Historical weather coverage is incomplete; missing observations are handled with an explicit fallback or fetched on demand."})
        duplicates = int(connection.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM laps GROUP BY event_id, Driver, LapNumber HAVING COUNT(*) > 1)"
        ).fetchone()[0])
    return {"tables": totals, "issues": issues, "missing_weather_events": missing_weather, "duplicate_event_driver_lap_keys": duplicates, "status": "review required" if issues or duplicates else "clean"}
