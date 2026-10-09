"""SQLite connections, schema setup, and database-backed lap access."""

from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = Path(os.getenv("F1_DATABASE_PATH", DATABASE_DIR / "f1_strategy.db"))
SCHEMA_PATH = DATABASE_DIR / "schema.sql"


def connect(*, row_factory: bool = True) -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    if row_factory:
        connection.row_factory = sqlite3.Row
    return connection


def initialize_schema() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        # CREATE TABLE IF NOT EXISTS does not update an existing database. Keep
        # the on-disk schema compatible with fields added to the import contract.
        tyre_columns = {
            str(row[1]).lower()
            for row in connection.execute("PRAGMA table_info(tyre_performance)")
        }
        if "degradation_std" not in tyre_columns:
            connection.execute(
                "ALTER TABLE tyre_performance ADD COLUMN degradation_std REAL"
            )
        # Older packaged databases predate the cutoff-version table. Seed their
        # existing race dates with the current import marker so valid bundles
        # remain addressable until a later race is appended or data is rebuilt.
        version_count = int(connection.execute("SELECT COUNT(*) FROM event_data_versions").fetchone()[0])
        lap_count = int(connection.execute("SELECT COUNT(*) FROM laps").fetchone()[0])
        if version_count == 0 and lap_count:
            marker = connection.execute(
                "SELECT value FROM app_metadata WHERE key='imported_at_utc'"
            ).fetchone()
            seed = str(marker[0]) if marker else "unversioned"
            dates = [str(row[0])[:10] for row in connection.execute(
                "SELECT DISTINCT EventDate FROM laps WHERE EventDate IS NOT NULL AND TRIM(EventDate)<>'' ORDER BY EventDate"
            )]
            connection.executemany(
                "INSERT OR IGNORE INTO event_data_versions(event_date,cumulative_version) VALUES (?,?)",
                [(day, seed) for day in dates],
            )


def table_count(table: str) -> int:
    allowed = {
        "events", "laps", "event_drivers", "pit_transitions", "weather_observations",
        "tyre_performance", "tyre_degradation", "circuit_history_laps", "circuit_history_pit_stops",
    }
    if table not in allowed:
        raise ValueError("Unknown database table.")
    with connect() as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def laps_frame(*, before: str | None = None) -> pd.DataFrame:
    query = "SELECT l.* FROM laps AS l"
    params: tuple[object, ...] = ()
    if before:
        query += " WHERE l.EventDate < ?"
        params = (before[:10],)
    query += " ORDER BY l.Season, l.Round, l.Driver, l.LapNumber"
    with connect() as connection:
        return pd.read_sql_query(query, connection, params=params)


def driver_laps_frame(driver: str, *, before: str | None = None) -> pd.DataFrame:
    """Load only one driver's usable pre-cutoff history for a prediction request."""
    query = "SELECT l.* FROM laps AS l WHERE UPPER(l.Driver)=?"
    params: list[object] = [str(driver).strip().upper()]
    if before:
        query += " AND l.EventDate < ?"
        params.append(str(before)[:10])
    query += " ORDER BY l.EventDate, l.Season, l.Round, l.LapNumber"
    with connect() as connection:
        return pd.read_sql_query(query, connection, params=params)


def event_driver(event_id: str, driver: str) -> dict | None:
    """Return one observed event entry, if the driver actually raced there."""
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM event_drivers WHERE event_id=? AND UPPER(driver_code)=?",
            (str(event_id), str(driver).strip().upper()),
        ).fetchone()
    return dict(row) if row else None


def imported_data_version() -> str:
    """Return the import marker used to bind offline artifacts to SQLite data."""
    try:
        with connect() as connection:
            row = connection.execute("SELECT value FROM app_metadata WHERE key='imported_at_utc'").fetchone()
        if row:
            return str(row[0])
    except sqlite3.Error:
        pass
    return "unversioned"


def prediction_data_version(cutoff: str) -> str:
    """Return a precomputed data version for rows strictly before a cutoff.

    Imports refresh cumulative date fingerprints offline. This keeps event
    bundle lookup fast and lets a newly observed race affect only bundles whose
    cutoff occurs after that race.
    """
    try:
        import pandas as pd

        timestamp = pd.to_datetime(cutoff, utc=True, errors="coerce")
        if pd.isna(timestamp):
            return imported_data_version()
        cutoff_day = timestamp.strftime("%Y-%m-%d")
        include_cutoff_day = timestamp.time() != pd.Timestamp(timestamp.date(), tz="UTC").time()
        operator = "<=" if include_cutoff_day else "<"
        with connect() as connection:
            row = connection.execute(
                "SELECT cumulative_version FROM event_data_versions "
                f"WHERE event_date {operator} ? ORDER BY event_date DESC LIMIT 1",
                (cutoff_day,),
            ).fetchone()
        return str(row[0]) if row else "empty-before-cutoff"
    except Exception:
        return imported_data_version()


def event_laps(event_id: str) -> pd.DataFrame:
    with connect() as connection:
        return pd.read_sql_query(
            "SELECT * FROM laps WHERE event_id = ? ORDER BY Driver, LapNumber",
            connection,
            params=(event_id,),
        )


def get_event(event_id: str) -> dict | None:
    with connect() as connection:
        row = connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    return dict(row) if row else None


def get_feature_cache(cache_key: str) -> dict | None:
    """Read one versioned aggregate payload stored in SQLite."""
    with connect() as connection:
        row = connection.execute(
            "SELECT payload_json FROM historical_feature_cache WHERE cache_key=?",
            (str(cache_key),),
        ).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row["payload_json"])
        return value if isinstance(value, dict) else None
    except (TypeError, json.JSONDecodeError):
        return None


def put_feature_cache(cache_key: str, payload: dict) -> None:
    """Persist computed aggregates so future requests skip the full lap scan."""
    safe = json.dumps(payload, ensure_ascii=False, default=_json_default)
    with connect() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO historical_feature_cache(cache_key, created_at, payload_json) VALUES (?, ?, ?)",
            (str(cache_key), datetime.now(timezone.utc).isoformat(), safe),
        )


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
