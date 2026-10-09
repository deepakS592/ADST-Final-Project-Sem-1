"""Weather lookup, historical summaries, and SQLite snapshot persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from services.database import connect


def historical_weather(event_id: str) -> dict:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM weather_observations WHERE event_id=? ORDER BY observed_at",
            (event_id,),
        ).fetchall()
    if not rows:
        return {"available": False, "source": "SQLite historical weather", "reason": "No historical weather observations are stored for this event."}
    frame = pd.DataFrame([dict(row) for row in rows])
    rain = pd.to_numeric(frame.get("rain"), errors="coerce").fillna(0)
    precipitation = pd.to_numeric(frame.get("precipitation"), errors="coerce").fillna(0)
    wet_hours = int(((rain > 0) | (precipitation > 0)).sum())
    return {
        "available": True,
        "source": str(frame["source"].dropna().iloc[0]) if frame["source"].notna().any() else "historical weather dataset",
        "observations": int(len(frame)),
        "mean_temperature_c": _mean(frame, "temperature"),
        "total_rain_mm": float(rain.sum()),
        "wet_hours": wet_hours,
        "dominant_condition": str(frame["condition"].mode().iloc[0]) if "condition" in frame and not frame["condition"].dropna().empty else None,
        "hourly": frame.to_dict(orient="records"),
    }


def live_weather(event: dict, *, forecast_days: int = 16, force_refresh: bool = False) -> dict:
    """Retrieve current conditions and event-date forecast with graceful fallback."""
    from weather_api import get_current_and_forecast_weather

    info = {
        "season": event["season"], "round": event["round"], "race": event["race"],
        "official_name": event.get("official_event_name") or event["race"],
        "circuit": event.get("circuit") or "Unknown", "location": event.get("location") or event.get("circuit") or "Unknown",
        "country": event.get("country") or "Unknown", "date": event.get("event_date") or "",
        "laps": event.get("race_laps"), "latitude": None, "longitude": None,
    }
    return get_current_and_forecast_weather(
        event["race"], info, forecast_days=forecast_days, force_refresh=force_refresh
    )


def cached_weather(event_id: str) -> dict:
    """Read only the latest saved weather snapshot; never performs network I/O."""
    package: dict = {
        "current": {"available": False, "reason": "No cached current weather is available."},
        "forecast": {"available": False, "reason": "No cached event forecast is available.", "hourly": []},
        "source": "cached weather snapshot",
        "cache_status": "miss",
    }
    with connect() as connection:
        rows = connection.execute(
            "SELECT kind, retrieved_at, source, payload_json FROM weather_snapshots "
            "WHERE event_id=? AND kind IN ('current','forecast')",
            (str(event_id),),
        ).fetchall()
    if not rows:
        return package
    timestamps = []
    for row in rows:
        try:
            value = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if row["kind"] in package:
            package[row["kind"]] = value
        package["source"] = row["source"] or package["source"]
        timestamps.append(str(row["retrieved_at"]))
    package["retrieved_at_utc"] = max(timestamps) if timestamps else None
    package["cache_status"] = "hit" if timestamps else "miss"
    return package


def historical_weather_for_circuit(circuit: str, *, before: str | None = None) -> dict:
    """Summarize only locally stored weather at this track before the target date."""
    sql = (
        "SELECT COUNT(*) AS observation_count, COUNT(DISTINCT w.event_id) AS event_count, "
        "AVG(w.temperature) AS mean_temperature_c, SUM(COALESCE(w.rain,0)) AS total_rain_mm, "
        "SUM(CASE WHEN COALESCE(w.rain,0)>0 OR COALESCE(w.precipitation,0)>0 THEN 1 ELSE 0 END) AS wet_hours "
        "FROM weather_observations w JOIN events e ON e.event_id=w.event_id "
        "WHERE e.circuit=?"
    )
    params: list[object] = [str(circuit)]
    if before:
        sql += " AND e.event_date < ?"
        params.append(str(before)[:10])
    with connect() as connection:
        row = dict(connection.execute(sql, params).fetchone())
    row["available"] = int(row.get("observation_count") or 0) > 0
    row["source"] = "SQLite historical circuit weather"
    return row


def refresh_historical_weather(event: dict) -> dict:
    """Fetch and persist hourly event-date reanalysis for one selected race."""
    from weather_api import get_historical_weather

    info = {
        "season": event.get("season"), "round": event.get("round"), "race": event.get("race"),
        "circuit": event.get("circuit") or "Unknown", "location": event.get("location") or event.get("circuit") or "Unknown",
        "country": event.get("country") or "Unknown", "date": event.get("event_date") or "",
        "latitude": None, "longitude": None,
    }
    result = get_historical_weather(str(event.get("race", "")), info)
    if result.get("source") != "Open-Meteo historical archive" or not result.get("hourly"):
        return result
    with connect() as connection:
        connection.execute("DELETE FROM weather_observations WHERE event_id=?", (event["event_id"],))
        records = []
        for row in result["hourly"]:
            records.append((
                event["event_id"], event.get("season"), event.get("race"), event.get("circuit"),
                event.get("country"), row.get("time"), row.get("temperature_2m"),
                row.get("precipitation"), row.get("rain"), row.get("precipitation_probability"),
                row.get("wind_speed_10m"), row.get("wind_direction_10m"), row.get("weather_code"),
                row.get("weather_name"), result.get("condition"), result.get("impact"), result.get("source"),
            ))
        connection.executemany(
            "INSERT INTO weather_observations(event_id, season, race, circuit, country, observed_at, temperature, precipitation, rain, precipitation_probability, wind_speed, wind_direction, weather_code, weather, condition, impact, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
    return result


def store_weather(event_id: str, package: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as connection:
        for kind in ("current", "forecast"):
            payload = package.get(kind, {})
            if not payload:
                continue
            connection.execute("DELETE FROM weather_snapshots WHERE event_id=? AND kind=?", (event_id, kind))
            connection.execute(
                "INSERT INTO weather_snapshots(event_id, kind, retrieved_at, source, payload_json) VALUES (?, ?, ?, ?, ?)",
                (event_id, kind, now, package.get("source", "Open-Meteo"), json.dumps(payload, default=_json_default)),
            )


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame.get(column), errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
