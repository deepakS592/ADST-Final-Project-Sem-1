"""Targeted offline backfill of completed race laps for under-covered circuits."""

from __future__ import annotations

import pandas as pd
from datetime import date
from pathlib import Path
from typing import Any

from config import HISTORICAL_CIRCUIT_LAPS_PATH, HISTORICAL_CIRCUIT_PIT_STOPS_PATH
from jolpica_f1 import JolpicaError, JolpicaF1Client, circuit_fields, race_date
from services.data_service import _event_id, app_today, circuit_identity_key, import_project_data, sync_circuit_history_data
from services.database import connect, initialize_schema


LAP_COLUMNS = [
    "Season", "Round", "Race", "Circuit", "Country", "Location", "EventDate", "RaceLaps",
    "Driver", "LapNumber", "LapTime", "Position", "Compound", "TyreLife", "Stint", "Source",
]
PIT_COLUMNS = ["Season", "Round", "Race", "Driver", "PitLap", "StopNumber", "DurationSeconds", "Source"]
DATA_SOURCE = "Jolpica F1 API"


def _load_rows(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False).reindex(columns=columns)
    except (OSError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame(columns=columns)


def _write_rows(path: Path, existing: pd.DataFrame, new_rows: list[dict], columns: list[str], key_columns: list[str]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(new_rows, columns=columns)
    merged = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
    if not merged.empty:
        merged = merged.drop_duplicates(key_columns, keep="last")
    merged = merged.reindex(columns=columns)
    merged.to_csv(path, index=False, encoding="utf-8")
    return merged


def _race_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row["season"]), int(row["round"]), str(row["race"])


def _event_id_from_source(row: dict[str, Any]) -> str:
    return _event_id(int(row["season"]), int(row["round"]), str(row["race"]))


def _completed_races(connection) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        "SELECT e.event_id,e.season,e.round,e.race,e.circuit,e.event_date "
        "FROM events e WHERE EXISTS (SELECT 1 FROM laps l WHERE l.event_id=e.event_id) "
        "OR EXISTS (SELECT 1 FROM circuit_history_laps h WHERE h.event_id=e.event_id)"
    )]


def _add_race(
    source_race: dict[str, Any], target_circuit: str, client: JolpicaF1Client,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
    season = int(source_race.get("season") or 0)
    round_number = int(source_race.get("round") or 0)
    day = race_date(source_race)
    if season < 1996 or round_number <= 0 or day is None:
        return None
    race_name = str(source_race.get("raceName") or "").strip()
    source_circuit, country, location = circuit_fields(source_race)
    race_meta, timing_rows = client.fetch_laps(season, round_number)
    if not timing_rows:
        return None
    try:
        stop_rows = client.fetch_pit_stops(season, round_number)
    except JolpicaError:
        if season >= 2011:
            raise
        stop_rows = []
    race_laps = max(int(row["lap_number"]) for row in timing_rows)
    pit_laps_by_driver: dict[str, list[int]] = {}
    for stop in stop_rows:
        pit_laps_by_driver.setdefault(str(stop["driver"]), []).append(int(stop["pit_lap"]))
    for stops in pit_laps_by_driver.values():
        stops.sort()

    event = {
        "season": season, "round": round_number, "race": race_name,
        "event_id": _event_id(season, round_number, race_name), "circuit": source_circuit or target_circuit,
        "country": country or "Unknown", "location": location or source_circuit or target_circuit,
        "event_date": day.isoformat(), "race_laps": race_laps,
        "source_circuit_name": source_circuit,
    }
    lap_output = []
    for row in timing_rows:
        driver = str(row["driver"]).upper()
        lap_number = int(row["lap_number"])
        stint = 1 + sum(1 for stop_lap in pit_laps_by_driver.get(str(row["driver"]), []) if stop_lap < lap_number)
        lap_output.append({
            "Season": season, "Round": round_number, "Race": race_name, "Circuit": event["circuit"],
            "Country": event["country"], "Location": event["location"], "EventDate": day.isoformat(),
            "RaceLaps": race_laps, "Driver": driver, "LapNumber": lap_number, "LapTime": row["lap_time"],
            "Position": row.get("position"), "Compound": "UNKNOWN", "TyreLife": None, "Stint": stint,
            "Source": DATA_SOURCE,
        })
    pit_output = [
        {"Season": season, "Round": round_number, "Race": race_name, "Driver": str(row["driver"]).upper(),
         "PitLap": int(row["pit_lap"]), "StopNumber": row.get("stop_number"),
         "DurationSeconds": row.get("duration_seconds"), "Source": DATA_SOURCE}
        for row in stop_rows
    ]
    return event, lap_output, pit_output


def update_historical_circuit_data(
    *, client: JolpicaF1Client | None = None, asof_date: date | None = None,
    event_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Backfill only source races needed to bring scheduled circuits toward six."""
    initialize_schema()
    import_project_data(force=False)
    asof = asof_date or app_today()
    api = client or JolpicaF1Client()
    # Import existing local backfill before counting coverage, making reruns idempotent.
    sync_circuit_history_data()
    schedule = api.fetch_schedule()
    if not schedule:
        raise RuntimeError("Jolpica returned no race-calendar rows; historical circuit data was not changed.")

    with connect() as connection:
        targets = [dict(row) for row in connection.execute(
            "SELECT event_id,season,round,race,circuit,event_date FROM events "
            "WHERE event_date>=? ORDER BY event_date,season,round", (asof.isoformat(),)
        )]
        if event_ids is not None:
            targets = [row for row in targets if str(row["event_id"]) in event_ids]
        completed = _completed_races(connection)

    existing_laps = _load_rows(HISTORICAL_CIRCUIT_LAPS_PATH, LAP_COLUMNS)
    existing_pits = _load_rows(HISTORICAL_CIRCUIT_PIT_STOPS_PATH, PIT_COLUMNS)
    loaded_ids = {str(row["event_id"]) for row in completed}
    planned_events = list(completed)
    events_added: list[dict[str, Any]] = []
    new_laps: list[dict[str, Any]] = []
    new_pits: list[dict[str, Any]] = []
    target_results: list[dict[str, Any]] = []

    source_by_identity: dict[str, list[dict[str, Any]]] = {}
    for race in schedule:
        circuit, _country, _location = circuit_fields(race)
        identity = circuit_identity_key(circuit)
        if identity:
            source_by_identity.setdefault(identity, []).append(race)
    for source_rows in source_by_identity.values():
        source_rows.sort(key=lambda row: (race_date(row) or date.min), reverse=True)

    for target in targets:
        target_day = date.fromisoformat(str(target["event_date"])[:10])
        identity = circuit_identity_key(str(target.get("circuit") or ""))
        if not identity:
            continue
        local_before = [
            row for row in planned_events
            if circuit_identity_key(str(row.get("circuit") or "")) == identity
            and str(row.get("event_date") or "")[:10] < target_day.isoformat()
        ]
        local_ids = {str(row["event_id"]) for row in local_before}
        needed = max(0, 6 - len(local_ids))
        if not needed:
            target_results.append({"event_id": target["event_id"], "circuit": target["circuit"], "races_needed": 0, "races_imported": 0})
            continue
        imported_for_target = 0
        # Race records on or after today are not treated as completed source data.
        candidates = [
            row for row in source_by_identity.get(identity, [])
            if (race_date(row) is not None and race_date(row) < target_day and race_date(row) < asof)
        ]
        for candidate in candidates:
            year, round_number = int(candidate.get("season") or 0), int(candidate.get("round") or 0)
            name = str(candidate.get("raceName") or "").strip()
            candidate_id = _event_id(year, round_number, name)
            if candidate_id in local_ids or candidate_id in loaded_ids:
                continue
            downloaded = _add_race(candidate, str(target["circuit"]), api)
            if downloaded is None:
                continue
            event, lap_rows, pit_rows = downloaded
            event["event_date"] = str(event["event_date"])
            events_added.append(event)
            new_laps.extend(lap_rows)
            new_pits.extend(pit_rows)
            planned_events.append(event)
            local_ids.add(str(event["event_id"]))
            loaded_ids.add(str(event["event_id"]))
            imported_for_target += 1
            if len(local_ids) >= 6:
                break
        target_results.append({
            "event_id": target["event_id"], "circuit": target["circuit"],
            "races_needed": needed, "races_imported": imported_for_target,
            "races_available": len(local_ids),
        })

    # A previous lap-only import may need its separately published pit-stop rows
    # filled in after a source/parser update. Fetch only those selected race events.
    pit_event_keys = {
        (int(row["Season"]), int(row["Round"]), str(row["Race"]))
        for row in existing_pits.to_dict(orient="records")
        if pd.notna(row.get("Season")) and pd.notna(row.get("Round")) and pd.notna(row.get("PitLap"))
    }
    for row in existing_laps.to_dict(orient="records"):
        try:
            key = (int(row["Season"]), int(row["Round"]), str(row["Race"]))
            day = date.fromisoformat(str(row["EventDate"])[:10])
        except (TypeError, ValueError):
            continue
        if key[0] < 2011 or key in pit_event_keys or day >= asof:
            continue
        stops = api.fetch_pit_stops(key[0], key[1])
        for stop in stops:
            new_pits.append({
                "Season": key[0], "Round": key[1], "Race": key[2],
                "Driver": str(stop["driver"]).upper(), "PitLap": int(stop["pit_lap"]),
                "StopNumber": stop.get("stop_number"), "DurationSeconds": stop.get("duration_seconds"),
                "Source": DATA_SOURCE,
            })
        if stops:
            pit_event_keys.add(key)

    if new_laps:
        existing_laps = _write_rows(
            HISTORICAL_CIRCUIT_LAPS_PATH, existing_laps, new_laps, LAP_COLUMNS,
            ["Season", "Round", "Race", "Driver", "LapNumber"],
        )
    elif not HISTORICAL_CIRCUIT_LAPS_PATH.exists():
        _write_rows(HISTORICAL_CIRCUIT_LAPS_PATH, existing_laps, [], LAP_COLUMNS, ["Season", "Round", "Race", "Driver", "LapNumber"])
    if new_pits:
        existing_pits = _write_rows(
            HISTORICAL_CIRCUIT_PIT_STOPS_PATH, existing_pits, new_pits, PIT_COLUMNS,
            ["Season", "Round", "Race", "Driver", "StopNumber", "PitLap"],
        )
    elif not HISTORICAL_CIRCUIT_PIT_STOPS_PATH.exists():
        _write_rows(HISTORICAL_CIRCUIT_PIT_STOPS_PATH, existing_pits, [], PIT_COLUMNS, ["Season", "Round", "Race", "Driver", "StopNumber", "PitLap"])

    imported_counts = sync_circuit_history_data()
    return {
        "asof_date": asof.isoformat(), "targets": target_results,
        "historical_races_added": len(events_added), "historical_laps_added": len(new_laps),
        "historical_pit_stops_added": len(new_pits), "database_import": imported_counts,
        "source": DATA_SOURCE,
    }
