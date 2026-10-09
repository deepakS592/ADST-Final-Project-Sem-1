"""Calendar-only update path for scheduled seasons, without inferred entrants."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from config import CALENDAR_UPDATES_PATH
from jolpica_f1 import JolpicaF1Client, circuit_fields, race_date
from services.data_service import sync_calendar_metadata
from services.database import initialize_schema


CALENDAR_COLUMNS = [
    "Season", "Round", "Race", "OfficialEventName", "Circuit", "Country", "Location",
    "EventDate", "RaceLaps", "RaceStartTimeUTC", "SourceURL", "CalendarSource",
]


def _jolpica_rows(year: int, client: JolpicaF1Client) -> list[dict[str, Any]]:
    output = []
    for race in client.fetch_schedule(year):
        day = race_date(race)
        try:
            round_number = int(race.get("round") or 0)
            season = int(race.get("season") or year)
        except (TypeError, ValueError):
            continue
        if day is None or round_number <= 0:
            continue
        circuit, country, location = circuit_fields(race)
        clock = str(race.get("time") or "").strip()
        start_utc = f"{day.isoformat()}T{clock}" if clock.endswith("Z") else None
        name = str(race.get("raceName") or "").strip()
        if not name or not circuit:
            continue
        output.append({
            "Season": season, "Round": round_number, "Race": name,
            "OfficialEventName": name, "Circuit": circuit, "Country": country or "Unknown",
            "Location": location or circuit, "EventDate": day.isoformat(), "RaceLaps": None,
            "RaceStartTimeUTC": start_utc, "SourceURL": race.get("url"),
            "CalendarSource": "Jolpica F1 race calendar",
        })
    return output


def _fastf1_rows(year: int) -> list[dict[str, Any]]:
    """Use FastF1's schedule metadata when available; never load race sessions here."""
    try:
        import fastf1

        schedule = fastf1.get_event_schedule(int(year), include_testing=False)
    except Exception:
        return []
    output = []
    for _, event in schedule.iterrows():
        try:
            round_number = int(pd.to_numeric(event.get("RoundNumber"), errors="coerce"))
        except (TypeError, ValueError, OverflowError):
            continue
        if round_number <= 0:
            continue
        race_name = str(event.get("EventName") or "").strip()
        if not race_name:
            continue
        race_start = None
        event_day = None
        for number in range(1, 9):
            if str(event.get(f"Session{number}") or "").casefold() == "race":
                value = event.get(f"Session{number}DateUtc")
                if value is None or pd.isna(value):
                    value = event.get(f"Session{number}Date")
                if value is not None and not pd.isna(value):
                    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
                    if pd.notna(timestamp):
                        event_day = timestamp.date().isoformat()
                        if str(event.get(f"Session{number}DateUtc") or "").strip():
                            race_start = timestamp.isoformat().replace("+00:00", "Z")
                break
        if not event_day:
            value = event.get("EventDate")
            timestamp = pd.to_datetime(value, utc=True, errors="coerce") if value is not None else pd.NaT
            if pd.isna(timestamp):
                continue
            event_day = timestamp.date().isoformat()
        output.append({
            "Season": int(year), "Round": round_number, "Race": race_name,
            "OfficialEventName": str(event.get("OfficialEventName") or race_name).strip(),
            "Circuit": str(event.get("Location") or "Unknown").strip(),
            "Country": str(event.get("Country") or "Unknown").strip(),
            "Location": str(event.get("Location") or "Unknown").strip(),
            "EventDate": event_day, "RaceLaps": None, "RaceStartTimeUTC": race_start,
            "SourceURL": None, "CalendarSource": "FastF1 event schedule",
        })
    return output


def update_calendar(seasons: list[int], *, client: JolpicaF1Client | None = None) -> dict[str, Any]:
    """Merge current upstream schedule metadata and make future seasons selectable."""
    api = client or JolpicaF1Client()
    existing = pd.DataFrame(columns=CALENDAR_COLUMNS)
    if CALENDAR_UPDATES_PATH.exists():
        try:
            existing = pd.read_csv(CALENDAR_UPDATES_PATH, low_memory=False).reindex(columns=CALENDAR_COLUMNS)
        except (OSError, pd.errors.ParserError, UnicodeError):
            existing = pd.DataFrame(columns=CALENDAR_COLUMNS)
    fetched_by_year: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
    by_key: dict[tuple[int, int], dict[str, Any]] = {
        (int(row["Season"]), int(row["Round"])): row
        for row in existing.to_dict(orient="records")
        if pd.notna(row.get("Season")) and pd.notna(row.get("Round"))
    }
    for year in sorted({int(value) for value in seasons}):
        for row in _jolpica_rows(year, api):
            fetched_by_year.setdefault(year, {})[(row["Season"], row["Round"])] = row
        for row in _fastf1_rows(year):
            # Prefer FastF1's event/session names and UTC session times when both feeds have a row.
            year_rows = fetched_by_year.setdefault(year, {})
            previous = year_rows.get((row["Season"], row["Round"]), {})
            merged = {**previous, **{key: value for key, value in row.items() if value not in (None, "")}}
            year_rows[(row["Season"], row["Round"])] = merged
    refreshed_seasons = [year for year, rows in fetched_by_year.items() if rows]
    for year in refreshed_seasons:
        # A non-empty upstream schedule replaces that season's prior metadata. When
        # both feeds return no rows, retain the last known schedule safely.
        by_key = {key: row for key, row in by_key.items() if key[0] != year}
        by_key.update(fetched_by_year[year])
    merged_calendar = pd.DataFrame(by_key.values(), columns=CALENDAR_COLUMNS)
    if not merged_calendar.empty:
        merged_calendar = merged_calendar.sort_values(["Season", "Round"]).reset_index(drop=True)
    CALENDAR_UPDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_calendar.to_csv(CALENDAR_UPDATES_PATH, index=False, encoding="utf-8")
    initialize_schema()
    inserted = sync_calendar_metadata(refreshed_seasons=refreshed_seasons)
    return {
        "requested_seasons": sorted({int(value) for value in seasons}),
        "source_rows_fetched": sum(len(rows) for rows in fetched_by_year.values()),
        "calendar_rows_saved": len(merged_calendar),
        "upcoming_events_added_to_database": inserted,
        "calendar_path": str(CALENDAR_UPDATES_PATH),
        "available_seasons": sorted({int(row["Season"]) for row in merged_calendar.to_dict("records") if pd.notna(row.get("Season"))}),
    }


def parse_calendar_seasons(value: str | None) -> list[int]:
    """Accept one or more explicit calendar years without a project season constant."""
    if not value:
        from services.data_service import app_today
        return [app_today().year + 1]
    result: set[int] = set()
    for group in value.split(","):
        pieces = [piece.strip() for piece in group.split("-") if piece.strip()]
        if len(pieces) == 1:
            result.add(int(pieces[0]))
        elif len(pieces) == 2:
            start, end = int(pieces[0]), int(pieces[1])
            if end < start:
                raise ValueError("Calendar season ranges must be ascending.")
            result.update(range(start, end + 1))
        else:
            raise ValueError("Use a year, comma-separated years, or an ascending range such as 2027-2029.")
    if any(year < 1950 or year > 2200 for year in result):
        raise ValueError("Calendar years must be between 1950 and 2200.")
    return sorted(result)
