"""Resumable FastF1 race-lap downloader and dynamic season/event catalog."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from config import (
    DATA_COVERAGE_PATH,
    DEFAULT_SEASON_END,
    DEFAULT_SEASON_START,
    DOWNLOAD_STATUS_PATH,
    EVENT_CATALOG_PATH,
    EVENT_METADATA_OVERRIDE_PATH,
    FASTF1_CACHE_DIR,
    LEGACY_RAW_LAP_DATA_PATH,
    RAW_DATA_DIR,
    RAW_LAP_DATA_PATH,
    RACES,
    SEASON_DATA_DIR,
    ensure_project_directories,
)

LOGGER = logging.getLogger(__name__)
RACE_TYPE = "R"
LAP_COLUMNS = [
    "Season", "Round", "Race", "EventName", "OfficialEventName", "Circuit",
    "Country", "Location", "EventDate", "RaceLaps", "Driver", "DriverName",
    "DriverNumber", "Team", "Compound", "LapNumber", "TyreLife", "Stint",
    "LapTime", "IsAccurate", "Deleted", "TrackStatus",
]
STATUS_COLUMNS = ["Season", "Round", "Race", "OfficialEventName", "Circuit", "Country", "Location", "EventDate", "RaceStartTimeUTC", "Status", "Rows", "Reason"]


def parse_seasons(value: str | Iterable[int] | None = None) -> tuple[int, ...]:
    """Parse year lists/ranges such as ``2019-2026`` without extending into the future."""

    if value is None:
        start, end = DEFAULT_SEASON_START, DEFAULT_SEASON_END
    elif isinstance(value, str):
        bits = value.strip().replace(",", "-").split("-")
        bits = [bit.strip() for bit in bits if bit.strip()]
        if len(bits) == 1:
            start = end = int(bits[0])
        elif len(bits) == 2:
            start, end = int(bits[0]), int(bits[1])
        else:
            raise ValueError("Seasons must be a year or a range such as 2019-2026.")
    else:
        values = sorted({int(year) for year in value})
        return tuple(year for year in values if 2018 <= year <= date.today().year)
    if start < 2018 or end < start:
        raise ValueError("FastF1 season range must start in 2018 or later and be in ascending order.")
    return tuple(range(start, min(end, date.today().year) + 1))


def _text(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def _race_session_date(event: pd.Series) -> pd.Timestamp | pd.NaT:
    """Find the scheduled date for the Race session in FastF1's event schedule."""

    for number in range(1, 9):
        session_name = _text(event.get(f"Session{number}"))
        if session_name.casefold() == "race":
            value = event.get(f"Session{number}Date")
            if value is None or pd.isna(value):
                value = event.get(f"Session{number}DateUtc")
            if value is not None and not pd.isna(value):
                return pd.to_datetime(value, errors="coerce")
    for key in ("EventDate", "EventDateUtc"):
        value = event.get(key)
        if value is not None and not pd.isna(value):
            return pd.to_datetime(value, errors="coerce")
    return pd.NaT


def _event_metadata(year: int, event: pd.Series) -> dict[str, Any]:
    event_date = _race_session_date(event)
    local_date = event_date.date().isoformat() if not pd.isna(event_date) else None
    race_start_utc = pd.NaT
    for number in range(1, 9):
        if _text(event.get(f"Session{number}")).casefold() == "race":
            value = event.get(f"Session{number}DateUtc")
            if value is not None and not pd.isna(value):
                race_start_utc = pd.to_datetime(value, utc=True, errors="coerce")
                break
    round_value = pd.to_numeric(event.get("RoundNumber"), errors="coerce")
    round_number = int(round_value) if pd.notna(round_value) else 0
    name = _text(event.get("EventName"), f"Round {round_number}")
    return {
        "Season": int(year),
        "Round": round_number,
        "Race": name,
        "EventName": name,
        "OfficialEventName": _text(event.get("OfficialEventName"), name),
        "Circuit": _text(event.get("Location"), "Unknown"),
        "Country": _text(event.get("Country"), "Unknown"),
        "Location": _text(event.get("Location"), "Unknown"),
        "EventDate": local_date,
        "RaceStartTimeUTC": race_start_utc.isoformat() if not pd.isna(race_start_utc) else None,
    }


def _driver_metadata(session: Any) -> dict[str, dict[str, str]]:
    results = getattr(session, "results", pd.DataFrame())
    if results is None or results.empty:
        return {}
    table = results.copy()
    if "DriverNumber" not in table.columns and table.index.name == "DriverNumber":
        table = table.reset_index()
    if "DriverNumber" not in table.columns:
        return {}
    result: dict[str, dict[str, str]] = {}
    for _, row in table.iterrows():
        number = _text(row.get("DriverNumber"))
        if not number:
            continue
        result[number] = {
            "DriverName": _text(row.get("FullName"), _text(row.get("BroadcastName"))),
            "Team": _text(row.get("TeamName")),
        }
    return result


def _extract_laps(session: Any, meta: dict[str, Any]) -> pd.DataFrame:
    laps = getattr(session, "laps", None)
    if laps is None or laps.empty:
        return pd.DataFrame(columns=LAP_COLUMNS)
    frame = pd.DataFrame(laps).copy()
    if "Driver" not in frame:
        return pd.DataFrame(columns=LAP_COLUMNS)
    info = _driver_metadata(session)
    frame["Driver"] = frame["Driver"].astype("string").str.upper().str.strip()
    if "LapTime" not in frame:
        frame["LapTime"] = pd.NaT
    frame["LapTime"] = pd.to_timedelta(frame["LapTime"], errors="coerce").dt.total_seconds()
    if "DriverNumber" not in frame:
        frame["DriverNumber"] = ""
    frame["DriverNumber"] = frame["DriverNumber"].astype("string").fillna("").str.strip()
    driver_map = {key: value for key, value in info.items()}
    frame["DriverName"] = frame["DriverNumber"].map(lambda key: driver_map.get(str(key), {}).get("DriverName", ""))
    frame["Team"] = frame.get("Team", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["Team"] = frame.apply(
        lambda row: row["Team"] or driver_map.get(str(row["DriverNumber"]), {}).get("Team", ""), axis=1
    )
    for column, default in (("Compound", "UNKNOWN"), ("TyreLife", pd.NA), ("Stint", pd.NA),
                            ("LapNumber", pd.NA), ("IsAccurate", pd.NA), ("Deleted", False),
                            ("TrackStatus", "")):
        if column not in frame:
            frame[column] = default
    frame["RaceLaps"] = pd.to_numeric(frame["LapNumber"], errors="coerce").max()
    for key, value in meta.items():
        frame[key] = value
    return frame.reindex(columns=LAP_COLUMNS)


def _load_fastf1() -> Any:
    try:
        import fastf1
    except ImportError as error:
        raise RuntimeError("FastF1 is required for data updates. Install project dependencies with: python -m pip install -r requirements.txt") from error
    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    return fastf1


def _status_row(meta: dict[str, Any], status: str, rows: int = 0, reason: str = "") -> dict[str, Any]:
    return {
        "Season": meta["Season"], "Round": meta["Round"], "Race": meta["Race"],
        "OfficialEventName": meta["OfficialEventName"], "Circuit": meta.get("Circuit"),
        "Country": meta.get("Country"), "Location": meta.get("Location"), "EventDate": meta["EventDate"],
        "RaceStartTimeUTC": meta.get("RaceStartTimeUTC"),
        "Status": status, "Rows": int(rows), "Reason": reason,
    }


def download_season(
    season: int,
    *,
    force: bool = False,
    today: date | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Download available completed race-session laps for one season, resuming from per-event files."""

    current_date = today or date.today()
    if int(season) > current_date.year:
        return pd.DataFrame(columns=LAP_COLUMNS), []
    fastf1 = _load_fastf1()
    event_cache = SEASON_DATA_DIR / str(int(season))
    event_cache.mkdir(parents=True, exist_ok=True)
    schedule = fastf1.get_event_schedule(int(season), include_testing=False)
    records: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    for _, event in schedule.iterrows():
        round_value = pd.to_numeric(event.get("RoundNumber"), errors="coerce")
        round_number = int(round_value) if pd.notna(round_value) else 0
        if round_number <= 0:
            continue
        meta = _event_metadata(int(season), event)
        if not meta["EventDate"]:
            statuses.append(_status_row(meta, "unavailable", reason="No race-session date in source schedule."))
            continue
        if date.fromisoformat(meta["EventDate"]) > current_date:
            statuses.append(_status_row(meta, "future/unrun", reason="Race date is in the future; no data requested."))
            continue
        cache_file = event_cache / f"round_{round_number:02d}_race_laps.csv"
        if cache_file.exists() and not force:
            try:
                cached = pd.read_csv(cache_file)
                if not cached.empty:
                    records.append(cached)
                    statuses.append(_status_row(meta, "cached", len(cached)))
                    continue
            except Exception as error:
                LOGGER.warning("Ignoring unreadable event cache %s: %s", cache_file, error)
        label = f"{season} Round {round_number}: {meta['Race']}"
        if progress:
            progress(label)
        try:
            session = fastf1.get_session(int(season), round_number, RACE_TYPE)
            session.load(telemetry=False, weather=False, messages=False, laps=True)
            event_laps = _extract_laps(session, meta)
            event_laps = event_laps.dropna(subset=["LapTime", "LapNumber", "Driver", "Compound"])
            if event_laps.empty:
                statuses.append(_status_row(meta, "unavailable", reason="The source returned no usable timed race laps."))
                continue
            event_laps["RaceLaps"] = int(pd.to_numeric(event_laps["LapNumber"], errors="coerce").max())
            event_laps.to_csv(cache_file, index=False, encoding="utf-8")
            records.append(event_laps)
            statuses.append(_status_row(meta, "downloaded", len(event_laps)))
        except Exception as error:
            LOGGER.warning("Could not load %s: %s", label, error)
            statuses.append(_status_row(meta, "unavailable", reason=str(error)[:500]))
    combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=LAP_COLUMNS)
    if not combined.empty:
        combined.to_csv(event_cache / "season_race_laps.csv", index=False, encoding="utf-8")
    return combined, statuses


def _legacy_laps() -> pd.DataFrame:
    if LEGACY_RAW_LAP_DATA_PATH.exists():
        legacy = pd.read_csv(LEGACY_RAW_LAP_DATA_PATH)
        if "Season" not in legacy:
            legacy["Season"] = 2024
        if "Circuit" not in legacy:
            legacy["Circuit"] = legacy["Race"].map({name: info["circuit"] for name, info in RACES.items()}).fillna("Unknown")
        if "Location" not in legacy:
            legacy["Location"] = legacy["Circuit"]
        if "Country" not in legacy:
            legacy["Country"] = "Unknown"
        if "EventDate" not in legacy:
            legacy["EventDate"] = legacy["Race"].map({name: info["date"] for name, info in RACES.items()}).fillna("")
        if "RaceLaps" not in legacy:
            legacy["RaceLaps"] = legacy["Race"].map({name: info["laps"] for name, info in RACES.items()})
        if "Round" not in legacy:
            # The supplied prototype CSV has no authoritative round metadata.
            # It is reconciled to the FastF1 schedule below when available.
            legacy["Round"] = 0
        if "OfficialEventName" not in legacy:
            legacy["OfficialEventName"] = legacy["Race"]
        if "EventName" not in legacy:
            legacy["EventName"] = legacy["Race"]
        return legacy
    return pd.DataFrame()


def _event_key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", _text(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", normalized.casefold())


def _reconcile_legacy_event_metadata(legacy: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Align overlapping prototype rows to source event names and round metadata."""

    if legacy.empty or source.empty:
        return legacy
    source = source.copy()
    if "Season" not in source or "Season" not in legacy:
        return legacy
    for column, fallback in (("Round", 0), ("Circuit", "Unknown"), ("EventName", ""),
                             ("OfficialEventName", ""), ("Country", "Unknown"),
                             ("Location", "Unknown"), ("EventDate", ""), ("RaceLaps", pd.NA)):
        if column not in source:
            source[column] = fallback
        if column not in legacy:
            legacy[column] = fallback
    metadata = source.drop_duplicates(["Season", "Race"]).copy()
    by_name: dict[tuple[int, str], pd.Series] = {}
    by_circuit: dict[tuple[int, str], pd.Series] = {}
    for _, row in metadata.iterrows():
        season = int(pd.to_numeric(row["Season"], errors="coerce"))
        for value in (row.get("Race"), row.get("EventName"), row.get("OfficialEventName")):
            key = _event_key(value)
            if key:
                by_name.setdefault((season, key), row)
        circuit_key = _event_key(row.get("Circuit"))
        if circuit_key:
            by_circuit.setdefault((season, circuit_key), row)
    for index, row in legacy.iterrows():
        season_value = pd.to_numeric(row.get("Season"), errors="coerce")
        if pd.isna(season_value):
            continue
        season = int(season_value)
        metadata_row = by_name.get((season, _event_key(row.get("Race"))))
        if metadata_row is None:
            metadata_row = by_circuit.get((season, _event_key(row.get("Circuit"))))
        if metadata_row is None:
            continue
        for column in ("Race", "Round", "Circuit", "EventName", "OfficialEventName", "Country", "Location", "EventDate", "RaceLaps"):
            value = metadata_row.get(column)
            if value is not None and not pd.isna(value) and str(value).strip():
                legacy.loc[index, column] = value
    return legacy


def build_data_catalog(laps: pd.DataFrame, statuses: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build available event/driver/season metadata only from actual lap rows."""

    if laps.empty:
        events = pd.DataFrame(columns=["Season", "Round", "Race", "OfficialEventName", "Circuit", "Country", "Location", "EventDate", "RaceLaps", "Rows", "Drivers", "Teams"])
    else:
        frame = laps.copy()
        for column, fallback in (("Round", 0), ("OfficialEventName", ""), ("Circuit", "Unknown"),
                                 ("Country", "Unknown"), ("Location", "Unknown"), ("EventDate", ""),
                                 ("Team", ""), ("RaceLaps", pd.NA)):
            if column not in frame:
                frame[column] = fallback
        events = frame.groupby(["Season", "Race"], as_index=False, dropna=False).agg(
            Round=("Round", "max"), OfficialEventName=("OfficialEventName", "first"),
            Circuit=("Circuit", "first"), Country=("Country", "first"), Location=("Location", "first"),
            EventDate=("EventDate", "first"), RaceLaps=("RaceLaps", "max"), Rows=("LapNumber", "count"),
            Drivers=("Driver", "nunique"), Teams=("Team", "nunique"),
        )
        events["RaceLaps"] = pd.to_numeric(events["RaceLaps"], errors="coerce")
        events = events.sort_values(["Season", "Round", "Race"]).reset_index(drop=True)
    available_seasons = sorted(pd.to_numeric(laps.get("Season", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    current_year = date.today().year
    season_summary = []
    for year in available_seasons:
        year_rows = laps[pd.to_numeric(laps["Season"], errors="coerce").eq(year)]
        season_summary.append({
            "season": year,
            "status": "current/partial season" if year == current_year else "historical; source-available races only",
            "lap_rows": int(len(year_rows)),
            "races": int(year_rows["Race"].nunique()),
            "drivers": int(year_rows["Driver"].nunique()),
            "circuits": int(year_rows.get("Circuit", pd.Series(dtype=str)).nunique()),
        })
    coverage = {
        "source": "FastF1 race-session data plus supplied 2024 historical CSV where needed",
        "configured_season_range": [DEFAULT_SEASON_START, min(DEFAULT_SEASON_END, current_year)],
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "available_seasons": available_seasons,
        "season_coverage": season_summary,
        "lap_rows": int(len(laps)),
        "race_count": int(events.shape[0]),
        "driver_count": int(laps["Driver"].nunique()) if not laps.empty else 0,
        "team_count": int(laps["Team"].nunique()) if "Team" in laps and not laps.empty else 0,
        "source_status_counts": statuses["Status"].value_counts().to_dict() if statuses is not None and not statuses.empty else {},
    }
    return events, coverage


def event_info_from_laps(laps: pd.DataFrame, season: int, race: str) -> dict[str, Any]:
    """Return event metadata for a race that is present in lap data."""

    if "Season" not in laps.columns:
        event_laps = laps[laps["Race"].eq(race)]
    else:
        event_laps = laps[pd.to_numeric(laps["Season"], errors="coerce").eq(int(season)) & laps["Race"].eq(race)]
    if event_laps.empty:
        raise ValueError(f"No historical laps are available for {season} {race}.")
    row = event_laps.sort_values([column for column in ("Round", "LapNumber") if column in event_laps]).iloc[0]
    def value(column: str, fallback: Any = None) -> Any:
        item = row.get(column, fallback)
        return fallback if item is None or pd.isna(item) else item
    race_laps = pd.to_numeric(event_laps.get("RaceLaps", event_laps["LapNumber"]), errors="coerce").dropna()
    distance = int(race_laps.max()) if not race_laps.empty else int(pd.to_numeric(event_laps["LapNumber"], errors="coerce").max())
    location = str(value("Location", value("Circuit", "Unknown")))
    return {
        "season": int(season),
        "round": int(value("Round", 0)),
        "race": str(race),
        "official_name": str(value("OfficialEventName", race)),
        "circuit": str(value("Circuit", location)),
        "location": location,
        "country": str(value("Country", "Unknown")),
        "date": str(value("EventDate", "")),
        "laps": distance,
        "latitude": value("Latitude"),
        "longitude": value("Longitude"),
    }


def available_seasons(laps: pd.DataFrame) -> list[int]:
    """Return only seasons represented by actual lap rows."""

    if laps.empty or "Season" not in laps:
        return []
    return sorted(pd.to_numeric(laps["Season"], errors="coerce").dropna().astype(int).unique().tolist())


def events_for_season(laps: pd.DataFrame, season: int) -> pd.DataFrame:
    """Return only event catalog entries with rows in the selected season."""

    events, _ = build_data_catalog(laps)
    if events.empty:
        return events
    selected = events[pd.to_numeric(events["Season"], errors="coerce").eq(int(season))]
    return selected.sort_values(["Round", "Race"]).reset_index(drop=True)


def laps_for_event(laps: pd.DataFrame, season: int, race: str) -> pd.DataFrame:
    """Filter a lap table to one actual season/event combination."""

    if "Season" not in laps:
        return pd.DataFrame()
    mask = pd.to_numeric(laps["Season"], errors="coerce").eq(int(season)) & laps["Race"].astype(str).eq(str(race))
    return laps[mask].copy()


def drivers_for_event(laps: pd.DataFrame, season: int, race: str) -> list[str]:
    """Return driver codes only when those drivers have rows for the event."""

    rows = laps_for_event(laps, season, race)
    if rows.empty:
        return []
    return sorted(rows["Driver"].dropna().astype(str).unique().tolist())


def build_event_calendar(
    laps: pd.DataFrame,
    statuses: pd.DataFrame | None = None,
    *,
    today: date | None = None,
) -> pd.DataFrame:
    """Combine source schedule ledger rows with observed history for pre-race choices.

    Race distance is not included in FastF1's event schedule. For upcoming events we
    expose a prior same-event median as an estimate with its source stated; no lap
    distance is presented as an official schedule fact.
    """

    current_day = today or date.today()
    if statuses is None:
        try:
            statuses = pd.read_csv(DOWNLOAD_STATUS_PATH)
        except Exception:
            statuses = pd.DataFrame()
    if statuses.empty:
        return pd.DataFrame()
    status = statuses.copy()
    if "EventDate" not in status or "Race" not in status:
        return pd.DataFrame()
    status["EventDate"] = pd.to_datetime(status["EventDate"], errors="coerce").dt.date
    status = status[status["EventDate"].notna()].copy()
    # Only today's and future schedule entries belong in the pre-race selector.
    # Older unavailable entries remain visible in the data-quality ledger.
    status = status[status["EventDate"].ge(current_day)]
    if status.empty:
        return pd.DataFrame()

    observed, _ = build_data_catalog(laps)
    records: list[dict[str, Any]] = []
    for row in status.drop_duplicates([c for c in ("Season", "Round") if c in status], keep="last").to_dict("records"):
        race = _text(row.get("Race"))
        season = int(pd.to_numeric(row.get("Season"), errors="coerce"))
        round_number = int(pd.to_numeric(row.get("Round"), errors="coerce") or 0)
        same_event = observed[observed["Race"].astype(str).eq(race)] if not observed.empty else pd.DataFrame()
        event_row = same_event.sort_values("Season").iloc[-1] if not same_event.empty else pd.Series(dtype=object)
        race_day = row["EventDate"]
        previous_rows = laps.copy()
        if "EventDate" in previous_rows:
            dates = pd.to_datetime(previous_rows["EventDate"], errors="coerce").dt.date
            previous_rows = previous_rows[dates < race_day]
        if "Race" in previous_rows:
            same_race_history = previous_rows[previous_rows["Race"].astype(str).eq(race)]
        else:
            same_race_history = pd.DataFrame()
        distance_values = pd.to_numeric(same_race_history.get("RaceLaps"), errors="coerce").dropna() if "RaceLaps" in same_race_history else pd.Series(dtype=float)
        distance = int(distance_values.median()) if not distance_values.empty else None
        scheduled_start = pd.to_datetime(row.get("RaceStartTimeUTC"), utc=True, errors="coerce")
        actual_start = scheduled_start.isoformat() if pd.notna(scheduled_start) else None
        safe_cutoff = f"{race_day.isoformat()}T00:00:00Z"
        records.append({
            "Season": season,
            "Round": round_number,
            "Race": race,
            "OfficialEventName": _text(row.get("OfficialEventName"), race),
            "Circuit": _text(row.get("Circuit"), _text(event_row.get("Circuit"), "Unknown")),
            "Country": _text(row.get("Country"), _text(event_row.get("Country"), "Unknown")),
            "Location": _text(row.get("Location"), _text(event_row.get("Location"), _text(event_row.get("Circuit"), "Unknown"))),
            "EventDate": race_day.isoformat(),
            "RaceLaps": distance,
            "RaceLapsSource": "prior same-event median; estimate" if distance else "unavailable",
            "CalendarStatus": _text(row.get("Status"), "schedule entry"),
            "EventStatus": "CURRENT" if race_day == current_day else "UPCOMING" if race_day > current_day else "COMPLETED",
            "RaceStartTimeUTC": actual_start,
            "CutoffTimeUTC": actual_start or safe_cutoff,
            "DataCutoffTimeUTC": safe_cutoff,
            "RaceStartTimeSource": "FastF1 scheduled Race session UTC time; lap data is date-only, so training stops at midnight UTC on the race date" if actual_start else "Race start time unavailable; lap data is date-only, so midnight UTC on race date is used",
            "CalendarSource": "FastF1 event schedule plus observed lap history",
        })
    if not records:
        return pd.DataFrame()
    calendar = pd.DataFrame(records)
    override_path = Path(EVENT_METADATA_OVERRIDE_PATH)
    if override_path.exists():
        try:
            overrides = pd.read_csv(override_path, dtype=str).fillna("")
        except Exception:
            overrides = pd.DataFrame()
        if not overrides.empty and {"Season", "Round"}.issubset(overrides.columns):
            overrides["SeasonKey"] = pd.to_numeric(overrides["Season"], errors="coerce")
            overrides["RoundKey"] = pd.to_numeric(overrides["Round"], errors="coerce")
            for _, override in overrides.iterrows():
                match = calendar[
                    pd.to_numeric(calendar["Season"], errors="coerce").eq(override["SeasonKey"])
                    & pd.to_numeric(calendar["Round"], errors="coerce").eq(override["RoundKey"])
                ]
                if match.empty:
                    continue
                index = match.index[0]
                for column in ("Race", "OfficialEventName", "Circuit", "Country", "Location", "EventDate", "RaceLaps", "RaceLapsSource", "RaceStartTimeUTC", "RaceStartTimeSource", "SourceURL"):
                    value = override.get(column)
                    if value is not None and str(value).strip():
                        if column == "RaceLaps":
                            value = pd.to_numeric(value, errors="coerce")
                            if pd.isna(value):
                                continue
                            value = int(value)
                        calendar.loc[index, column] = value
                calendar.loc[index, "CalendarSource"] = "official event metadata override"
                if calendar.loc[index, "RaceStartTimeUTC"]:
                    calendar.loc[index, "CutoffTimeUTC"] = calendar.loc[index, "RaceStartTimeUTC"]
                    event_day = pd.to_datetime(calendar.loc[index, "EventDate"], errors="coerce")
                    calendar.loc[index, "DataCutoffTimeUTC"] = f"{event_day.date().isoformat()}T00:00:00Z" if pd.notna(event_day) else calendar.loc[index, "DataCutoffTimeUTC"]
    return calendar.sort_values(["Season", "Round"]).reset_index(drop=True)


def drivers_available_before(laps: pd.DataFrame, season: int, cutoff_time: str | pd.Timestamp) -> list[str]:
    """List drivers with usable laps strictly before the requested pre-race cutoff."""

    if laps.empty or "EventDate" not in laps or "Driver" not in laps:
        return []
    cutoff = pd.to_datetime(cutoff_time, utc=True, errors="coerce")
    if pd.isna(cutoff):
        return []
    times = pd.to_datetime(laps["EventDate"], utc=True, errors="coerce")
    prior = laps[times < cutoff].copy()
    if "Season" in prior:
        selected_season = prior[pd.to_numeric(prior["Season"], errors="coerce").eq(int(season))]
        # Use the current season's already observed entries when available.
        if not selected_season.empty:
            prior = selected_season
        else:
            prior = prior[pd.to_numeric(prior["Season"], errors="coerce").lt(int(season))]
    return sorted(prior["Driver"].dropna().astype(str).str.upper().unique().tolist())


def filter_laps_before_cutoff(laps: pd.DataFrame, cutoff_time: str | pd.Timestamp) -> pd.DataFrame:
    """Strictly retain source laps whose event date precedes the cutoff instant."""

    if laps.empty or "EventDate" not in laps:
        return laps.iloc[0:0].copy()
    cutoff = pd.to_datetime(cutoff_time, utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("A valid timezone-aware race cutoff is required.")
    event_times = pd.to_datetime(laps["EventDate"], utc=True, errors="coerce")
    return laps[event_times < cutoff].copy()


def update_historical_data(
    seasons: Iterable[int] | str | None = None,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fetch configured seasons once, keep per-event caches, and rebuild the shared data files."""

    ensure_project_directories()
    years = parse_seasons(seasons)
    status_rows: list[dict[str, Any]] = []
    for season in years:
        if progress:
            progress(f"Checking {season} event schedule")
        try:
            _, rows = download_season(season, force=force, progress=progress)
            status_rows.extend(rows)
        except Exception as error:
            LOGGER.exception("Season %s could not be refreshed", season)
            status_rows.append({
                "Season": int(season), "Round": 0, "Race": "", "OfficialEventName": "",
                "EventDate": "", "Status": "season unavailable", "Rows": 0,
                "Reason": str(error)[:500],
            })
    requested_years = set(years)
    season_files = [
        path for path in SEASON_DATA_DIR.glob("*/round_*_race_laps.csv")
        if path.parent.name.isdigit() and int(path.parent.name) in requested_years
    ]
    partitions: list[pd.DataFrame] = []
    for path in sorted(season_files):
        try:
            partitions.append(pd.read_csv(path))
        except Exception as error:
            LOGGER.warning("Skipping unreadable cached race file %s: %s", path, error)
    source = pd.concat(partitions, ignore_index=True, sort=False) if partitions else pd.DataFrame()
    legacy = _reconcile_legacy_event_metadata(_legacy_laps(), source)
    if not legacy.empty and 2024 in requested_years:
        partitions.append(legacy)
    combined = pd.concat(partitions, ignore_index=True, sort=False) if partitions else pd.DataFrame()
    if combined.empty:
        raise RuntimeError("No historical lap rows are available from FastF1 or the supplied local data.")
    combined.to_csv(RAW_LAP_DATA_PATH, index=False, encoding="utf-8")
    status_frame = pd.DataFrame(status_rows, columns=STATUS_COLUMNS).drop_duplicates(subset=["Season", "Round"], keep="last")
    status_frame.to_csv(DOWNLOAD_STATUS_PATH, index=False, encoding="utf-8")
    events, coverage = build_data_catalog(combined, status_frame)
    events.to_csv(EVENT_CATALOG_PATH, index=False, encoding="utf-8")
    DATA_COVERAGE_PATH.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return {"raw_path": RAW_LAP_DATA_PATH, "rows": len(combined), "events": events, "coverage": coverage, "status": status_frame}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download/refresh available completed F1 race laps from FastF1.")
    parser.add_argument("--seasons", default=None, help="Season year or range (default 2019-2026).")
    parser.add_argument("--force", action="store_true", help="Reload all event sessions, using FastF1's disk cache when possible.")
    parser.add_argument("--status-only", action="store_true", help="Print locally available season/event coverage without downloading.")
    args = parser.parse_args(argv)
    if args.status_only:
        if not RAW_LAP_DATA_PATH.exists() and not LEGACY_RAW_LAP_DATA_PATH.exists():
            print("No local lap dataset is available yet.")
            return 0
        source = RAW_LAP_DATA_PATH if RAW_LAP_DATA_PATH.exists() else LEGACY_RAW_LAP_DATA_PATH
        frame = pd.read_csv(source)
        events, coverage = build_data_catalog(frame)
        print(json.dumps(coverage, indent=2))
        print(events.to_string(index=False))
        return 0
    def announce(message: str) -> None:
        print(message, flush=True)
    result = update_historical_data(args.seasons, force=args.force, progress=announce)
    print(f"Saved {result['rows']:,} source lap rows to {result['raw_path']}")
    print(f"Available years: {', '.join(map(str, result['coverage']['available_seasons']))}")
    print(f"Events with loaded lap data: {result['coverage']['race_count']}")
    print("2026 is labeled as a current/partial season when data is present; future sessions are skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
