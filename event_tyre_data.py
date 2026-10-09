"""Load optional, source-attributed event tyre nominations and allocations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import EVENT_TYRE_NOMINATIONS_PATH, TYRE_ALLOCATIONS_PATH, TYRE_SET_LIMITS, WET_TYRE_SET_LIMITS


NOMINATION_COLUMNS = [
    "Season", "Round", "Race", "HardCompound", "MediumCompound", "SoftCompound", "NominatedRange",
    "HardSets", "MediumSets", "SoftSets", "SprintWeekend", "IntermediateAvailable",
    "WetAvailable", "Source", "SourceURL", "VerifiedAt", "Status",
]


def ensure_nomination_template(path: str | Path = EVENT_TYRE_NOMINATIONS_PATH) -> Path:
    """Create an empty editable register when it is missing; never fill guessed values."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        pd.DataFrame(columns=NOMINATION_COLUMNS).to_csv(destination, index=False)
    allocation_path = Path(TYRE_ALLOCATIONS_PATH)
    if not allocation_path.exists():
        allocation_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "Season", "Round", "Race", "HardSets", "MediumSets", "SoftSets",
            "IntermediateSets", "WetSets", "SprintWeekend", "Source", "SourceURL",
            "VerifiedAt", "Status",
        ]).to_csv(allocation_path, index=False)
    return destination


def load_event_nomination(
    season: int,
    race: str,
    round_number: int | None = None,
    *,
    path: str | Path = EVENT_TYRE_NOMINATIONS_PATH,
) -> dict[str, Any]:
    """Return verified source details for an event, or an explicit unavailable record."""

    source_path = ensure_nomination_template(path)
    try:
        frame = pd.read_csv(source_path, dtype=str).fillna("")
    except Exception as error:
        return {"available": False, "status": "unavailable", "season": int(season), "race": str(race), "round": round_number, "allocation": {}, "reason": f"Register could not be read: {error}"}
    if frame.empty:
        return {"available": False, "status": "unavailable", "season": int(season), "race": str(race), "round": round_number, "allocation": {}, "reason": "No official event nominations have been entered into the source-attributed register."}
    frame["SeasonKey"] = pd.to_numeric(frame.get("Season"), errors="coerce")
    frame["RoundKey"] = pd.to_numeric(frame.get("Round"), errors="coerce")
    match = frame[frame["SeasonKey"].eq(int(season)) & frame["Race"].astype(str).str.casefold().eq(str(race).casefold())]
    if match.empty and round_number is not None:
        match = frame[frame["SeasonKey"].eq(int(season)) & frame["RoundKey"].eq(int(round_number))]
    if match.empty:
        return {"available": False, "status": "unavailable", "season": int(season), "race": str(race), "round": round_number, "allocation": {}, "reason": "No event-specific official record is available."}
    row = match.iloc[-1].to_dict()
    status = str(row.get("Status", "")).strip().casefold()
    values = {name: str(row.get(name, "")).strip() or None for name in (
        "HardCompound", "MediumCompound", "SoftCompound", "NominatedRange", "Source", "SourceURL", "VerifiedAt"
    )}
    role_mapping_available = all(values[name] for name in ("HardCompound", "MediumCompound", "SoftCompound"))
    role_mapping_source = "explicit official role mapping" if role_mapping_available else None
    range_match = re.fullmatch(r"\s*C([1-5])\s*-\s*C([1-5])\s*", str(values.get("NominatedRange") or ""), flags=re.IGNORECASE)
    if not role_mapping_available and range_match and status in {"official", "verified", "official range"}:
        first, last = (int(part) for part in range_match.groups())
        if last - first == 2:
            # Compound numbers encode hardness order (C1 hardest). Map this
            # event's source-published consecutive three-compound range by
            # position, never by a season-wide fixed C-number/role assumption.
            values["HardCompound"], values["MediumCompound"], values["SoftCompound"] = (
                f"C{first}", f"C{first + 1}", f"C{last}"
            )
            role_mapping_available = True
            role_mapping_source = "derived from the official consecutive range in compound-hardness order"
    official_compounds = role_mapping_available or bool(values["NominatedRange"])
    source_present = bool(values["Source"] and values["SourceURL"])
    available = official_compounds and source_present and status in {"official", "verified", "official range"}
    allocation = {}
    allocation_source = values["Source"]
    allocation_url = values["SourceURL"]
    allocation_status = status
    allocation_verified_at = values["VerifiedAt"]
    allocation_path = Path(TYRE_ALLOCATIONS_PATH)
    if allocation_path.exists():
        try:
            allocation_frame = pd.read_csv(allocation_path, dtype=str).fillna("")
        except Exception:
            allocation_frame = pd.DataFrame()
        if not allocation_frame.empty:
            allocation_frame["SeasonKey"] = pd.to_numeric(allocation_frame.get("Season"), errors="coerce")
            allocation_frame["RoundKey"] = pd.to_numeric(allocation_frame.get("Round"), errors="coerce")
            allocation_match = allocation_frame[
                allocation_frame["SeasonKey"].eq(int(season))
                & allocation_frame["Race"].astype(str).str.casefold().eq(str(race).casefold())
            ]
            if allocation_match.empty and round_number is not None:
                allocation_match = allocation_frame[
                    allocation_frame["SeasonKey"].eq(int(season)) & allocation_frame["RoundKey"].eq(int(round_number))
                ]
            if not allocation_match.empty:
                row = {**row, **allocation_match.iloc[-1].to_dict()}
                allocation_source = str(row.get("Source", "")).strip() or allocation_source
                allocation_url = str(row.get("SourceURL", "")).strip() or allocation_url
                allocation_status = str(row.get("Status", "")).strip().casefold() or allocation_status
                allocation_verified_at = str(row.get("VerifiedAt", "")).strip() or allocation_verified_at
    for field, compound in (("HardSets", "HARD"), ("MediumSets", "MEDIUM"), ("SoftSets", "SOFT")):
        number = pd.to_numeric(row.get(field), errors="coerce")
        if pd.notna(number) and float(number).is_integer() and int(number) >= 0:
            allocation[compound] = int(number)
    for compound, count in (WET_TYRE_SET_LIMITS.items()):
        wet_field = "IntermediateSets" if compound == "INTERMEDIATE" else "WetSets"
        number = pd.to_numeric(row.get(wet_field), errors="coerce")
        if pd.notna(number) and float(number).is_integer() and int(number) >= 0:
            allocation[compound] = int(number)
    official_allocation_complete = all(compound in allocation for compound in ("HARD", "MEDIUM", "SOFT")) and bool(allocation_source and allocation_url and allocation_status in {"official", "verified"})
    return {
        "available": bool(available),
        "role_mapping_available": bool(role_mapping_available and status in {"official", "verified", "official range"}),
        "role_mapping_source": role_mapping_source,
        "status": "official/verified" if available else (status or "unverified"),
        "season": int(season),
        "race": str(race),
        "round": round_number,
        **values,
        "allocation": allocation,
        "official_allocation_complete": official_allocation_complete,
        "allocation_source": allocation_source,
        "allocation_source_url": allocation_url,
        "allocation_status": allocation_status,
        "allocation_verified_at": allocation_verified_at,
        "sprint_weekend": str(row.get("SprintWeekend", "")).strip().casefold() in {"true", "yes", "1"},
        "source_status": str(row.get("Status", "unavailable")),
        "reason": "Official values require all three nominated compounds, a source URL, and verified status." if not available else "Nomination and allocation are source-attributed in the project register.",
    }


def allocation_for_event(nomination: dict[str, Any]) -> tuple[dict[str, int], str]:
    """Choose verified per-event allocation or a clearly labeled baseline assumption."""

    allocation_values = dict(nomination.get("allocation", {}))
    allocation_verified = bool(
        nomination.get("allocation_source")
        and nomination.get("allocation_source_url")
        and nomination.get("allocation_status") in {"official", "verified"}
    )
    if not allocation_verified and nomination.get("season") is not None:
        allocation_path = Path(TYRE_ALLOCATIONS_PATH)
        try:
            frame = pd.read_csv(allocation_path, dtype=str).fillna("") if allocation_path.exists() else pd.DataFrame()
        except Exception:
            frame = pd.DataFrame()
        if not frame.empty:
            frame["SeasonKey"] = pd.to_numeric(frame.get("Season"), errors="coerce")
            frame["RoundKey"] = pd.to_numeric(frame.get("Round"), errors="coerce")
            matched = frame[
                frame["SeasonKey"].eq(int(nomination["season"]))
                & frame.get("Race", pd.Series(index=frame.index, dtype=str)).astype(str).str.casefold().eq(str(nomination.get("race", "")).casefold())
            ]
            if matched.empty and nomination.get("round") is not None:
                matched = frame[frame["SeasonKey"].eq(int(nomination["season"])) & frame["RoundKey"].eq(int(nomination["round"]))]
            if not matched.empty:
                row = matched.iloc[-1]
                status = str(row.get("Status", "")).strip().casefold()
                source = str(row.get("Source", "")).strip()
                source_url = str(row.get("SourceURL", "")).strip()
                if source and source_url and status in {"official", "verified"}:
                    for field, compound in (("HardSets", "HARD"), ("MediumSets", "MEDIUM"), ("SoftSets", "SOFT"), ("IntermediateSets", "INTERMEDIATE"), ("WetSets", "WET")):
                        value = pd.to_numeric(row.get(field), errors="coerce")
                        if pd.notna(value) and float(value).is_integer() and int(value) >= 0:
                            allocation_values[compound] = int(value)
                    allocation_verified = bool(allocation_values)
    if allocation_verified and allocation_values:
        allocation = {**TYRE_SET_LIMITS, **WET_TYRE_SET_LIMITS, **allocation_values}
        dry_complete = all(compound in allocation_values for compound in TYRE_SET_LIMITS)
        source_note = "official event set counts" if dry_complete else "official partial set counts with model defaults for missing roles"
        return allocation, source_note + "; wet-set defaults remain model assumptions when absent"
    return {**TYRE_SET_LIMITS, **WET_TYRE_SET_LIMITS}, "model assumption: 8/3/2 slick sets and 4/4 wet sets"
