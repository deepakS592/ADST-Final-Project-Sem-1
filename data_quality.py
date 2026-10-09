"""Generate an auditable data quality summary from the local F1 lap tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    CLEAN_LAP_DATA_PATH,
    DATA_QUALITY_CSV_PATH,
    DATA_QUALITY_JSON_PATH,
    DOWNLOAD_STATUS_PATH,
    EVENT_TYRE_NOMINATIONS_PATH,
    TYRE_ALLOCATIONS_PATH,
    RAW_LAP_DATA_PATH,
    LEGACY_RAW_LAP_DATA_PATH,
    WEATHER_ANALYSIS_PATH,
)


def build_data_quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    checks: list[dict[str, Any]] = []

    def add(check: str, count: int, severity: str, detail: str) -> None:
        checks.append({"Check": check, "Count": int(count), "Severity": severity, "Detail": detail})

    add("raw_lap_rows", len(raw), "INFO", "Rows present in the selected local source file.")
    add("clean_lap_rows", len(clean), "INFO", "Rows retained by the standard cleaning pipeline.")
    add("rows_removed_by_cleaning", max(0, len(raw) - len(clean)), "INFO", "Difference between source rows and strategy-ready cleaned rows.")
    expected = ["Season", "Race", "Driver", "Compound", "LapNumber", "LapTime", "TyreLife"]
    missing_columns = [column for column in expected if column not in raw.columns]
    add("missing_required_source_columns", len(missing_columns), "ERROR" if missing_columns else "PASS", ", ".join(missing_columns) if missing_columns else "All expected source fields are present.")
    if not raw.empty:
        duplicate_key = [column for column in ("Season", "Race", "Driver", "LapNumber") if column in raw]
        duplicate_count = int(raw.duplicated(duplicate_key).sum()) if len(duplicate_key) == 4 else 0
        add("duplicate_event_driver_lap_keys", duplicate_count, "REVIEW" if duplicate_count else "PASS", "Duplicate key rows are removed during cleaning.")
        for field in ("Race", "Driver"):
            if field in raw:
                missing_count = int(raw[field].isna().sum() + raw[field].astype(str).str.strip().eq("").sum())
                add(f"missing_{field.casefold()}_rows", missing_count, "REVIEW" if missing_count else "PASS", f"Source rows without a usable {field.lower()} label.")
        for field in ("LapTime", "TyreLife", "Compound", "Driver", "EventDate"):
            if field in raw:
                count = int(raw[field].isna().sum())
                add(f"missing_{field.casefold()}", count, "REVIEW" if count else "PASS", f"Missing values in source field {field}.")
        if "Compound" in raw:
            supported = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
            observed = raw["Compound"].fillna("").astype(str).str.upper().str.strip()
            unknown = raw.loc[~observed.isin(supported) & observed.ne("") , "Compound"].value_counts()
            unknown_total = int(unknown.sum())
            add("unsupported_compound_rows", unknown_total, "REVIEW" if unknown_total else "PASS", "; ".join(f"{name}: {count}" for name, count in unknown.items()) or "No unsupported compound labels found.")
    if not clean.empty and "Race" in clean:
        event_key = [column for column in ("Season", "Race") if column in clean]
        event_count = int(clean[event_key].drop_duplicates().shape[0])
        add("strategy_ready_events", event_count, "INFO", "Distinct season/event pairs with usable cleaned laps.")
        add("clean_compound_count", int(clean.get("Compound", pd.Series(dtype=str)).nunique()), "INFO", "Distinct normalized tyre labels in cleaned data.")

    try:
        statuses = pd.read_csv(DOWNLOAD_STATUS_PATH)
    except Exception:
        statuses = pd.DataFrame()
    incomplete = int(statuses.get("Status", pd.Series(dtype=str)).astype(str).str.contains("unavailable|future|empty", case=False).sum()) if not statuses.empty else 0
    add("incomplete_or_future_schedule_entries", incomplete, "REVIEW" if incomplete else "PASS", "Includes unavailable sessions and future/unrun calendar rows; future rows are expected and are not completed-race data.")
    try:
        weather = pd.read_csv(WEATHER_ANALYSIS_PATH)
    except Exception:
        weather = pd.DataFrame()
    weather_events = int(weather[[column for column in ("Season", "Race") if column in weather]].drop_duplicates().shape[0]) if not weather.empty and any(column in weather for column in ("Season", "Race")) else 0
    add("weather_covered_events_in_saved_analysis", weather_events, "INFO", "Events represented in the latest historical weather output; not all event history is pre-downloaded.")
    try:
        nominations = pd.read_csv(EVENT_TYRE_NOMINATIONS_PATH)
    except Exception:
        nominations = pd.DataFrame()
    verified_nominations = int(nominations.get("Status", pd.Series(dtype=str)).astype(str).str.casefold().isin(["official", "verified", "official range"]).sum()) if not nominations.empty else 0
    add("source_attributed_event_tyre_nominations", verified_nominations, "INFO", "Number of event nomination rows marked official/verified, including source-confirmed ranges without role mapping.")
    scheduled_events = int(statuses[[column for column in ("Season", "Round") if column in statuses]].drop_duplicates().shape[0]) if not statuses.empty and any(column in statuses for column in ("Season", "Round")) else 0
    add("schedule_events_with_verified_nominations", verified_nominations, "INFO", f"Nomination coverage: {verified_nominations}/{scheduled_events} locally recorded schedule entries.")

    frame = pd.DataFrame(checks)
    report = {
        "source": "local FastF1 source CSV, clean output, schedule ledger, weather output, and nomination register",
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "checks": checks,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    return report, frame


def run_data_quality_report(
    raw_path: str | Path | None = None,
    clean_path: str | Path = CLEAN_LAP_DATA_PATH,
) -> dict[str, Any]:
    if raw_path is None:
        raw_path = RAW_LAP_DATA_PATH if RAW_LAP_DATA_PATH.exists() else LEGACY_RAW_LAP_DATA_PATH
    source = Path(raw_path)
    raw = pd.read_csv(source, low_memory=False) if source.exists() else pd.DataFrame()
    clean_file = Path(clean_path)
    clean = pd.read_csv(clean_file) if clean_file.exists() else pd.DataFrame()
    report, table = build_data_quality_report(raw, clean)
    Path(DATA_QUALITY_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(DATA_QUALITY_JSON_PATH).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    table.to_csv(DATA_QUALITY_CSV_PATH, index=False)
    from event_tyre_data import ensure_nomination_template
    ensure_nomination_template()
    allocation_path = Path(TYRE_ALLOCATIONS_PATH)
    if not allocation_path.exists():
        allocation_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["Season", "Round", "Race", "HardSets", "MediumSets", "SoftSets", "IntermediateSets", "WetSets", "SprintWeekend", "Source", "SourceURL", "VerifiedAt", "Status"]).to_csv(allocation_path, index=False)
    return report


if __name__ == "__main__":
    result = run_data_quality_report()
    print(f"Wrote quality checks for {result['raw_rows']:,} raw and {result['clean_rows']:,} cleaned rows.")
    print(DATA_QUALITY_JSON_PATH)
