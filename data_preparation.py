"""Clean historical laps and derive reusable tyre and pit-stop datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ALL_COMPOUNDS,
    DATA_COVERAGE_PATH,
    EVENT_CATALOG_PATH,
    FALLBACK_DEGRADATION,
    FALLBACK_OFFSET,
    PIT_TRAINING_DATA_PATH,
    PROCESSED_DATA_DIR,
    RAW_LAP_DATA_PATH,
    LEGACY_RAW_LAP_DATA_PATH,
    RACES,
    TYRE_DEGRADATION_PATH,
    TYRE_MODEL_RESULTS_PATH,
    TYRE_PERFORMANCE_PATH,
    CLEAN_LAP_DATA_PATH,
    ensure_project_directories,
)


REQUIRED_COLUMNS = ("Race", "Driver", "Compound", "LapNumber", "TyreLife", "LapTime")
PIT_COLUMNS = (
    "Season",
    "Round",
    "Race",
    "Circuit",
    "Team",
    "Driver",
    "PitLap",
    "CurrentLap",
    "RaceLaps",
    "RaceProgress",
    "PreviousTyre",
    "PreviousTyreAge",
    "NewTyre",
    "AverageLapTimeBefore",
    "AverageLapTimeAfter",
)


def _numeric_lap_time(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric
    try:
        return pd.to_timedelta(values, errors="coerce").dt.total_seconds()
    except (TypeError, ValueError, AttributeError):
        return numeric


def clean_lap_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return normalized, de-duplicated laps without inventing observations."""

    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Lap data is missing required columns: {', '.join(missing)}")

    df = data.copy()
    if "Season" not in df.columns:
        df["Season"] = 2024
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df = df.dropna(subset=["Season"]).copy()
    df["Season"] = df["Season"].astype(int)

    # Accept common names used by public F1 sources and keep their metadata.
    aliases = {
        "Team": ("TeamName", "Constructor", "ConstructorName"),
        "Circuit": ("CircuitName", "Location"),
        "Round": ("RoundNumber",),
        "DriverName": ("FullName", "Name"),
        "DriverNumber": ("Number",),
        "EventDate": ("Date", "RaceDate"),
    }
    for target, candidates in aliases.items():
        if target not in df.columns:
            source = next((column for column in candidates if column in df.columns), None)
            if source:
                df[target] = df[source]
    for column, fallback in (("Team", "Unknown"), ("Circuit", "Unknown"),
                             ("DriverName", "Unknown"), ("DriverNumber", ""),
                             ("Round", 0), ("EventDate", ""), ("RaceLaps", pd.NA)):
        if column not in df.columns:
            df[column] = fallback
    if "EventName" not in df.columns:
        df["EventName"] = df["Race"]
    parsed_dates = pd.to_datetime(df["EventDate"], errors="coerce", utc=True)
    df["EventDate"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna("")
    df["Compound"] = df["Compound"].astype("string").str.strip().str.upper()
    compound_aliases = {
        "INTER": "INTERMEDIATE", "EXTREME WET": "WET", "FULL WET": "WET",
        "SUPERSOFT": "SOFT", "ULTRASOFT": "SOFT", "HYPERSOFT": "SOFT",
        "SUPER SOFT": "SOFT", "ULTRA SOFT": "SOFT",
    }
    df["Compound"] = df["Compound"].replace(compound_aliases)
    df["Race"] = df["Race"].astype("string").str.strip()
    df["Driver"] = df["Driver"].astype("string").str.strip().str.upper()
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    df["LapTime"] = _numeric_lap_time(df["LapTime"])

    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    invalid_text = {"", "NAN", "NONE", "<NA>"}
    for column in ("Race", "Driver", "Compound"):
        df = df[~df[column].astype(str).str.upper().isin(invalid_text)]

    df["Race"] = df["Race"].astype(str)
    df["Driver"] = df["Driver"].astype(str)
    df["Compound"] = df["Compound"].astype(str)
    for column in ("Team", "Circuit", "DriverName", "DriverNumber", "EventName"):
        df[column] = df[column].fillna("").astype(str).str.strip()
    number_values = pd.to_numeric(df["DriverNumber"], errors="coerce")
    whole_numbers = number_values.notna() & number_values.mod(1).eq(0)
    df.loc[whole_numbers, "DriverNumber"] = number_values.loc[whole_numbers].astype("int64").astype(str)
    df.loc[df["DriverNumber"].str.casefold().isin({"nan", "none", "<na>"}), "DriverNumber"] = ""
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce").fillna(0).astype(int)
    df = df[df["Compound"].isin(ALL_COMPOUNDS)]
    df = df[
        df["LapNumber"].between(1, 100)
        & df["TyreLife"].between(1, 100)
        & df["LapTime"].between(50, 150, inclusive="neither")
    ].copy()

    # Add metadata for the supplied baseline dataset, then validate each event
    # against source/observed distance rather than a fixed race calendar.
    for race, race_info in RACES.items():
        mask = df["Race"].eq(race) & df["Season"].eq(2024)
        df.loc[mask & df["Circuit"].isin(["", "Unknown"]), "Circuit"] = race_info["circuit"]
        df.loc[mask & df["EventDate"].eq(""), "EventDate"] = race_info["date"]
        df.loc[mask & df["RaceLaps"].isna(), "RaceLaps"] = race_info["laps"]
    df["RaceLaps"] = pd.to_numeric(df["RaceLaps"], errors="coerce")
    observed_distance = df.groupby(["Season", "Race"])["LapNumber"].transform("max")
    df["RaceLaps"] = df["RaceLaps"].fillna(observed_distance).clip(lower=1, upper=100)
    df = df[df["LapNumber"].le(df["RaceLaps"])]
    df["RaceLaps"] = df["RaceLaps"].astype(int)

    df["LapNumber"] = df["LapNumber"].astype(int)
    df = df.drop_duplicates(subset=["Season", "Race", "Driver", "LapNumber"], keep="first")
    df = df.sort_values(["Season", "Round", "Race", "Driver", "LapNumber"]).reset_index(drop=True)
    return df


def load_lap_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        candidates = [
            CLEAN_LAP_DATA_PATH,
            RAW_LAP_DATA_PATH,
            LEGACY_RAW_LAP_DATA_PATH,
        ]
        source = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    else:
        source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Historical lap dataset not found: {source}")
    return clean_lap_data(pd.read_csv(source, low_memory=False))


def build_pit_training_data(laps: pd.DataFrame) -> pd.DataFrame:
    """Infer observed stint transitions from tyre-age resets and stint IDs."""

    if laps.empty:
        return pd.DataFrame(columns=PIT_COLUMNS)

    group_keys = [column for column in ("Season", "Race", "Driver") if column in laps]
    sort_cols = group_keys + ["LapNumber"]
    df = laps.sort_values(sort_cols).copy()
    groups = df.groupby(group_keys, sort=False)
    df["PreviousTyreAge"] = groups["TyreLife"].shift(1)
    df["PreviousTyre"] = groups["Compound"].shift(1)
    df["PreviousLap"] = groups["LapNumber"].shift(1)
    if "Stint" in df.columns:
        df["PreviousStint"] = groups["Stint"].shift(1)
        stint_changed = df["Stint"].notna() & df["PreviousStint"].notna() & df["Stint"].ne(df["PreviousStint"])
    else:
        stint_changed = pd.Series(False, index=df.index)

    age_reset = df["TyreLife"].le(2) & df["PreviousTyreAge"].gt(5)
    adjacent_laps = (df["LapNumber"] - df["PreviousLap"]).between(1, 2)
    transitions = df[(age_reset | stint_changed) & adjacent_laps].copy()
    if transitions.empty:
        return pd.DataFrame(columns=PIT_COLUMNS)

    records: list[dict[str, Any]] = []
    by_driver_race = {
        key: group.sort_values("LapNumber")
        for key, group in df.groupby(group_keys, sort=False)
    }
    for row in transitions.itertuples(index=False):
        group_values = tuple(getattr(row, key) for key in group_keys)
        history = by_driver_race[group_values]
        before = history[history["LapNumber"] < row.LapNumber]["LapTime"].tail(5)
        after = history[history["LapNumber"] >= row.LapNumber]["LapTime"].head(5)
        records.append(
            {
                "Season": int(row.Season) if "Season" in group_keys else 2024,
                "Round": int(row.Round) if hasattr(row, "Round") else 0,
                "Race": row.Race,
                "Circuit": row.Circuit if hasattr(row, "Circuit") else "Unknown",
                "Team": row.Team if hasattr(row, "Team") else "Unknown",
                "Driver": row.Driver,
                "PitLap": int(row.LapNumber),
                "CurrentLap": int(row.LapNumber),
                "RaceLaps": int(row.RaceLaps) if hasattr(row, "RaceLaps") and pd.notna(row.RaceLaps) else int(history["LapNumber"].max()),
                "RaceProgress": float(row.LapNumber) / float(
                    row.RaceLaps if hasattr(row, "RaceLaps") and pd.notna(row.RaceLaps) else history["LapNumber"].max()
                ),
                "PreviousTyre": row.PreviousTyre,
                "PreviousTyreAge": float(row.PreviousTyreAge),
                "NewTyre": row.Compound,
                "AverageLapTimeBefore": float(before.mean()) if not before.empty else np.nan,
                "AverageLapTimeAfter": float(after.mean()) if not after.empty else np.nan,
            }
        )
    return (
        pd.DataFrame(records, columns=PIT_COLUMNS)
        .dropna(subset=["PreviousTyre", "NewTyre", "PreviousTyreAge"])
        .drop_duplicates(subset=["Season", "Race", "Driver", "PitLap"], keep="first")
        .sort_values(["Season", "Round", "Race", "Driver", "PitLap"])
        .reset_index(drop=True)
    )


def _stint_slope_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    # Grouped sufficient statistics replace one NumPy fit per stint.
    work = frame[keys + ["TyreLife", "SlopeLapTime"]].copy()
    work["TyreLife"] = pd.to_numeric(work["TyreLife"], errors="coerce")
    work["SlopeLapTime"] = pd.to_numeric(work["SlopeLapTime"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=keys + ["TyreLife", "SlopeLapTime"])
    if work.empty:
        return pd.DataFrame(columns=["Compound", "Slope", "R2"])
    work["AgeSquared"] = work["TyreLife"] ** 2
    work["AgeTimesY"] = work["TyreLife"] * work["SlopeLapTime"]
    work["YSquared"] = work["SlopeLapTime"] ** 2
    grouped = work.groupby(keys, dropna=True, sort=False).agg(
        count=("TyreLife", "size"), age_count=("TyreLife", "nunique"),
        min_age=("TyreLife", "min"), max_age=("TyreLife", "max"),
        sum_age=("TyreLife", "sum"), sum_y=("SlopeLapTime", "sum"),
        sum_age2=("AgeSquared", "sum"), sum_agey=("AgeTimesY", "sum"),
        sum_y2=("YSquared", "sum"),
    )
    grouped = grouped[
        grouped["count"].ge(5) & grouped["age_count"].ge(4)
        & (grouped["max_age"] - grouped["min_age"]).ge(3)
    ].copy()
    if grouped.empty:
        return pd.DataFrame(columns=["Compound", "Slope", "R2"])
    count = grouped["count"].astype(float)
    covariance = grouped["sum_agey"] - grouped["sum_age"] * grouped["sum_y"] / count
    age_variance = grouped["sum_age2"] - grouped["sum_age"] ** 2 / count
    y_variance = grouped["sum_y2"] - grouped["sum_y"] ** 2 / count
    grouped["Slope"] = covariance / age_variance.where(age_variance.gt(0))
    grouped["R2"] = (covariance.pow(2) / (age_variance * y_variance).where(y_variance.gt(0))).clip(0.0, 1.0)
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["Slope"])
    return grouped.reset_index()[[keys[-1], "Slope", "R2"]].rename(columns={keys[-1]: "Compound"})


def analyze_tyre_performance(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate normalized tyre pace, degradation and cross-stint spread."""
    summary_columns = ["Compound", "DataPoints", "CompoundOffset", "OffsetSource",
                       "DegradationRate", "DegradationSource", "StintsUsed", "DegradationStd", "R2"]
    empty_summary = pd.DataFrame(columns=summary_columns)
    empty_curve = pd.DataFrame(columns=["Compound", "TyreLife", "LapTime"])
    if laps.empty:
        return empty_summary, empty_curve

    df = laps.copy()
    context_keys = [column for column in ("Season", "Race", "Driver") if column in df]
    df["LapTime"] = pd.to_numeric(df["LapTime"], errors="coerce")
    df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    df = df.dropna(subset=["LapTime", "TyreLife", "Compound"])
    if df.empty:
        return empty_summary, empty_curve
    df["Compound"] = df["Compound"].astype(str).str.upper()
    if context_keys:
        df["DriverRaceMedian"] = df.groupby(context_keys, dropna=False)["LapTime"].transform("median")
    else:
        df["DriverRaceMedian"] = df["LapTime"].median()
    df["NormalizedLapTime"] = df["LapTime"] - df["DriverRaceMedian"]

    if "Stint" in df.columns and df["Stint"].notna().any():
        stint_keys = context_keys + ["Stint", "Compound"]
    else:
        if context_keys:
            df["StintGroup"] = df.groupby(context_keys, dropna=False)["TyreLife"].diff().le(0).groupby(
                [df[column] for column in context_keys], sort=False
            ).cumsum()
        else:
            df["StintGroup"] = df["TyreLife"].diff().le(0).cumsum()
        stint_keys = context_keys + ["StintGroup", "Compound"]

    # Clip only the compound-level tails to reduce traffic-lap influence.
    limits = df.groupby("Compound")["NormalizedLapTime"].quantile([0.10, 0.90]).unstack()
    df["SlopeLapTime"] = df["NormalizedLapTime"].clip(
        lower=df["Compound"].map(limits[0.10]), upper=df["Compound"].map(limits[0.90])
    )
    slopes = _stint_slope_table(df, stint_keys)
    records: list[dict[str, Any]] = []
    for compound in ALL_COMPOUNDS:
        compound_rows = df[df["Compound"].eq(compound)]
        count = int(len(compound_rows))
        learned_offset = float(compound_rows["NormalizedLapTime"].median()) if count >= 20 else np.nan
        if compound in FALLBACK_OFFSET and not np.isfinite(learned_offset):
            offset, offset_source = float(FALLBACK_OFFSET[compound]), "fallback"
        elif np.isfinite(learned_offset):
            offset, offset_source = learned_offset, "historical"
        else:
            offset, offset_source = np.nan, "unavailable"

        compound_slopes = slopes[slopes["Compound"].eq(compound)] if not slopes.empty else pd.DataFrame()
        valid_slopes = compound_slopes[compound_slopes["Slope"].between(0.002, 0.20)] if not compound_slopes.empty else pd.DataFrame()
        if count >= 30 and len(valid_slopes) >= 3:
            degradation = float(valid_slopes["Slope"].median())
            degradation_source = "historical"
            degradation_std = 1.4826 * float((valid_slopes["Slope"] - degradation).abs().median())
        elif compound in FALLBACK_DEGRADATION:
            degradation, degradation_source, degradation_std = float(FALLBACK_DEGRADATION[compound]), "fallback", np.nan
        else:
            degradation, degradation_source, degradation_std = np.nan, "unavailable", np.nan
        records.append({
            "Compound": compound, "DataPoints": count, "CompoundOffset": offset,
            "OffsetSource": offset_source, "DegradationRate": degradation,
            "DegradationSource": degradation_source, "StintsUsed": int(len(valid_slopes)),
            "DegradationStd": degradation_std,
            "R2": float(valid_slopes["R2"].median()) if not valid_slopes.empty else np.nan,
        })

    summary = pd.DataFrame(records)
    curve = (df.groupby(["Compound", "TyreLife"], as_index=False)["NormalizedLapTime"].median()
             .rename(columns={"NormalizedLapTime": "LapTime"}))
    return summary, curve

def prepare_data(raw_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Run all deterministic preprocessing steps and persist their outputs."""

    ensure_project_directories()
    if raw_path is None:
        source = RAW_LAP_DATA_PATH if RAW_LAP_DATA_PATH.exists() else LEGACY_RAW_LAP_DATA_PATH
    else:
        source = raw_path
    laps = load_lap_data(source)
    pit_data = build_pit_training_data(laps)
    tyre_summary, degradation_curve = analyze_tyre_performance(laps)

    laps.to_csv(CLEAN_LAP_DATA_PATH, index=False)
    pit_data.to_csv(PIT_TRAINING_DATA_PATH, index=False)
    tyre_summary.to_csv(TYRE_PERFORMANCE_PATH, index=False)
    degradation_curve.to_csv(TYRE_DEGRADATION_PATH, index=False)
    tyre_summary[["Compound", "DataPoints", "DegradationRate", "R2"]].to_csv(
        TYRE_MODEL_RESULTS_PATH, index=False
    )

    try:
        coverage = json.loads(DATA_COVERAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        coverage = {}
    season_coverage = []
    for season, rows in laps.groupby("Season", sort=True):
        season_coverage.append(
            {
                "season": int(season),
                "status": "current/partial season" if int(season) == pd.Timestamp.now().year else "historical; strategy-ready laps",
                "lap_rows": int(len(rows)),
                "races": int(rows["Race"].nunique()),
                "drivers": int(rows["Driver"].nunique()),
                "circuits": int(rows["Circuit"].nunique()),
            }
        )
    clean_event_keys = set(zip(laps["Season"].astype(int), laps["Race"].astype(str)))
    try:
        source_events = pd.read_csv(EVENT_CATALOG_PATH, low_memory=False)
        source_event_keys = set(zip(source_events["Season"].astype(int), source_events["Race"].astype(str)))
    except Exception:
        source_event_keys = clean_event_keys
    coverage["cleaned_coverage"] = {
        "lap_rows": int(len(laps)),
        "race_count": int(laps[["Season", "Race"]].drop_duplicates().shape[0]),
        "driver_count": int(laps["Driver"].nunique()),
        "team_count": int(laps["Team"].nunique()),
        "available_seasons": sorted(laps["Season"].astype(int).unique().tolist()),
        "season_coverage": season_coverage,
        "source_events_without_strategy_ready_laps": [
            {"season": season, "race": race}
            for season, race in sorted(source_event_keys - clean_event_keys)
        ],
    }
    DATA_COVERAGE_PATH.write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    return {
        "laps": laps,
        "pit_training": pit_data,
        "tyre_performance": tyre_summary,
        "degradation_curve": degradation_curve,
    }


if __name__ == "__main__":
    outputs = prepare_data()
    print(
        f"Prepared {len(outputs['laps']):,} historical laps, "
        f"{len(outputs['pit_training']):,} observed pit transitions."
    )
    print(f"Processed data written to: {PROCESSED_DATA_DIR}")
