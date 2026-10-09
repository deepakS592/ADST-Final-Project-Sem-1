"""Leakage-aware pre-race backtesting over completed historical events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import CLEAN_LAP_DATA_PATH, PROCESSED_DATA_DIR, RAW_LAP_DATA_PATH, LEGACY_RAW_LAP_DATA_PATH
from data_loader import build_data_catalog
from pre_race_analysis import run_pre_race_analysis


def _prior_distance(laps: pd.DataFrame, season: int, race: str, race_date: str) -> int | None:
    dates = pd.to_datetime(laps["EventDate"], errors="coerce")
    prior = laps[
        (pd.to_numeric(laps["Season"], errors="coerce").lt(int(season)) | dates.lt(pd.to_datetime(race_date)))
        & laps["Race"].astype(str).eq(str(race))
    ]
    values = pd.to_numeric(prior.get("RaceLaps"), errors="coerce").dropna()
    return int(values.median()) if not values.empty else None


def _actual_driver_strategy(event_laps: pd.DataFrame, driver: str) -> tuple[list[str], int, float | None, int]:
    rows = event_laps[event_laps["Driver"].astype(str).str.upper().eq(str(driver).upper())].copy()
    if rows.empty:
        return [], 0, None, 0
    rows = rows.sort_values("LapNumber")
    if "Stint" in rows and rows["Stint"].notna().any():
        group = rows.dropna(subset=["Stint"]).groupby("Stint", sort=True)
        sequence = [str(item["Compound"].dropna().mode().iloc[0]) for _, item in group if not item["Compound"].dropna().empty]
    else:
        sequence = rows["Compound"].astype(str).drop_duplicates().tolist()
    lap_times = pd.to_numeric(rows["LapTime"], errors="coerce")
    valid_laps = int(lap_times.notna().sum())
    actual_time = lap_times.sum(min_count=1)
    return sequence, max(0, len(sequence) - 1), float(actual_time) if pd.notna(actual_time) else None, valid_laps


def run_backtest(
    laps: pd.DataFrame | None = None,
    *,
    seasons: list[int] | None = None,
    limit: int = 3,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if laps is None:
        if not CLEAN_LAP_DATA_PATH.exists():
            source = RAW_LAP_DATA_PATH if RAW_LAP_DATA_PATH.exists() else LEGACY_RAW_LAP_DATA_PATH
            laps = pd.read_csv(source)
        else:
            laps = pd.read_csv(CLEAN_LAP_DATA_PATH)
    events, _ = build_data_catalog(laps)
    if events.empty:
        raise ValueError("No completed events are available for pre-race backtesting.")
    events["EventDateParsed"] = pd.to_datetime(events["EventDate"], errors="coerce")
    events = events.dropna(subset=["EventDateParsed"]).sort_values("EventDateParsed")
    if seasons:
        events = events[pd.to_numeric(events["Season"], errors="coerce").isin(seasons)]
    # Use a compact deterministic sample: the latest event in each requested season,
    # then walk backward if the requested number is not met.
    selected = []
    for year, group in events.groupby("Season", sort=True):
        selected.append(group.iloc[-1])
    selected = sorted(selected, key=lambda row: row["EventDateParsed"], reverse=True)[: max(1, int(limit))]
    if len(selected) < max(1, int(limit)):
        chosen_keys = {(int(row["Season"]), str(row["Race"])) for row in selected}
        for _, row in events.sort_values("EventDateParsed", ascending=False).iterrows():
            key = (int(row["Season"]), str(row["Race"]))
            if key in chosen_keys:
                continue
            selected.append(row)
            chosen_keys.add(key)
            if len(selected) >= int(limit):
                break

    records = []
    for event in selected:
        season = int(event["Season"])
        race = str(event["Race"])
        date_value = pd.Timestamp(event["EventDateParsed"]).date().isoformat()
        distance = _prior_distance(laps, season, race, date_value)
        if not distance:
            records.append({"Season": season, "Race": race, "EventDate": date_value, "Status": "skipped", "Reason": "No prior same-event distance estimate; cutoff-safe simulation unavailable."})
            continue
        event_laps = laps[
            pd.to_numeric(laps["Season"], errors="coerce").eq(season)
            & laps["Race"].astype(str).eq(race)
        ]
        drivers = sorted(event_laps["Driver"].dropna().astype(str).unique().tolist())
        if not drivers:
            continue
        cutoff = pd.Timestamp(date_value, tz="UTC")
        prior_mask = pd.to_datetime(laps["EventDate"], utc=True, errors="coerce").lt(cutoff)
        prior_drivers = set(laps.loc[prior_mask, "Driver"].dropna().astype(str).str.upper())
        eligible_drivers = [driver for driver in drivers if driver.upper() in prior_drivers]
        if not eligible_drivers:
            records.append({"Season": season, "Race": race, "EventDate": date_value, "Status": "skipped", "Reason": "No driver in the evaluation event has pre-cutoff history."})
            continue
        # Driver identity is selected only from the pre-cutoff roster. The completed
        # event is read later for evaluation metrics, never for model features.
        driver = eligible_drivers[0]
        target = event.to_dict()
        target.update({
            "Season": season,
            "Race": race,
            "EventDate": date_value,
            "RaceLaps": distance,
            "RaceLapsSource": "prior same-event median; backtest-safe estimate",
            "CutoffTimeUTC": f"{date_value}T00:00:00Z",
            "RaceStartTimeSource": "race start time unavailable; event-date midnight UTC conservative cutoff",
        })
        # Hold the backtest's completed-event starter as an evaluation context but
        # ensure that their completed-event data never enters the training frame.
        result = run_pre_race_analysis(
            target,
            driver,
            laps,
            fetch_weather=False,
            output_dir=(Path(output_dir) if output_dir else PROCESSED_DATA_DIR / "backtesting") / f"{season}_{race.replace(' ', '_')}",
        )["result"]
        actual_sequence, actual_stops, actual_lap_time, actual_laps = _actual_driver_strategy(event_laps, driver)
        predicted_sequence = str(result["expected_strategy"]["strategy"]).split(" → ")
        sequence_matches = sum(a == b for a, b in zip(predicted_sequence, actual_sequence))
        similarity = sequence_matches / max(len(predicted_sequence), len(actual_sequence), 1)
        estimated = float(result["expected_strategy"]["estimated_race_time"])
        full_distance_comparison = actual_lap_time is not None and actual_laps >= int(distance * 0.9)
        records.append({
            "Season": season,
            "Race": race,
            "EventDate": date_value,
            "Driver": driver,
            "Status": "complete",
            "CutoffTimeUTC": result["training_cutoff_time_utc"],
            "TrainingRows": result["training_rows"],
            "EstimatedRecommendedTimeSeconds": estimated,
            "ActualRecordedLapTimeSumSeconds": actual_lap_time,
            "EstimatedRaceDistanceLaps": int(distance),
            "ActualDriverLapsWithTime": actual_laps,
            "TimeDifferenceSeconds": estimated - actual_lap_time if full_distance_comparison else None,
            "RecommendedStrategy": result["expected_strategy"]["strategy"],
            "ActualObservedStrategy": " → ".join(actual_sequence),
            "RecommendedStops": len(predicted_sequence) - 1,
            "ActualObservedStops": actual_stops,
            "StrategySequenceSimilarity": similarity,
            "SelectedEventLapsUsedForTraining": result["selected_event_laps_used_for_training"],
            "WeatherForecastAccuracy": "unavailable; archived forecast not queried in this backtest run",
            "ComparisonNote": (
                "Time difference compares a near-full-distance model estimate (including modeled pit loss) with recorded lap-time sum; it is approximate and excludes a calibrated race-time reconstruction."
                if full_distance_comparison else
                "Time difference omitted because the evaluation driver recorded fewer than 90% of estimated race-distance laps; strategy sequence remains an observed partial-race comparison."
            ),
        })
    output = pd.DataFrame(records)
    destination = Path(output_dir) if output_dir else PROCESSED_DATA_DIR / "backtesting"
    destination.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination / "backtest_report.csv", index=False)
    completed = output[output.get("Status", pd.Series(dtype=str)).eq("complete")] if not output.empty else pd.DataFrame()
    comparable_time_differences = pd.to_numeric(completed.get("TimeDifferenceSeconds", pd.Series(dtype=float)), errors="coerce").dropna()
    summary = {
        "mode": "PRE-RACE BACKTEST",
        "events_requested": int(limit),
        "events_completed": int(len(completed)),
        "events_skipped": int(len(output) - len(completed)),
        "mean_strategy_sequence_similarity": float(completed["StrategySequenceSimilarity"].mean()) if not completed.empty else None,
        "mean_time_difference_seconds_approximate": float(comparable_time_differences.mean()) if not comparable_time_differences.empty else None,
        "weather_prediction_metrics": "unavailable: this backtest uses no archived pre-race forecast snapshots",
        "limitations": "Starting driver is selected from the completed event for evaluation only. Race distance is estimated from earlier same-event history. Recorded lap sums and modeled race-time estimates are not a calibrated elapsed-time comparison.",
        "rows": records,
    }
    (destination / "backtest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cutoff-safe pre-race backtests over completed F1 events.")
    parser.add_argument("--seasons", default=None, help="Comma-separated seasons, for example 2024,2025.")
    parser.add_argument("--limit", type=int, default=2, help="Maximum number of events (default: 2).")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    years = [int(value) for value in args.seasons.split(",")] if args.seasons else None
    result = run_backtest(seasons=years, limit=args.limit, output_dir=args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
