"""Constraint-aware candidate generation and deterministic race simulation."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from config import (
    ALL_COMPOUNDS,
    BACKEND_RESULT_PATH,
    CLEAN_LAP_DATA_PATH,
    FALLBACK_DEGRADATION,
    FALLBACK_OFFSET,
    LAP_TIME_MODEL_PATH,
    MIN_STINT,
    PIT_LOSS,
    PIT_STOP_MODEL_PATH,
    PROCESSED_DATA_DIR,
    RAW_LAP_DATA_PATH,
    LEGACY_RAW_LAP_DATA_PATH,
    RACES,
    SLICK_COMPOUNDS,
    STRATEGY_RESULTS_PATH,
    TYRE_MAX_AGE,
    TYRE_MIN_AGE,
    TYRE_SET_LIMITS,
    TYRE_PERFORMANCE_PATH,
    WET_COMPOUNDS,
    WET_TYRE_MAX_AGE,
    WET_TYRE_MIN_AGE,
    WET_TYRE_SET_LIMITS,
    ensure_project_directories,
)
from data_preparation import analyze_tyre_performance, build_pit_training_data, prepare_data
from data_loader import event_info_from_laps, filter_laps_before_cutoff
from train_models import load_model_artifact, train_lap_time_model, train_pit_stop_model, train_models
from weather_api import fallback_weather, get_historical_weather, weather_rows


MODEL_BLEND_WEIGHT = 0.65
WET_FALLBACK_OFFSET = {"INTERMEDIATE": 4.5, "WET": 7.0}


def strategy_feasibility_reasons(
    tyres: Iterable[str], weather_condition: str = "DRY", inventory: dict[str, int] | None = None
) -> list[str]:
    compounds = [str(tyre).strip().upper() for tyre in tyres]
    condition = str(weather_condition).strip().upper()
    reasons: list[str] = []
    if condition not in {"DRY", "DAMP", "WET"}:
        return [f"Unsupported weather condition: {condition or 'empty'}"]
    if len(compounds) not in (2, 3, 4):
        reasons.append("A strategy must contain two, three, or four stints.")
    invalid = sorted(set(compounds) - set(ALL_COMPOUNDS))
    if invalid:
        reasons.append(f"Unsupported compound(s): {', '.join(invalid)}")
        return reasons

    counts = Counter(compounds)
    if condition == "DRY":
        limits = {**TYRE_SET_LIMITS, **(inventory or {})}
        if set(compounds) - set(SLICK_COMPOUNDS):
            reasons.append("A dry strategy can use slick compounds only.")
        if len(set(compounds) & set(SLICK_COMPOUNDS)) < 2:
            reasons.append("A dry race strategy must use at least two different slick compounds.")
    else:
        limits = {**TYRE_SET_LIMITS, **WET_TYRE_SET_LIMITS, **(inventory or {})}
        if not (set(compounds) & set(WET_COMPOUNDS)):
            reasons.append(f"A {condition.lower()} strategy must include an Intermediate or Wet stint.")
        if condition == "WET" and set(compounds) & set(SLICK_COMPOUNDS):
            reasons.append("A wet strategy cannot use slick compounds while the race is classified WET.")

    for compound, count in counts.items():
        limit = limits.get(compound, 0)
        if count > limit:
            reasons.append(
                f"{count} {compound} sets are required, but the baseline allocation allows {limit}."
            )
    return reasons


def strategy_is_feasible(
    tyres: Iterable[str], weather_condition: str = "DRY", inventory: dict[str, int] | None = None
) -> bool:
    """Check physical compound inventory and weather-specific compound rules."""

    return not strategy_feasibility_reasons(tyres, weather_condition, inventory)


@lru_cache(maxsize=8)
def stint_age_bounds(compound: str) -> tuple[int, int]:
    tyre = str(compound).strip().upper()
    if tyre in TYRE_MIN_AGE:
        return max(MIN_STINT, TYRE_MIN_AGE[tyre]), TYRE_MAX_AGE[tyre]
    if tyre in WET_TYRE_MIN_AGE:
        return max(MIN_STINT, WET_TYRE_MIN_AGE[tyre]), WET_TYRE_MAX_AGE[tyre]
    raise ValueError(f"Unsupported tyre compound: {compound}")


def validate_stint_lengths(
    tyres: Iterable[str],
    stint_lengths: Iterable[int],
    race_distance: int,
    weather_condition: str = "DRY",
    inventory: dict[str, int] | None = None,
) -> tuple[bool, list[str]]:
    compounds = [str(tyre).strip().upper() for tyre in tyres]
    lengths_raw = list(stint_lengths)
    reasons = strategy_feasibility_reasons(compounds, weather_condition, inventory)
    if isinstance(race_distance, bool) or not isinstance(race_distance, (int, np.integer)) or race_distance <= 0:
        reasons.append("Race distance must be a positive whole number of laps.")
        return False, reasons
    if len(compounds) != len(lengths_raw):
        reasons.append("Each tyre compound must have one matching stint length.")
        return False, reasons

    lengths: list[int] = []
    for value in lengths_raw:
        try:
            number = float(value)
        except (TypeError, ValueError):
            reasons.append(f"Invalid stint length: {value!r}.")
            continue
        if not np.isfinite(number) or not number.is_integer():
            reasons.append(f"Stint length must be a whole number of laps: {value!r}.")
            continue
        lengths.append(int(number))
    if len(lengths) != len(compounds):
        return False, reasons

    if sum(lengths) != int(race_distance):
        reasons.append("Stint lengths must add up to the race distance.")
    for compound, length in zip(compounds, lengths):
        try:
            minimum, maximum = stint_age_bounds(compound)
        except ValueError as error:
            reasons.append(str(error))
            continue
        if length < minimum:
            reasons.append(
                f"{compound} stint of {length} laps is below the {minimum}-lap minimum."
            )
        if length > maximum:
            reasons.append(
                f"{compound} stint of {length} laps exceeds the {maximum}-lap maximum tyre age."
            )
    return not reasons, reasons


def simulate_race_time(
    tyres: Iterable[str],
    stint_lengths: Iterable[int],
    race_distance: int,
    lap_time_predictor: Callable[[str, int, int], float],
    *,
    weather_condition: str = "DRY",
    pit_loss: float = PIT_LOSS,
    inventory: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Simulate a race using a per-lap predictor and explicit pit-stop loss."""

    compounds = [str(tyre).strip().upper() for tyre in tyres]
    lengths = list(stint_lengths)
    valid, reasons = validate_stint_lengths(
        compounds, lengths, race_distance, weather_condition, inventory
    )
    if not valid:
        raise ValueError("Invalid strategy: " + " ".join(reasons))
    lap_total = 0.0
    pit_laps: list[int] = []
    race_lap = 1
    for compound, stint_length in zip(compounds, lengths):
        for tyre_age in range(1, int(stint_length) + 1):
            predicted = float(lap_time_predictor(compound, tyre_age, race_lap))
            if not np.isfinite(predicted) or predicted <= 0:
                raise ValueError("Lap-time predictor returned a non-positive or non-finite value.")
            lap_total += predicted
            race_lap += 1
        if race_lap <= race_distance:
            pit_laps.append(race_lap - 1)
    stop_cost = float(pit_loss) * len(pit_laps)
    return {
        "estimated_race_time": float(lap_total + stop_cost),
        "predicted_lap_time_total": float(lap_total),
        "pit_stop_cost": stop_cost,
        "pit_laps": pit_laps,
        "stint_lengths": [int(value) for value in lengths],
    }


def generate_compound_sequences(
    weather_condition: str = "DRY", inventory: dict[str, int] | None = None,
    *, include_three_stop: bool = False,
) -> list[tuple[str, ...]]:
    condition = str(weather_condition).strip().upper()
    options = SLICK_COMPOUNDS if condition == "DRY" else ALL_COMPOUNDS
    if condition == "WET":
        options = WET_COMPOUNDS
    sequences: list[tuple[str, ...]] = []
    stint_counts = (2, 3, 4) if include_three_stop else (2, 3)
    for stint_count in stint_counts:
        for sequence in itertools.product(options, repeat=stint_count):
            if strategy_is_feasible(sequence, condition, inventory):
                sequences.append(tuple(sequence))
    return sequences


def _length_grid(lower: int, upper: int, step: int) -> list[int]:
    values = list(range(lower, upper + 1, max(1, int(step))))
    if upper >= lower and (not values or values[-1] != upper):
        values.append(upper)
    return values


def generate_stint_lengths(
    tyres: Iterable[str], race_distance: int, *, resolution: int = 1
) -> Iterable[tuple[int, ...]]:
    compounds = [str(tyre).strip().upper() for tyre in tyres]
    if isinstance(race_distance, bool) or not isinstance(race_distance, (int, np.integer)) or race_distance <= 0:
        return
    bounds = [stint_age_bounds(compound) for compound in compounds]
    if len(compounds) == 2:
        (minimum_a, maximum_a), (minimum_b, maximum_b) = bounds
        for first in range(minimum_a, min(maximum_a, race_distance - minimum_b) + 1):
            second = race_distance - first
            if minimum_b <= second <= maximum_b:
                yield (first, second)
    elif len(compounds) == 3:
        (minimum_a, maximum_a), (minimum_b, maximum_b), (minimum_c, maximum_c) = bounds
        for first in range(minimum_a, min(maximum_a, race_distance - minimum_b - minimum_c) + 1):
            for second in range(minimum_b, min(maximum_b, race_distance - first - minimum_c) + 1):
                third = race_distance - first - second
                if minimum_c <= third <= maximum_c:
                    yield (first, second, third)
    elif len(compounds) == 4:
        bounds = [stint_age_bounds(compound) for compound in compounds]
        (min_a, max_a), (min_b, max_b), (min_c, max_c), (min_d, max_d) = bounds
        first_max = min(max_a, race_distance - min_b - min_c - min_d)
        for first in _length_grid(min_a, first_max, resolution):
            second_max = min(max_b, race_distance - first - min_c - min_d)
            for second in _length_grid(min_b, second_max, resolution):
                third_max = min(max_c, race_distance - first - second - min_d)
                for third in _length_grid(min_c, third_max, resolution):
                    fourth = race_distance - first - second - third
                    if min_d <= fourth <= max_d:
                        yield (first, second, third, fourth)


def _ensure_training_assets() -> tuple[pd.DataFrame, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    ensure_project_directories()
    raw_source = RAW_LAP_DATA_PATH if RAW_LAP_DATA_PATH.exists() else LEGACY_RAW_LAP_DATA_PATH
    raw_exists = raw_source.exists()
    clean_is_stale = not CLEAN_LAP_DATA_PATH.exists() or (
        raw_exists and raw_source.stat().st_mtime > CLEAN_LAP_DATA_PATH.stat().st_mtime
    )
    if clean_is_stale:
        prepare_data(raw_source if raw_exists else None)
    try:
        laps = pd.read_csv(CLEAN_LAP_DATA_PATH)
    except Exception:
        if not raw_exists:
            raise ValueError("Processed lap data is unreadable and the raw dataset is unavailable.")
        prepare_data(raw_source)
        laps = pd.read_csv(CLEAN_LAP_DATA_PATH)
    raw_modified = raw_source.stat().st_mtime if raw_exists else 0.0
    stale = not LAP_TIME_MODEL_PATH.exists() or not PIT_STOP_MODEL_PATH.exists()
    if not stale:
        stale = max(raw_modified, CLEAN_LAP_DATA_PATH.stat().st_mtime) > min(
            LAP_TIME_MODEL_PATH.stat().st_mtime, PIT_STOP_MODEL_PATH.stat().st_mtime
        )
    if stale:
        train_models()
    lap_artifact = load_model_artifact(LAP_TIME_MODEL_PATH)
    pit_artifact = load_model_artifact(PIT_STOP_MODEL_PATH)
    if lap_artifact is None and raw_exists:
        train_models()
        lap_artifact = load_model_artifact(LAP_TIME_MODEL_PATH)
        pit_artifact = load_model_artifact(PIT_STOP_MODEL_PATH)
    metrics_path = PROCESSED_DATA_DIR / "model_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {
            "lap_time": lap_artifact.get("metrics", {}) if lap_artifact else {"available": False},
            "pit_stop": pit_artifact.get("metrics", {}) if pit_artifact else {"available": False},
        }
    if laps.empty:
        raise ValueError("The cleaned historical lap dataset is empty.")
    return laps, lap_artifact, pit_artifact, metrics


def _weather_override(race_name: str, race_info: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    condition = str(override.get("condition", override.get("track_condition", "DRY"))).upper()
    if condition not in {"DRY", "DAMP", "WET"}:
        return fallback_weather(race_name, race_info, f"Unsupported override condition: {condition}")
    impact = str(override.get("impact", override.get("weather_impact", "LOW"))).upper()
    if impact not in {"LOW", "MEDIUM", "HIGH"}:
        impact = "HIGH" if condition == "WET" else "MEDIUM" if condition == "DAMP" else "LOW"
    return {
        "season": race_info.get("season"),
        "race": race_name,
        "circuit": race_info.get("circuit"),
        "country": race_info.get("country"),
        "date": race_info["date"],
        "condition": condition,
        "impact": impact,
        "temperature": override.get("temperature"),
        "precipitation": override.get("precipitation"),
        "rain": override.get("rain"),
        "precipitation_probability": override.get("precipitation_probability"),
        "wind_speed": override.get("wind_speed"),
        "wind_direction": override.get("wind_direction"),
        "dominant_weather": override.get("dominant_weather", "User-selected scenario"),
        "source": override.get("source", "scenario override"),
        "weather_certainty": override.get("weather_certainty", "MEDIUM"),
        "reason": override.get("reason", "Weather condition supplied as a simulation scenario."),
        "hourly": override.get("hourly", []),
    }


def _performance_lookups(
    laps: pd.DataFrame, summary: pd.DataFrame | None = None
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    if summary is None:
        summary, _ = analyze_tyre_performance(laps)
    offsets: dict[str, float] = {}
    degradation: dict[str, float] = {}
    samples: dict[str, int] = {}
    for record in summary.to_dict(orient="records"):
        compound = record["Compound"]
        offset = record.get("CompoundOffset")
        if pd.notna(offset):
            offsets[compound] = float(offset)
        elif compound in FALLBACK_OFFSET:
            offsets[compound] = float(FALLBACK_OFFSET[compound])
        rate = record.get("DegradationRate")
        if pd.notna(rate):
            degradation[compound] = float(rate)
        elif compound in FALLBACK_DEGRADATION:
            degradation[compound] = float(FALLBACK_DEGRADATION[compound])
        else:
            degradation[compound] = 0.05
        samples[compound] = int(record.get("DataPoints", 0))
    return offsets, degradation, samples


def _driver_baseline(
    laps: pd.DataFrame,
    race_name: str,
    driver: str,
    offsets: dict[str, float],
    degradation: dict[str, float],
    season: int | None = None,
    circuit: str | None = None,
) -> tuple[float, pd.DataFrame]:
    mask = (laps["Race"] == race_name) & (laps["Driver"] == driver)
    if season is not None and "Season" in laps:
        mask &= pd.to_numeric(laps["Season"], errors="coerce").eq(int(season))
    history = laps[mask].copy()
    if history.empty and circuit and "Circuit" in laps:
        history = laps[
            laps["Driver"].astype(str).str.upper().eq(str(driver).upper())
            & laps["Circuit"].astype(str).eq(str(circuit))
        ].copy()
    if history.empty:
        history = laps[laps["Driver"].astype(str).str.upper().eq(str(driver).upper())].copy()
    if history.empty:
        raise ValueError(f"No usable historical laps exist for {driver} before this analysis.")
    corrections = history["Compound"].map(offsets).fillna(0.0) + history.apply(
        lambda row: degradation.get(row["Compound"], 0.05) * max(float(row["TyreLife"]) - 1.0, 0.0),
        axis=1,
    )
    baseline = float((history["LapTime"].astype(float) - corrections).median())
    if not np.isfinite(baseline):
        baseline = float(history["LapTime"].median())
    return baseline, history


def _lap_model_features(
    race_name: str,
    driver: str,
    compound: str,
    tyre_ages: np.ndarray,
    race_laps: np.ndarray,
    race_distance: int,
    season: int = 2024,
    circuit: str = "Unknown",
    team: str = "Unknown",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": str(season),
            "Race": race_name,
            "Circuit": circuit,
            "Driver": driver,
            "Team": team,
            "Compound": compound,
            "LapNumber": race_laps.astype(int),
            "TyreLife": tyre_ages.astype(float),
            "RaceProgress": race_laps.astype(float) / float(race_distance),
        }
    )


def _build_prediction_surface(
    *,
    race_name: str,
    driver: str,
    season: int,
    circuit: str,
    team: str,
    race_distance: int,
    laps: pd.DataFrame,
    model_artifact: dict[str, Any] | None,
    weather: dict[str, Any],
    offsets: dict[str, float],
    degradation: dict[str, float],
    samples: dict[str, int],
    baseline: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    surface: dict[str, np.ndarray] = {}
    weather_adjustments: dict[str, np.ndarray] = {}
    prediction_sources: dict[str, str] = {}
    condition = weather["condition"]
    temperature = weather.get("temperature")
    try:
        temperature = float(temperature) if temperature is not None else None
    except (TypeError, ValueError):
        temperature = None
    rain_amount = weather.get("rain")
    try:
        rain_amount = float(rain_amount) if rain_amount is not None else 0.0
    except (TypeError, ValueError):
        rain_amount = 0.0

    for compound in ALL_COMPOUNDS:
        _, maximum_age = stint_age_bounds(compound)
        max_age = min(maximum_age, race_distance)
        ages = np.repeat(np.arange(1, max_age + 1, dtype=int), race_distance)
        race_lap = np.tile(np.arange(1, race_distance + 1, dtype=int), max_age)
        statistical = (
            baseline
            + offsets.get(compound, WET_FALLBACK_OFFSET.get(compound, 0.0))
            + degradation.get(compound, 0.05) * (ages - 1)
        ).astype(float)

        has_observed_compound = samples.get(compound, 0) >= 20
        can_use_model = model_artifact is not None and (compound in SLICK_COMPOUNDS or has_observed_compound)
        if compound == "WET" and samples.get(compound, 0) < 20:
            can_use_model = False
            wet_offset = 5.5 if rain_amount >= 10 else WET_FALLBACK_OFFSET[compound]
            statistical = (
                baseline + wet_offset + degradation.get(compound, 0.05) * (ages - 1)
            ).astype(float)
        elif compound == "INTERMEDIATE" and samples.get(compound, 0) < 20:
            statistical = (
                baseline + WET_FALLBACK_OFFSET[compound] + degradation.get(compound, 0.05) * (ages - 1)
            ).astype(float)
            can_use_model = False

        if can_use_model:
            features = _lap_model_features(
                race_name, driver, compound, ages, race_lap, race_distance, season, circuit, team
            )
            try:
                model_times = np.asarray(model_artifact["model"].predict(features), dtype=float)
                if model_times.shape == statistical.shape and np.isfinite(model_times).all():
                    predictions = MODEL_BLEND_WEIGHT * model_times + (1.0 - MODEL_BLEND_WEIGHT) * statistical
                    prediction_sources[compound] = "Random Forest + historical tyre profile"
                else:
                    predictions = statistical
                    prediction_sources[compound] = "historical tyre profile fallback"
            except Exception:
                predictions = statistical
                prediction_sources[compound] = "historical tyre profile fallback"
        else:
            predictions = statistical
            if compound == "WET" and samples.get(compound, 0) < 20:
                prediction_sources[compound] = "estimated wet-tyre heuristic; no trained WET model"
            elif compound == "INTERMEDIATE" and samples.get(compound, 0) < 20:
                prediction_sources[compound] = "estimated intermediate fallback; insufficient wet laps"
            else:
                prediction_sources[compound] = "historical tyre profile fallback"

        adjustment = np.zeros_like(predictions)
        if condition == "DRY" and temperature is not None:
            heat = max(temperature - 28.0, 0.0)
            cold = max(18.0 - temperature, 0.0)
            adjustment += heat * 0.002 * (ages - 1)
            adjustment += np.where(ages <= 3, cold * 0.025, 0.0)
        elif condition == "DAMP":
            if compound in SLICK_COMPOUNDS:
                drying_proxy = 1.0 - (race_lap / race_distance)
                rain_factor = 1.0 + min(max(rain_amount, 0.0), 5.0) * 0.08
                adjustment += 1.25 * rain_factor * drying_proxy
        elif condition == "WET" and compound == "WET" and rain_amount >= 10:
            # The additional wet-intensity adjustment is a scenario assumption;
            # there are no historical WET laps in this project's supplied data.
            adjustment -= min(rain_amount - 10.0, 20.0) * 0.02

        predictions = np.clip(predictions + adjustment, 50.0, 150.0)
        matrix = np.full((max_age + 1, race_distance + 1), np.nan, dtype=float)
        extra_matrix = np.zeros_like(matrix)
        matrix[ages, race_lap] = predictions
        extra_matrix[ages, race_lap] = adjustment
        surface[compound] = matrix
        weather_adjustments[compound] = extra_matrix
    return surface, weather_adjustments, prediction_sources


def _classifier_probabilities(
    artifact: dict[str, Any] | None,
    race_name: str,
    driver: str,
    previous_tyre: str,
    previous_age: int,
    pit_lap: int,
    season: int = 2024,
    circuit: str = "Unknown",
    team: str = "Unknown",
    race_distance: int = 1,
) -> dict[str, float]:
    if artifact is None:
        return {}
    features = pd.DataFrame(
        {
            "Season": [str(season)],
            "Race": [race_name],
            "Circuit": [circuit],
            "Driver": [driver],
            "Team": [team],
            "PreviousTyre": [previous_tyre],
            "PreviousTyreAge": [float(previous_age)],
            "CurrentLap": [float(pit_lap)],
            "RaceProgress": [float(pit_lap) / max(1, int(race_distance))],
        }
    )
    try:
        model = artifact["model"]
        probabilities = model.predict_proba(features)[0]
        classes = model.named_steps["model"].classes_
        return {str(label): float(value) for label, value in zip(classes, probabilities)}
    except Exception:
        return {}


def _classifier_probabilities_batch(
    artifact: dict[str, Any] | None,
    contexts: Iterable[tuple[Any, ...]],
) -> dict[tuple[Any, ...], dict[str, float]]:
    """Score unique pit contexts in one vectorized model call."""

    unique = list(dict.fromkeys(contexts))
    if artifact is None or not unique:
        return {}
    def unpack(key: tuple[Any, ...]) -> dict[str, Any]:
        race, driver, compound, age, pit_lap = key[:5]
        season, circuit, team, distance = key[5:9] if len(key) >= 9 else (2024, "Unknown", "Unknown", 1)
        return {
            "Season": str(season), "Race": race, "Circuit": circuit, "Driver": driver, "Team": team,
            "PreviousTyre": compound, "PreviousTyreAge": float(age), "CurrentLap": float(pit_lap),
            "RaceProgress": float(pit_lap) / max(1, int(distance)),
        }
    frame = pd.DataFrame([unpack(key) for key in unique])
    try:
        model = artifact["model"]
        probability_matrix = model.predict_proba(frame)
        classes = model.named_steps["model"].classes_
        return {
            key: {str(label): float(value) for label, value in zip(classes, probabilities)}
            for key, probabilities in zip(unique, probability_matrix)
        }
    except Exception:
        return {key: {} for key in unique}


def _simulation_record(
    *,
    race_name: str,
    driver: str,
    season: int,
    circuit: str,
    team: str,
    race_distance: int,
    tyres: tuple[str, ...],
    stint_lengths: tuple[int, ...],
    weather: dict[str, Any],
    surface: dict[str, np.ndarray],
    weather_adjustments: dict[str, np.ndarray],
    degradation: dict[str, float],
    classifier: dict[str, Any] | None,
    classifier_cache: dict[tuple[Any, ...], dict[str, float]],
    inventory: dict[str, int] | None = None,
    historical_strategy_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Keep one full validation at the scoring boundary. Candidate construction
    # already guarantees physical compounds and stint bounds.
    valid, reasons = validate_stint_lengths(
        tyres, stint_lengths, race_distance, weather["condition"], inventory
    )
    if not valid:
        raise ValueError("An invalid stint reached the scoring stage: " + " ".join(reasons))

    # Index the lap-time surfaces with NumPy once per stint instead of calling
    # a Python predictor once per lap for every candidate plan.
    race_lap = 1
    lap_time_total = 0.0
    degradation_impact = 0.0
    weather_adjustment = 0.0
    max_age_use = 0.0
    for compound, length in zip(tyres, stint_lengths):
        ages = np.arange(1, length + 1, dtype=int)
        laps_here = np.arange(race_lap, race_lap + length, dtype=int)
        values = surface[compound][ages, laps_here]
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Lap-time surface returned a non-positive or non-finite estimate.")
        lap_time_total += float(values.sum())
        compound_age_max = stint_age_bounds(compound)[1]
        max_age_use = max(max_age_use, length / compound_age_max)
        degradation_impact += degradation.get(compound, 0.05) * length * (length - 1) / 2.0
        weather_adjustment += float(weather_adjustments[compound][ages, laps_here].sum())
        race_lap += length
    pit_laps = np.cumsum(stint_lengths)[:-1].astype(int).tolist()
    pit_stop_cost = float(PIT_LOSS * len(pit_laps))
    estimated_race_time = lap_time_total + pit_stop_cost

    transition_probabilities: list[float] = []
    all_probabilities: list[dict[str, float]] = []
    for index, pit_lap in enumerate(pit_laps):
        current = tyres[index]
        next_compound = tyres[index + 1]
        key = (race_name, driver, current, int(stint_lengths[index]), int(pit_lap), season, circuit, team, race_distance)
        if classifier is None:
            probabilities = {}
        else:
            if key not in classifier_cache:
                classifier_cache[key] = _classifier_probabilities(
                    classifier, race_name, driver, current, int(stint_lengths[index]), int(pit_lap), season, circuit, team, race_distance
                )
            probabilities = classifier_cache[key]
        all_probabilities.append(probabilities)
        if probabilities:
            transition_probabilities.append(float(probabilities.get(next_compound, 0.0)))
    ml_support = float(np.mean(transition_probabilities)) if transition_probabilities else None

    max_set_use = 0.0
    tyre_usage = dict(Counter(tyres))
    all_limits = {**TYRE_SET_LIMITS, **WET_TYRE_SET_LIMITS, **(inventory or {})}
    for compound, count in tyre_usage.items():
        max_set_use = max(max_set_use, count / all_limits[compound])
    age_penalty = max(0.0, max_age_use - 0.72) ** 2 * 4.0
    inventory_penalty = max(0.0, max_set_use - 0.67) ** 2 * 1.5
    ml_penalty = 0.0 if ml_support is None else 0.35 * (1.0 - ml_support)
    context = historical_strategy_context or {}
    strategy_key = " → ".join(tyres)
    circuit_counts = context.get("strategy_counts_circuit", {})
    global_counts = context.get("strategy_counts_recent_global", {})
    driver_counts = context.get("driver_strategy_counts_circuit", {}).get(str(driver).upper(), {})
    selected_counts = circuit_counts if sum(circuit_counts.values()) >= 6 else global_counts
    total_count = sum(selected_counts.values())
    category_count = max(6, len(selected_counts))
    if total_count:
        support_probability = (float(selected_counts.get(strategy_key, 0)) + 1.0) / (float(total_count) + category_count)
        strongest_probability = (max(selected_counts.values()) + 1.0) / (float(total_count) + category_count)
        if sum(driver_counts.values()) >= 4:
            driver_probability = (float(driver_counts.get(strategy_key, 0)) + 1.0) / (float(sum(driver_counts.values())) + max(6, len(driver_counts)))
            support_probability = 0.85 * support_probability + 0.15 * driver_probability
        historical_strategy_penalty = min(0.25, 0.12 * max(0.0, float(np.log(strongest_probability / max(support_probability, 1e-9)))))
    else:
        support_probability = 0.0
        historical_strategy_penalty = 0.0
    pit_progress = context.get("pit_progress_circuit", [])
    pit_weights = context.get("pit_progress_weights", [])
    historical_pit_penalty = 0.0
    if len(pit_progress) >= 8:
        values = np.asarray(pit_progress, dtype=float)
        weights = np.asarray(pit_weights, dtype=float) if len(pit_weights) == len(values) else np.ones(len(values), dtype=float)
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        cumulative = np.cumsum(weights) / max(float(weights.sum()), 1e-9)
        lower, upper = np.interp([0.25, 0.75], cumulative, values)
        distances = [max(float(lower - lap / race_distance), 0.0, float(lap / race_distance - upper)) for lap in pit_laps]
        historical_pit_penalty = min(0.30, 0.55 * sum(distances))
    unsupported_wet_penalty = (
        0.5 * sum(compound == "WET" for compound in tyres)
        if weather["condition"] in {"DAMP", "WET"}
        else 0.0
    )
    strategy_penalty = float(
        age_penalty + inventory_penalty + ml_penalty + unsupported_wet_penalty
        + historical_strategy_penalty + historical_pit_penalty
    )
    strategy_score = float(estimated_race_time + strategy_penalty)
    score_breakdown = {
        "estimated_tyre_time_seconds": float(lap_time_total),
        "degradation_contribution_seconds": float(degradation_impact),
        "pit_loss_seconds": float(pit_stop_cost),
        "weather_adjustment_seconds": float(weather_adjustment),
        "historical_strategy_penalty_seconds": float(historical_strategy_penalty),
        "historical_pit_window_penalty_seconds": float(historical_pit_penalty),
        "ml_support_penalty_seconds": float(ml_penalty),
        "age_inventory_penalty_seconds": float(age_penalty + inventory_penalty),
        "unsupported_wet_penalty_seconds": float(unsupported_wet_penalty),
        "historical_strategy_support": float(support_probability),
        "total_score_seconds": float(strategy_score),
        "interpretation": "Tyre time already includes fitted degradation and weather adjustments; those terms are shown as diagnostics. Penalties are added to estimated tyre time and pit loss.",
    }
    return {
        "Season": int(season),
        "Race": race_name,
        "Circuit": circuit,
        "Driver": driver,
        "Team": team,
        "StrategyType": f"{len(tyres) - 1}-STOP",
        "Tyres": tuple(tyres),
        "Strategy": " → ".join(tyres),
        "StintLengths": tuple(int(value) for value in stint_lengths),
        "PitLaps": tuple(pit_laps),
        "FirstTyre": tyres[0],
        "SecondTyre": tyres[1],
        "ThirdTyre": tyres[2] if len(tyres) == 3 else None,
        "FourthTyre": tyres[3] if len(tyres) == 4 else None,
        "PitLap1": pit_laps[0] if pit_laps else None,
        "PitLap2": pit_laps[1] if len(pit_laps) > 1 else None,
        "PitLap3": pit_laps[2] if len(pit_laps) > 2 else None,
        "EstimatedRaceTime": float(estimated_race_time),
        "PredictedLapTimeTotal": float(lap_time_total),
        "PitStopCost": pit_stop_cost,
        "DegradationImpact": float(degradation_impact),
        "WeatherAdjustment": float(weather_adjustment),
        "StrategyPenalty": strategy_penalty,
        "StrategyScore": strategy_score,
        "ScoreBreakdown": score_breakdown,
        "HistoricalStrategySupport": float(support_probability),
        "HistoricalStrategyPenalty": float(historical_strategy_penalty),
        "HistoricalPitWindowPenalty": float(historical_pit_penalty),
        "TyreUsage": tyre_usage,
        "PhysicalConstraintsSatisfied": True,
        "MLSupportProbability": ml_support,
        "MLTransitionProbabilities": all_probabilities,
        "MaxTyreAgeUtilization": float(max_age_use),
        "MaxSetUtilization": float(max_set_use),
    }


def generate_feasible_strategies(
    *,
    race_name: str,
    driver: str,
    race_distance: int,
    weather: dict[str, Any],
    surface: dict[str, np.ndarray],
    weather_adjustments: dict[str, np.ndarray],
    degradation: dict[str, float],
    classifier: dict[str, Any] | None,
    season: int = 2024,
    circuit: str = "Unknown",
    team: str = "Unknown",
    inventory: dict[str, int] | None = None,
    include_three_stop: bool = True,
    historical_strategy_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Enumerate legal 1/2-stop plans and 3-stop plans for sufficiently long races."""

    if isinstance(race_distance, bool) or not isinstance(race_distance, (int, np.integer)) or race_distance <= 0:
        raise ValueError("Race distance must be a positive integer number of laps.")
    condition = str(weather.get("condition", "DRY")).upper()
    records: list[dict[str, Any]] = []
    candidate_specs: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    seen_specs: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
    # Four-stint enumeration grows rapidly for mixed-weather compounds. Restrict
    # the additional search to long dry races where it can be useful and keep
    # damp/wet scenario runs within the same bounded 1/2-stop candidate budget.
    allow_three_stop = bool(include_three_stop and condition == "DRY" and int(race_distance) >= 65)
    four_stint_sequences: set[tuple[str, ...]] = set()
    generation_started = perf_counter()
    for sequence in generate_compound_sequences(condition, inventory, include_three_stop=allow_three_stop):
        resolution = 5 if len(sequence) == 4 else 1
        if len(sequence) == 4:
            four_stint_sequences.add(sequence)
        for lengths in generate_stint_lengths(sequence, int(race_distance), resolution=resolution):
            candidate_specs.append((sequence, lengths))
            seen_specs.add((sequence, lengths))
    generation_ms = (perf_counter() - generation_started) * 1000.0

    def contexts_for(specs):
        contexts = []
        for sequence, lengths in specs:
            for index, pit_lap in enumerate(np.cumsum(lengths)[:-1]):
                contexts.append((race_name, driver, sequence[index], int(lengths[index]), int(pit_lap), season, circuit, team, race_distance))
        return contexts

    def score_specs(specs, cache):
        scored = []
        for sequence, lengths in specs:
            scored.append(_simulation_record(
                race_name=race_name,
                driver=driver,
                season=season,
                circuit=circuit,
                team=team,
                race_distance=race_distance,
                tyres=sequence,
                stint_lengths=lengths,
                weather=weather,
                surface=surface,
                weather_adjustments=weather_adjustments,
                degradation=degradation,
                classifier=classifier,
                classifier_cache=cache,
                inventory=inventory,
                historical_strategy_context=historical_strategy_context,
            ))
        return scored

    simulation_started = perf_counter()
    classifier_cache = _classifier_probabilities_batch(classifier, contexts_for(candidate_specs))
    records.extend(score_specs(candidate_specs, classifier_cache))

    # Four-stint search is much larger than the one- and two-stop space. Score
    # it on a five-lap grid, then inspect every adjacent lap split around the
    # strongest three-stop plans. Other stop counts remain exhaustively scored.
    if four_stint_sequences and records:
        best_by_sequence: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in records:
            sequence = tuple(row["Tyres"])
            if len(sequence) == 4 and (sequence not in best_by_sequence or row["StrategyScore"] < best_by_sequence[sequence]["StrategyScore"]):
                best_by_sequence[sequence] = row
        best_score = min(row["StrategyScore"] for row in records)
        refinements = sorted(
            (row for row in best_by_sequence.values() if row["StrategyScore"] <= best_score + 2.5),
            key=lambda row: row["StrategyScore"],
        )[:20]
        refine_specs = []
        for row in refinements:
            sequence = tuple(row["Tyres"])
            coarse = tuple(row["StintLengths"])
            bounds = [stint_age_bounds(compound) for compound in sequence]
            for deltas in itertools.product(range(-2, 3), repeat=3):
                first_three = tuple(coarse[index] + deltas[index] for index in range(3))
                lengths = first_three + (int(race_distance) - sum(first_three),)
                if any(length < lower or length > upper for length, (lower, upper) in zip(lengths, bounds)):
                    continue
                key = (sequence, lengths)
                if key not in seen_specs:
                    seen_specs.add(key)
                    refine_specs.append(key)
        if refine_specs:
            missing_contexts = [context for context in contexts_for(refine_specs) if context not in classifier_cache]
            classifier_cache.update(_classifier_probabilities_batch(classifier, missing_contexts))
            records.extend(score_specs(refine_specs, classifier_cache))
    if not records:
        raise ValueError(
            f"No feasible strategy exists for {race_distance} laps under {condition} conditions."
        )
    results = pd.DataFrame(records)
    results = results.sort_values(
        ["StrategyScore", "EstimatedRaceTime", "PitLap1", "PitLap2", "PitLap3"],
        na_position="last",
    ).reset_index(drop=True)
    # Defense in depth: impossible candidates must never leave this function.
    if not results["PhysicalConstraintsSatisfied"].all():
        raise AssertionError("An impossible strategy appeared in ranked results.")
    results.attrs["prediction_timings"] = {
        "strategy_generation_ms": round(generation_ms, 2),
        "strategy_simulation_ms": round((perf_counter() - simulation_started) * 1000.0, 2),
    }
    results.attrs["search_method"] = (
        "One- and two-stop plans are exhaustive. Four-stint plans use a five-lap coarse grid, "
        "with adjacent-lap refinement for three-stop plans close to the overall minimum."
    ) if allow_three_stop else "One- and two-stop plans are exhaustively evaluated."
    return results


def _record_for_json(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, dict):
            result[key] = _record_for_json(value)
        elif isinstance(value, list):
            result[key] = [_record_for_json(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[key] = float(value) if np.isfinite(value) else None
        elif isinstance(value, (np.bool_, bool)):
            result[key] = bool(value)
        elif value is pd.NA or (not isinstance(value, (str, int, bool)) and pd.isna(value)):
            result[key] = None
        else:
            result[key] = value
    return result


def _summary_record(row: pd.Series) -> dict[str, Any]:
    record = row.to_dict()
    return _record_for_json(record)


def _confidence(
    *,
    driver_laps: int,
    lap_metrics: dict[str, Any],
    candidates: pd.DataFrame,
    weather: dict[str, Any],
    recommended: pd.Series,
    prediction_sources: dict[str, str],
) -> tuple[float, str, dict[str, float]]:
    data_support = min(driver_laps / 200.0, 1.0)
    mae = lap_metrics.get("MAE")
    model_quality = max(0.0, 1.0 - float(mae) / 8.0) if mae is not None and np.isfinite(float(mae)) else 0.25
    by_sequence = candidates.groupby("Strategy", as_index=False)["StrategyScore"].min().sort_values("StrategyScore")
    if len(by_sequence) >= 2:
        gap = max(0.0, float(by_sequence.iloc[1]["StrategyScore"] - by_sequence.iloc[0]["StrategyScore"]))
    else:
        gap = 0.0
    gap_signal = min(gap / 4.0, 1.0)
    certainty = {"HIGH": 1.0, "MEDIUM": 0.65, "LOW": 0.3}.get(
        str(weather.get("weather_certainty", "LOW")).upper(), 0.3
    )
    robustness = 1.0 - min(
        1.0,
        0.60 * float(recommended["MaxTyreAgeUtilization"])
        + 0.40 * float(recommended["MaxSetUtilization"]),
    )
    selected_tyres = set(recommended["Tyres"])
    wet_unsupported = (
        weather.get("condition") == "WET"
        and "WET" in selected_tyres
        and "estimated wet-tyre heuristic; no trained WET model" in prediction_sources.get("WET", "")
    ) or any(
        "estimated intermediate fallback" in prediction_sources.get(compound, "")
        for compound in selected_tyres
    )
    ml = recommended.get("MLSupportProbability")
    ml_signal = 0.35 if ml is None or pd.isna(ml) else float(ml)
    # Each term is a measurable signal; the 20-point floor represents model
    # uncertainty, not a claim that the strategy is correct with 20% certainty.
    score = (
        20.0
        + 15.0 * data_support
        + 15.0 * model_quality
        + 25.0 * gap_signal
        + 10.0 * certainty
        + 10.0 * robustness
        + 5.0 * ml_signal
        - (12.0 if wet_unsupported else 0.0)
    )
    score = float(np.clip(score, 20.0, 95.0))
    level = "HIGH" if score >= 75 else "MEDIUM" if score >= 50 else "LOW"
    drivers = {
        "historical_data_support": round(15.0 * data_support, 2),
        "lap_model_quality": round(15.0 * model_quality, 2),
        "strategy_margin": round(25.0 * gap_signal, 2),
        "weather_certainty": round(10.0 * certainty, 2),
        "strategy_robustness": round(10.0 * robustness, 2),
        "ml_transition_support": round(5.0 * ml_signal, 2),
        "unsupported_wet_model_adjustment": -12.0 if wet_unsupported else 0.0,
        "top_distinct_sequence_gap_seconds": round(gap, 3),
    }
    return score, level, drivers


def _model_next_tyre(
    artifact: dict[str, Any] | None,
    race_name: str,
    driver: str,
    recommendation: pd.Series,
    *,
    season: int = 2024,
    circuit: str = "Unknown",
    team: str = "Unknown",
    race_distance: int = 1,
) -> tuple[str | None, dict[str, float], str]:
    if artifact is None or len(recommendation["PitLaps"]) < 1:
        return None, {}, "Classifier unavailable or recommendation has no pit transition."
    current = recommendation["FirstTyre"]
    next_tyre = recommendation["SecondTyre"]
    pit_lap = int(recommendation["PitLap1"])
    probabilities = _classifier_probabilities(
        artifact, race_name, driver, current, int(recommendation["StintLengths"][0]), pit_lap,
        season, circuit, team, race_distance
    )
    if not probabilities:
        return None, {}, "Classifier could not score this transition context."
    predicted = max(probabilities, key=probabilities.get)
    return predicted, probabilities, f"The classifier supports the recommendation with P({next_tyre})={probabilities.get(next_tyre, 0.0):.3f}. It does not override inventory checks."


def _write_outputs(
    *,
    result: dict[str, Any],
    candidates: pd.DataFrame,
    weather: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / STRATEGY_RESULTS_PATH.name
    weather_path = output_dir / "weather_analysis.csv"
    backend_path = output_dir / BACKEND_RESULT_PATH.name
    candidates.to_csv(result_path, index=False)
    pd.DataFrame(weather_rows(weather)).to_csv(weather_path, index=False)
    backend_path.write_text(json.dumps(_record_for_json(result), indent=2, ensure_ascii=False), encoding="utf-8")


def run_analysis(
    race_name: str,
    driver: str,
    *,
    season: int | None = None,
    weather_override: dict[str, Any] | None = None,
    fetch_weather: bool = True,
    output_dir: str | Path | None = None,
    analysis_mode: str = "historical",
    cutoff_time: str | pd.Timestamp | None = None,
    event_info_override: dict[str, Any] | None = None,
    training_laps: pd.DataFrame | None = None,
    lap_artifact_override: dict[str, Any] | None = None,
    pit_artifact_override: dict[str, Any] | None = None,
    metrics_override: dict[str, Any] | None = None,
    inventory: dict[str, int] | None = None,
    performance_summary_override: pd.DataFrame | None = None,
    precomputed_context: dict[str, Any] | None = None,
    historical_strategy_context: dict[str, Any] | None = None,
    persist_outputs: bool = True,
) -> dict[str, Any]:
    """Run historical analysis or strict as-of pre-race analysis and export results."""

    driver_code = str(driver).strip().upper()
    if not driver_code:
        raise ValueError("Select a valid driver before running analysis.")
    mode = str(analysis_mode).strip().casefold().replace("_", "-")
    pre_race = mode in {"pre-race", "prerace"}
    if mode not in {"historical", "pre-race", "prerace"}:
        raise ValueError("Analysis mode must be 'historical' or 'pre-race'.")
    if pre_race:
        if cutoff_time is None or event_info_override is None or training_laps is None:
            raise ValueError("Pre-race mode requires a cutoff time, target event metadata, and source laps.")
        laps = filter_laps_before_cutoff(training_laps, cutoff_time)
        if laps.empty:
            raise ValueError("No historical lap rows precede this race's conservative cutoff.")
        lap_artifact = lap_artifact_override
        pit_artifact = pit_artifact_override
        metrics = metrics_override or {
            "lap_time": lap_artifact.get("metrics", {}) if lap_artifact else {"available": False, "reason": "Insufficient pre-cutoff history."},
            "pit_stop": pit_artifact.get("metrics", {}) if pit_artifact else {"available": False, "reason": "Insufficient pre-cutoff transitions."},
        }
        if lap_artifact_override is None or pit_artifact_override is None:
            raise ValueError("Pre-race mode requires pre-trained cutoff-safe model artifacts; live training is disabled.")
    else:
        if not CLEAN_LAP_DATA_PATH.exists():
            prepare_data()
        if training_laps is not None:
            laps = training_laps.copy()
            lap_artifact = lap_artifact_override
            pit_artifact = pit_artifact_override
            metrics = metrics_override or {}
        else:
            laps, lap_artifact, pit_artifact, metrics = _ensure_training_assets()
    race_history = laps[laps["Race"].eq(race_name)] if not pre_race else pd.DataFrame()
    if season is None:
        available_season = race_history if not race_history.empty else laps
        if "Season" in available_season and not available_season.empty:
            season = int(pd.to_numeric(available_season["Season"], errors="coerce").max())
    season = int(season if season is not None else 2024)
    if "Season" in race_history:
        race_history = race_history[pd.to_numeric(race_history["Season"], errors="coerce").eq(season)]
    if not pre_race and race_history.empty:
        raise ValueError(f"No historical laps are available for {season} {race_name}.")
    driver_history = (
        laps[laps["Driver"].astype(str).str.upper().eq(driver_code)].copy()
        if pre_race else race_history[race_history["Driver"].astype(str).str.upper().eq(driver_code)].copy()
    )
    if driver_history.empty:
        available = ", ".join(sorted(laps["Driver"].dropna().astype(str).str.upper().unique().tolist()))
        raise ValueError(f"No usable laps for {driver_code} at {race_name}. Available drivers: {available}")

    if pre_race:
        race_info = dict(event_info_override or {})
        race_info.setdefault("season", season)
        race_info.setdefault("round", 0)
        race_info.setdefault("race", race_name)
        race_info.setdefault("official_name", race_name)
        race_info.setdefault("circuit", "Unknown")
        race_info.setdefault("location", race_info.get("circuit", "Unknown"))
        race_info.setdefault("country", "Unknown")
        race_info.setdefault("date", "")
        race_info.setdefault("laps", race_info.get("RaceLaps"))
        if not race_info.get("laps"):
            raise ValueError("Race distance is unavailable; pre-race simulation needs an identified distance.")
    else:
        race_info = event_info_from_laps(laps, season, race_name)
    driver_history = driver_history.copy()
    team_values = driver_history.get("Team", pd.Series(dtype=str)).dropna().astype(str)
    team_values = team_values[team_values.str.strip().ne("")]
    selected_team = team_values.mode().iloc[0] if not team_values.empty else "Unknown"
    driver_name_values = driver_history.get("DriverName", pd.Series(dtype=str)).dropna().astype(str)
    driver_name_values = driver_name_values[driver_name_values.str.strip().ne("")]
    selected_driver_name = driver_name_values.mode().iloc[0] if not driver_name_values.empty else driver_code
    driver_number_values = driver_history.get("DriverNumber", pd.Series(dtype=str)).dropna().astype(str)
    driver_number_values = driver_number_values[driver_number_values.str.strip().ne("")]
    selected_driver_number = driver_number_values.mode().iloc[0] if not driver_number_values.empty else None
    if weather_override is not None:
        weather = _weather_override(race_name, race_info, weather_override)
    elif fetch_weather:
        weather = get_historical_weather(race_name, race_info)
    else:
        weather = fallback_weather(race_name, race_info, "Weather lookup disabled for this run.")

    tyre_summary = (
        performance_summary_override.copy()
        if performance_summary_override is not None
        else analyze_tyre_performance(laps)[0]
    )
    offsets, degradation, compound_samples = _performance_lookups(laps, summary=tyre_summary)
    baseline, _ = _driver_baseline(
        laps, race_name, driver_code, offsets, degradation,
        season=None if pre_race else season,
        circuit=str(race_info.get("circuit", "Unknown")),
    )
    surface_started = perf_counter()
    surface, weather_adjustments, prediction_sources = _build_prediction_surface(
        race_name=race_name,
        driver=driver_code,
        season=season,
        circuit=str(race_info["circuit"]),
        team=selected_team,
        race_distance=int(race_info["laps"]),
        laps=laps,
        model_artifact=lap_artifact,
        weather=weather,
        offsets=offsets,
        degradation=degradation,
        samples=compound_samples,
        baseline=baseline,
    )
    surface_ms = (perf_counter() - surface_started) * 1000.0
    candidates = generate_feasible_strategies(
        race_name=race_name,
        driver=driver_code,
        race_distance=int(race_info["laps"]),
        weather=weather,
        surface=surface,
        weather_adjustments=weather_adjustments,
        degradation=degradation,
        classifier=pit_artifact,
        season=season,
        circuit=str(race_info["circuit"]),
        team=selected_team,
        inventory=inventory,
        historical_strategy_context=historical_strategy_context,
    )
    strategy_timings = candidates.attrs.get("prediction_timings", {})
    recommended = candidates.iloc[0]
    best_one_stop = candidates[candidates["StrategyType"].eq("1-STOP")].head(1)
    best_two_stop = candidates[candidates["StrategyType"].eq("2-STOP")].head(1)
    best_three_stop = candidates[candidates["StrategyType"].eq("3-STOP")].head(1)
    next_tyre, probabilities, ml_reason = _model_next_tyre(
        pit_artifact, race_name, driver_code, recommended,
        season=season,
        circuit=str(race_info["circuit"]),
        team=selected_team,
        race_distance=int(race_info["laps"]),
    )
    confidence, confidence_level, confidence_factors = _confidence(
        driver_laps=len(driver_history),
        lap_metrics=metrics.get("lap_time", {}),
        candidates=candidates,
        weather=weather,
        recommended=recommended,
        prediction_sources=prediction_sources,
    )

    recommendation = _summary_record(recommended)
    first_compound = recommendation["FirstTyre"]
    first_lap_prediction = float(surface[first_compound][1, 1]) if np.isfinite(surface[first_compound][1, 1]) else None
    comparison = []
    if not best_one_stop.empty:
        comparison.append(_summary_record(best_one_stop.iloc[0]))
    if not best_two_stop.empty:
        comparison.append(_summary_record(best_two_stop.iloc[0]))
    if not best_three_stop.empty:
        comparison.append(_summary_record(best_three_stop.iloc[0]))
    diverse_candidates = candidates.drop_duplicates(subset=["Strategy"], keep="first").head(10)
    top_strategies = [_summary_record(row) for _, row in diverse_candidates.iterrows()]
    stint_strings = [f"{length} laps on {compound}" for compound, length in zip(recommended["Tyres"], recommended["StintLengths"])]
    alternative = candidates.iloc[1] if len(candidates) > 1 else None
    separation = float(alternative["StrategyScore"] - recommended["StrategyScore"]) if alternative is not None else None
    constraint_summary = (
        "physical set, dry-compound-diversity, and tyre-age constraints"
        if weather["condition"] == "DRY"
        else "physical set, weather-appropriate compound, and tyre-age constraints"
    )
    reason = (
        f"{recommended['Strategy']} is the lowest-scoring feasible plan in {len(candidates):,} simulated candidates "
        f"for {selected_driver_name} at {race_info['circuit']} in {season}. It uses {len(recommended['PitLaps'])} pit stop(s) "
        f"with stint lengths {', '.join(map(str, recommended['StintLengths']))}; the estimate includes a {PIT_LOSS:.0f}-second "
        f"loss per stop and the fitted tyre degradation. The {constraint_summary} passed before ranking."
    )
    if separation is not None:
        reason += f" The next-ranked candidate scores {separation:.2f} seconds higher."
    if lap_artifact is not None:
        reason += " Lap-time estimates blend the Random Forest with historical tyre profiles."
    else:
        reason += " Lap-time estimates use historical tyre profiles because no trained lap-time model is available."
    available_seasons = (
        list(precomputed_context.get("available_seasons", []))
        if pre_race and precomputed_context else
        sorted(pd.to_numeric(laps.get("Season", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    )
    season_is_partial = season == date.today().year
    season_history = laps[pd.to_numeric(laps["Season"], errors="coerce").eq(season)] if "Season" in laps else laps
    source_season_races = (precomputed_context or {}).get("season_event_counts", {}).get(str(season), int(season_history["Race"].nunique()))
    source_season_drivers = (precomputed_context or {}).get("season_driver_counts", {}).get(str(season), int(season_history["Driver"].nunique()))
    data_coverage = {
        "available_seasons": available_seasons,
        "season_status": "current/partial season" if season_is_partial else "historical source-available data",
        "selected_race_rows": int(len(race_history)),
        "selected_driver_rows": int(len(driver_history)),
        "selected_season_races": int(source_season_races),
        "selected_season_drivers": int(source_season_drivers),
        "total_lap_rows": int((precomputed_context or {}).get("training_rows", len(laps))),
    }
    result = {
        "analysis_mode": "PRE-RACE" if pre_race else "HISTORICAL",
        "training_cutoff_time_utc": str(pd.to_datetime(cutoff_time, utc=True).isoformat()) if pre_race else None,
        "selected_event_laps_used_for_training": False if pre_race else None,
        "training_rows": int((precomputed_context or {}).get("training_rows", len(laps))) if pre_race else None,
        "season": season,
        "season_status": data_coverage["season_status"],
        "race": race_name,
        "official_event_name": race_info["official_name"],
        "event_round": race_info["round"],
        "driver": driver_code,
        "driver_name": selected_driver_name,
        "driver_number": selected_driver_number,
        "team": selected_team,
        "circuit": race_info["circuit"],
        "location": race_info["location"],
        "country": race_info["country"],
        "race_date": race_info["date"],
        "race_distance": int(race_info["laps"]),
        "data_coverage": data_coverage,
        "inventory": inventory or {**TYRE_SET_LIMITS, **WET_TYRE_SET_LIMITS},
        "inventory_source": "event nomination register" if inventory else "8/3/2 dry-set baseline and 4/4 wet-set model assumption",
        "weather": weather,
        "track_condition": weather["condition"],
        "weather_impact": weather["impact"],
        "weather_source": weather["source"],
        "temperature": weather.get("temperature"),
        "rain": weather.get("rain"),
        "precipitation": weather.get("precipitation"),
        "precipitation_probability": weather.get("precipitation_probability"),
        "wind_speed": weather.get("wind_speed"),
        "wind_direction": weather.get("wind_direction"),
        "weather_reason": weather["reason"],
        "recommended_strategy": recommendation,
        "strategy": recommendation["Strategy"],
        "strategy_type": recommendation["StrategyType"],
        "pit_laps": recommendation["PitLaps"],
        "stint_lengths": recommendation["StintLengths"],
        "stints": stint_strings,
        "tyre_usage": recommendation["TyreUsage"],
        "physical_constraints_satisfied": bool(recommendation["PhysicalConstraintsSatisfied"]),
        "estimated_race_time": recommendation["EstimatedRaceTime"],
        "first_lap_prediction": first_lap_prediction,
        "pit_stop_cost": recommendation["PitStopCost"],
        "degradation_impact": recommendation["DegradationImpact"],
        "weather_adjustment_seconds": recommendation["WeatherAdjustment"],
        "strategy_penalty": recommendation["StrategyPenalty"],
        "strategy_score": recommendation["StrategyScore"],
        "strategy_score_breakdown": recommendation.get("ScoreBreakdown", {}),
        "confidence_score": confidence,
        "confidence_level": confidence_level,
        "confidence_factors": confidence_factors,
        "ml_predicted_tyre": next_tyre,
        "ml_probabilities_fraction": probabilities,
        "ml_probabilities": {key: round(value * 100.0, 3) for key, value in probabilities.items()},
        "ml_explanation": ml_reason,
        "model_metrics": metrics,
        "lap_prediction_sources": prediction_sources,
        "tyre_performance": tyre_summary.to_dict(orient="records"),
        "degradation_rates": degradation,
        "strategy_comparison": comparison,
        "top_feasible_strategies": top_strategies,
        "strategy_alternatives": top_strategies[1:],
        "feasible_candidate_count": int(len(candidates)),
        "pit_loss_seconds_per_stop": float(PIT_LOSS),
        "reason": reason,
        # Flat compatibility fields retained for earlier dashboard consumers.
        "dry_strategy": recommendation["Strategy"],
        "weather_strategy": recommendation["Strategy"],
        "preferred_strategy": recommendation["StrategyType"],
        "primary_tyre": recommendation["FirstTyre"],
        "backup_tyre": recommendation["SecondTyre"],
        "pit_lap_1": recommendation["PitLap1"],
        "pit_lap_2": recommendation["PitLap2"],
        "weather_pit_lap_1": recommendation["PitLap1"],
        "one_stop_strategy": None if best_one_stop.empty else best_one_stop.iloc[0]["Strategy"],
        "one_stop_time": None if best_one_stop.empty else float(best_one_stop.iloc[0]["EstimatedRaceTime"]),
        "two_stop_strategy": None if best_two_stop.empty else best_two_stop.iloc[0]["Strategy"],
        "two_stop_time": None if best_two_stop.empty else float(best_two_stop.iloc[0]["EstimatedRaceTime"]),
        "three_stop_strategy": None if best_three_stop.empty else best_three_stop.iloc[0]["Strategy"],
        "three_stop_time": None if best_three_stop.empty else float(best_three_stop.iloc[0]["EstimatedRaceTime"]),
        "strategy_score_difference": (
            None
            if best_one_stop.empty or best_two_stop.empty
            else float(abs(best_one_stop.iloc[0]["StrategyScore"] - best_two_stop.iloc[0]["StrategyScore"]))
        ),
        "prediction_timings": {
            "surface_model_ms": round(surface_ms, 2),
            "strategy_generation_ms": float(strategy_timings.get("strategy_generation_ms", 0.0)),
            "strategy_simulation_ms": float(strategy_timings.get("strategy_simulation_ms", 0.0)),
        },
    }
    destination = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
    if persist_outputs:
        _write_outputs(result=result, candidates=candidates, weather=weather, output_dir=destination)
    return {"result": result, "strategies": candidates, "weather": weather}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run the F1 pit-stop strategy simulator.")
    parser.add_argument("--season", type=int, default=None, help="Season year from locally available data.")
    parser.add_argument("--race", default=None, help="Event name available in the selected season.")
    parser.add_argument("--driver", default=None, help="Driver code available for the selected event.")
    parser.add_argument("--no-weather", action="store_true", help="Use a labeled dry fallback instead of calling Open-Meteo.")
    args = parser.parse_args(argv)

    if not CLEAN_LAP_DATA_PATH.exists():
        prepare_data()
    laps = pd.read_csv(CLEAN_LAP_DATA_PATH)
    available_seasons = sorted(pd.to_numeric(laps.get("Season", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    if not available_seasons:
        raise SystemExit("The local dataset has no season metadata.")
    season = args.season if args.season is not None else max(available_seasons)
    season_rows = laps[pd.to_numeric(laps["Season"], errors="coerce").eq(season)]
    if season_rows.empty:
        raise SystemExit(f"No lap data is available for season {season}. Available seasons: {available_seasons}")
    if args.race is None:
        race_names = sorted(season_rows["Race"].dropna().unique().tolist())
        race = race_names[0]
    else:
        race = args.race
    race_laps = season_rows[season_rows["Race"].eq(race)]
    drivers = sorted(race_laps["Driver"].dropna().unique().tolist())
    if not drivers:
        raise SystemExit(f"No historical drivers are available for {season} {race}.")
    driver = str(args.driver or drivers[0]).upper()
    analysis = run_analysis(race, driver, season=season, fetch_weather=not args.no_weather)
    result = analysis["result"]
    print(json.dumps(_record_for_json(result), indent=2, ensure_ascii=True))
    print(f"\nSaved strategy results to {STRATEGY_RESULTS_PATH}")
    print(f"Saved weather analysis to {PROCESSED_DATA_DIR / 'weather_analysis.csv'}")
    print(f"Saved backend result to {BACKEND_RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
