"""Leakage-aware, cached prediction service built around the strategy engine."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from threading import RLock
from time import monotonic, perf_counter
from typing import Any
from pathlib import Path

import pandas as pd

from services.data_service import (
    app_today,
    classify_event_status,
    driver_options_for_event,
    historical_summary,
    tyre_degradation_for_circuit,
    tyre_performance_for_circuit,
)
from services.database import (
    connect, driver_laps_frame, event_driver, get_event, prediction_data_version, laps_frame,
)
from services.weather_service import cached_weather, historical_weather, historical_weather_for_circuit, live_weather, store_weather
from config import EVENT_TYRE_NOMINATIONS_PATH, TYRE_ALLOCATIONS_PATH


@lru_cache(maxsize=1)
def _all_laps() -> pd.DataFrame:
    return laps_frame()


_CACHE_LOCK = RLock()
_PREDICTION_CACHE: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_CACHE_TTL_SECONDS = 6 * 60 * 60
_CACHE_MAX_ENTRIES = 48
_CACHE_GENERATION = 0


class PredictionArtifactUnavailable(RuntimeError):
    """The requested event needs an offline cutoff-safe artifact refresh."""


@lru_cache(maxsize=96)
def _cached_history_summary(circuit: str, before: str, version: str, exclude_event_id: str = "") -> dict[str, Any]:
    return historical_summary(circuit, before=before, exclude_event_id=exclude_event_id or None)


@lru_cache(maxsize=96)
def _cached_tyre_profile(circuit: str, before: str, version: str, exclude_event_id: str = "") -> tuple[dict[str, Any], ...]:
    return tuple(tyre_performance_for_circuit(circuit, before=before, exclude_event_id=exclude_event_id or None))


@lru_cache(maxsize=96)
def _cached_degradation_curve(circuit: str, before: str, version: str, exclude_event_id: str = "") -> tuple[dict[str, Any], ...]:
    return tuple(tyre_degradation_for_circuit(circuit, before=before, exclude_event_id=exclude_event_id or None))


def _data_version(cutoff: str) -> str:
    return prediction_data_version(cutoff)


def clear_lap_cache() -> None:
    """Invalidate prediction and aggregate caches after importing updated data."""
    global _CACHE_GENERATION
    _all_laps.cache_clear()
    _cached_history_summary.cache_clear()
    _cached_tyre_profile.cache_clear()
    _cached_degradation_curve.cache_clear()
    try:
        from services.data_service import clear_circuit_analysis_cache
        clear_circuit_analysis_cache()
    except Exception:
        pass
    try:
        from pre_race_analysis import clear_pre_race_caches
        clear_pre_race_caches()
    except Exception:
        pass
    with _CACHE_LOCK:
        _CACHE_GENERATION += 1
        _PREDICTION_CACHE.clear()


def _forecast_fingerprint(package: dict[str, Any]) -> str:
    forecast = package.get("forecast", {})
    packed = json.dumps(forecast, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:20]


def _event_tyre_register_version() -> str:
    digest = hashlib.sha256()
    for path in (EVENT_TYRE_NOMINATIONS_PATH, TYRE_ALLOCATIONS_PATH):
        try:
            digest.update(Path(path).read_bytes())
        except OSError:
            digest.update(b"missing")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _cached_prediction(key: str) -> dict[str, Any] | None:
    now = monotonic()
    with _CACHE_LOCK:
        item = _PREDICTION_CACHE.get(key)
        if item is None:
            return None
        created, payload = item
        if now - created >= _CACHE_TTL_SECONDS:
            _PREDICTION_CACHE.pop(key, None)
            return None
        _PREDICTION_CACHE.move_to_end(key)
        return copy.deepcopy(payload)


def _save_prediction(key: str, payload: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _PREDICTION_CACHE[key] = (monotonic(), copy.deepcopy(payload))
        _PREDICTION_CACHE.move_to_end(key)
        while len(_PREDICTION_CACHE) > _CACHE_MAX_ENTRIES:
            _PREDICTION_CACHE.popitem(last=False)


def predict_strategy(
    event_id: str,
    driver: str,
    *,
    fetch_weather: bool = True,
    refresh_weather: bool = False,
    output_dir=None,
    weather_package_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict from offline artifacts and selected-driver history; never trains models."""
    started = perf_counter()
    prediction_start = datetime.now(timezone.utc).isoformat()
    database_started = perf_counter()
    event = get_event(event_id)
    if not event:
        raise ValueError("The selected race is not available in the local SQLite database.")
    driver_code = str(driver or "").strip().upper()
    if not driver_code:
        raise ValueError("Select a valid driver before requesting a prediction.")
    event_status = classify_event_status(event, app_today())
    driver_context = "observed participant"
    participant = event_driver(event_id, driver_code)
    if participant is None:
        if event_status not in {"UPCOMING", "CURRENT"}:
            raise ValueError("That driver did not participate in the selected race according to the loaded data.")
        options = driver_options_for_event(event_id)
        if driver_code not in {row["driver_code"] for row in options["drivers"]}:
            raise ValueError("Choose a driver with historical laps before this race date.")
        driver_context = "historical driver candidate; entry is not confirmed"

    event_date = str(event.get("event_date") or "")[:10]
    if not event_date or event_date == "None":
        raise ValueError("The selected race does not have a usable event date for a pre-race cutoff.")
    # Date-only lap rows are conservatively excluded for the full local day of a current weekend.
    analysis_time = datetime.now(timezone.utc)
    if event_status == "CURRENT":
        cutoff = f"{app_today().isoformat()}T00:00:00Z"
        cutoff_source = "date-only lap source; all current-day rows excluded to avoid session leakage"
    else:
        cutoff = f"{event_date}T00:00:00Z"
        cutoff_source = "conservative midnight UTC event-date cutoff"
    data_version = _data_version(cutoff)
    driver_history = driver_laps_frame(driver_code, before=cutoff)
    if driver_history.empty:
        raise ValueError(f"{driver_code} has no historical laps available before the selected race cutoff.")
    database_load_ms = (perf_counter() - database_started) * 1000.0

    weather_started = perf_counter()
    if weather_package_override is not None:
        weather_package = weather_package_override
    elif fetch_weather:
        # The prediction path is cache-only by default. Explicit refreshes and
        # the separate weather endpoint are responsible for network I/O.
        weather_package = live_weather(event, force_refresh=True) if refresh_weather else cached_weather(event_id)
    else:
        weather_package = {
            "current": {"available": False, "reason": "Weather lookup disabled."},
            "forecast": {"available": False, "reason": "Weather lookup disabled.", "hourly": []},
            "source": "disabled",
        }
    weather_ms = (perf_counter() - weather_started) * 1000.0
    weather_key = _forecast_fingerprint(weather_package)
    cache_key = "|".join((str(event_id), driver_code, cutoff, data_version, _event_tyre_register_version(), weather_key, str(_CACHE_GENERATION)))
    if not refresh_weather:
        cached = _cached_prediction(cache_key)
        if cached is not None:
            if isinstance(cached.get("result"), dict):
                cached["result"]["weather"] = weather_package
                cached["result"]["analysis_timestamp_utc"] = analysis_time.isoformat()
            cache_hits = {"prediction": True, "model_artifact": True, "precomputed_features": True, "weather": bool(weather_package.get("cache_status") == "hit")}
            cached["prediction_timings"] = {
                **cached.get("prediction_timings", {}),
                "prediction_start": prediction_start,
                "database_load_ms": round(database_load_ms, 2),
                "weather_ms": round(weather_ms, 2),
                "response_ms": round((perf_counter() - started) * 1000.0, 2),
            }
            cached["prediction_timings"] = _expose_timing_aliases(cached["prediction_timings"])
            cached["cache_hits"] = cache_hits
            cached["performance"] = {
                "seconds": round(perf_counter() - started, 3), "cache": "hit", "cache_ttl_seconds": _CACHE_TTL_SECONDS,
            }
            return cached

    target_distance = pd.to_numeric(pd.Series([event.get("race_laps")]), errors="coerce").iloc[0]
    if pd.isna(target_distance) or target_distance <= 0:
        raise ValueError("Race distance is unavailable for this event.")
    event_info = {
        "Season": int(event["season"]),
        "event_id": str(event_id),
        "Round": int(event["round"]),
        "Race": event["race"],
        "OfficialEventName": event.get("official_event_name") or event["race"],
        "Circuit": event.get("circuit") or "Unknown",
        "Country": event.get("country") or "Unknown",
        "Location": event.get("location") or event.get("circuit") or "Unknown",
        "EventDate": event_date,
        "RaceLaps": int(target_distance),
        "RaceLapsSource": "event metadata; used only as known race distance",
        "CutoffTimeUTC": cutoff,
        "DataCutoffTimeUTC": cutoff,
        "AnalysisTimestampUTC": analysis_time.isoformat(),
        "RaceStartTimeSource": cutoff_source,
        "EventStatus": event_status,
    }

    from pre_race_analysis import load_prediction_bundle, run_pre_race_analysis

    model_started = perf_counter()
    artifact, artifact_cache_hit = load_prediction_bundle(event_id, cutoff, data_version)
    model_load_ms = (perf_counter() - model_started) * 1000.0
    if artifact is None:
        from pre_race_analysis import prediction_bundle_status
        bundle_status = prediction_bundle_status(event_id, cutoff, data_version)
        status = str(bundle_status.get("status", "MISSING")).upper()
        if status == "BUILDING":
            raise PredictionArtifactUnavailable("MODEL UPDATE IN PROGRESS — wait for the cutoff-safe bundle rebuild to finish.")
        if status == "FAILED":
            reason = bundle_status.get("reason") or "Build failure reason unavailable."
            raise PredictionArtifactUnavailable(f"MODEL BUNDLE BUILD FAILED — {reason}")
        if status == "STALE":
            reason = bundle_status.get("reason") or "The saved bundle does not match the current cutoff data."
            raise PredictionArtifactUnavailable(
                f"CUTOFF-SAFE MODEL BUNDLE STALE — {reason} Run scripts/train_model.py --asof-event {event_id}."
            )
        raise PredictionArtifactUnavailable(
            f"No pre-trained cutoff-safe model bundle is ready for {event.get('race')} at cutoff {cutoff}. "
            f"Run: python scripts/train_model.py --asof-event {event_id}"
        )

    strategy_started = perf_counter()
    result_package = run_pre_race_analysis(
        event_info,
        driver_code,
        driver_history,
        fetch_weather=False,
        output_dir=output_dir,
        weather_package_override=weather_package,
        precomputed_artifact=artifact,
    )
    strategy_analysis_ms = (perf_counter() - strategy_started) * 1000.0
    result = result_package["result"]
    result["driver_context"] = driver_context
    result["event_status"] = event_status
    result["analysis_timestamp_utc"] = analysis_time.isoformat()
    result["training_cutoff_time_utc"] = cutoff
    result["weather"] = weather_package
    if fetch_weather or weather_package_override is not None:
        store_weather(event_id, weather_package)
    context_started = perf_counter()
    circuit = str(event.get("circuit") or "Unknown")
    before = cutoff[:10]
    historical_track = _cached_history_summary(circuit, before, data_version, str(event_id))
    tyre_profile = _cached_tyre_profile(circuit, before, data_version, str(event_id))
    degradation_curve = _cached_degradation_curve(circuit, before, data_version, str(event_id))
    event_weather = historical_weather(event_id)
    weather_context = historical_weather_for_circuit(circuit, before=before)
    database_load_ms += (perf_counter() - context_started) * 1000.0
    strategy_timings = result.get("prediction_timings", {})
    expected_strategy = result.get("expected_strategy", {})
    prediction_timings = {
        "prediction_start": prediction_start,
        "database_load_ms": round(database_load_ms, 2),
        "model_load_ms": round(model_load_ms, 2),
        "weather_ms": round(weather_ms, 2),
        "strategy_generation_ms": float(strategy_timings.get("strategy_generation_ms", 0.0)),
        "strategy_simulation_ms": float(strategy_timings.get("strategy_simulation_ms", 0.0)),
        "strategy_analysis_ms": round(strategy_analysis_ms, 2),
        "response_ms": round((perf_counter() - started) * 1000.0, 2),
    }
    prediction_timings = _expose_timing_aliases(prediction_timings)
    cache_hits = {
        "prediction": False,
        "model_artifact": bool(artifact_cache_hit),
        "precomputed_features": True,
        "weather": bool(weather_package.get("cache_status") == "hit"),
    }
    payload = {
        "event": {**event, "event_status": event_status},
        "driver": {"driver_code": driver_code, "entry_status": driver_context, "confirmed": participant is not None},
        "driver_context": driver_context,
        "result": result,
        "strategy": expected_strategy.get("strategy"),
        "pit_window": expected_strategy.get("pit_window"),
        "pit_laps": expected_strategy.get("pit_laps", []),
        "confidence": {"score": result.get("confidence_score"), "level": result.get("confidence_level"), "is_win_probability": False},
        "weather": weather_package,
        "forecast": weather_package.get("forecast", {}),
        "historical_context": {"track": historical_track, "weather": weather_context, "strategy": artifact.get("historical_context", {})},
        "strategy_alternatives": result.get("strategy_alternatives", []),
        "strategy_score_breakdown": result.get("strategy_score_breakdown", {}),
        "model_metrics": result.get("model_metrics", {}),
        "prediction_timings": prediction_timings,
        "cache_hits": cache_hits,
        "timeline": _timeline(result.get("expected_strategy", {}), int(target_distance)),
        "strategy_support": _strategy_support(result.get("scenarios", [])),
        "historical_track": copy.deepcopy(historical_track),
        "circuit_history": copy.deepcopy(historical_track.get("coverage", {})),
        "circuit_races_used": copy.deepcopy(historical_track.get("races", [])),
        "tyre_performance": [dict(row) for row in tyre_profile],
        "tyre_degradation": [dict(row) for row in degradation_curve],
        "historical_weather": event_weather,
        "historical_weather_context": weather_context,
        "performance": {
            "seconds": round(perf_counter() - started, 3),
            "cache": "miss",
            "cache_ttl_seconds": _CACHE_TTL_SECONDS,
            "training_data_rows": int(result.get("training_rows") or 0),
        },
    }
    payload["performance"]["cache_hits"] = cache_hits
    payload["performance"]["prediction_timings"] = prediction_timings
    payload["prediction_timings"]["response_ms"] = round((perf_counter() - started) * 1000.0, 2)
    _save_prediction(cache_key, payload)
    return payload


def _expose_timing_aliases(timings: dict[str, Any]) -> dict[str, Any]:
    """Keep the requested short timing names beside the existing detailed names."""
    timings.update({
        "database_ms": timings.get("database_load_ms", 0.0),
        "model_ms": timings.get("model_load_ms", 0.0),
        "simulation_ms": timings.get("strategy_simulation_ms", 0.0),
    })
    return timings


def _timeline(strategy: dict, race_distance: int) -> list[dict]:
    compounds = str(strategy.get("strategy", "")).split(" → ")
    lengths = strategy.get("stint_lengths", [])
    if not compounds or len(compounds) != len(lengths):
        return []
    start = 1
    output = []
    pit_laps = strategy.get("pit_laps", [])
    for index, (compound, length) in enumerate(zip(compounds, lengths), start=1):
        end = start + int(length) - 1
        output.append({
            "stint": index,
            "compound": compound,
            "start_lap": start,
            "end_lap": end,
            "laps": int(length),
            "pit_after_lap": pit_laps[index - 1] if index <= len(pit_laps) else None,
            "start_pct": round((start - 1) / race_distance * 100, 2),
            "width_pct": round(int(length) / race_distance * 100, 2),
        })
        start = end + 1
    return output


def _strategy_support(scenarios: list[dict]) -> list[dict]:
    support: dict[str, float] = {}
    for row in scenarios:
        name = str(row.get("strategy", ""))
        if name:
            support[name] = support.get(name, 0.0) + max(0.0, float(row.get("weight", 0.0)))
    total = sum(support.values())
    return [
        {"strategy": key, "scenario_weight": round(value / total, 4) if total else 0.0}
        for key, value in sorted(support.items(), key=lambda pair: pair[1], reverse=True)
    ]
