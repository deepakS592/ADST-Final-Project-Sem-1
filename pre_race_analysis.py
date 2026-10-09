"""As-of pre-race strategy orchestration with strict event-date leakage checks."""

from __future__ import annotations

import json
import hashlib
import uuid
from functools import lru_cache
from pathlib import Path
from collections import Counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from config import LAP_TIME_MODEL_PATH, MODEL_METRICS_PATH, PIT_STOP_MODEL_PATH, PRE_RACE_MODEL_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from data_loader import filter_laps_before_cutoff
from data_preparation import analyze_tyre_performance, build_pit_training_data
from event_tyre_data import allocation_for_event, load_event_nomination
from services.database import connect, get_feature_cache, laps_frame, prediction_data_version, put_feature_cache
from services.data_service import circuit_identity_key
from strategy_engine import _record_for_json, run_analysis
from train_models import _lap_features, load_model_artifact, train_lap_time_model, train_pit_stop_model
from weather_api import get_current_and_forecast_weather, weather_scenario_probabilities


ASOF_BUNDLE_VERSION = 2
ASOF_MODEL_CACHE_VERSION = f"sklearn_{sklearn.__version__}"


@lru_cache(maxsize=12)
def _load_asof_cache(path_text: str, fingerprint: str) -> dict[str, Any] | None:
    try:
        cached = joblib.load(path_text)
        return cached if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint else None
    except Exception:
        return None


def _frame_fingerprint(laps: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(laps, index=False).values.tobytes()).hexdigest()[:20]


def _saved_models_match_history(laps: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Use packaged fitted models only when their complete training window is safe."""
    lap_artifact = load_model_artifact(LAP_TIME_MODEL_PATH)
    pit_artifact = load_model_artifact(PIT_STOP_MODEL_PATH)
    if not lap_artifact or not pit_artifact:
        return None, None, {}
    lap_features = list(lap_artifact.get("categorical_features", [])) + list(lap_artifact.get("numeric_features", []))
    raw_required = [name for name in lap_features if name != "RaceProgress"] + ["LapTime"]
    valid_lap_rows = 0
    if lap_features and set(raw_required).issubset(laps.columns):
        valid_laps = laps.dropna(subset=raw_required)
        prepared_features = _lap_features(valid_laps)
        valid_lap_rows = int(prepared_features.dropna(subset=lap_features).shape[0])
    if not valid_lap_rows or int(lap_artifact.get("training_rows", -1)) != valid_lap_rows:
        return None, None, {}
    with connect() as connection:
        available_transitions = int(connection.execute("SELECT COUNT(*) FROM pit_transitions").fetchone()[0])
    if int(pit_artifact.get("training_rows", -1)) != available_transitions:
        return None, None, {}
    try:
        metrics = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        metrics = {"lap_time": lap_artifact.get("metrics", {}), "pit_stop": pit_artifact.get("metrics", {})}
    return lap_artifact, pit_artifact, metrics


def _asof_models(
    laps: pd.DataFrame,
    cutoff: str,
    *,
    fingerprint: str | None = None,
    allow_saved_models: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Reuse cutoff-specific fitted artifacts; retrain only when eligible data changes."""

    fingerprint = fingerprint or _frame_fingerprint(laps)
    # The eligible-data fingerprint is the real cache boundary. Different race
    # cutoffs with identical pre-cutoff rows can safely share the same fitted models.
    path = Path(PRE_RACE_MODEL_DIR) / f"asof_{ASOF_MODEL_CACHE_VERSION}_{fingerprint}.joblib"
    if path.exists():
        cached = _load_asof_cache(str(path), fingerprint)
        if cached:
            return cached.get("lap_artifact"), cached.get("pit_artifact"), cached.get("metrics", {})
    if allow_saved_models:
        lap_artifact, pit_artifact, metrics = _saved_models_match_history(laps)
        if lap_artifact is not None and pit_artifact is not None:
            cached = {"fingerprint": fingerprint, "lap_artifact": lap_artifact, "pit_artifact": pit_artifact, "metrics": metrics}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            joblib.dump(cached, temporary)
            temporary.replace(path)
            _load_asof_cache.cache_clear()
            return lap_artifact, pit_artifact, metrics
    lap_artifact, lap_metrics = train_lap_time_model(laps)
    pit_artifact, pit_metrics = train_pit_stop_model(build_pit_training_data(laps))
    metrics = {"lap_time": lap_metrics, "pit_stop": pit_metrics}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump({"fingerprint": fingerprint, "lap_artifact": lap_artifact, "pit_artifact": pit_artifact, "metrics": metrics}, temporary)
    temporary.replace(path)
    _load_asof_cache.cache_clear()
    return lap_artifact, pit_artifact, metrics


def clear_pre_race_caches() -> None:
    _load_asof_cache.cache_clear()
    _load_prediction_bundle.cache_clear()


def _prediction_bundle_path(event_id: str, cutoff: str, data_version: str) -> Path:
    cutoff_key = pd.to_datetime(cutoff, utc=True).strftime("%Y%m%dT%H%M%SZ")
    version_key = hashlib.sha256(str(data_version).encode("utf-8")).hexdigest()[:12]
    safe_event_id = "".join(character for character in str(event_id) if character.isalnum() or character in "-_")
    return Path(PRE_RACE_MODEL_DIR) / f"event_{safe_event_id}_{cutoff_key}_{version_key}.joblib"


def _bundle_metadata_matches(bundle: Any, event_id: str, cutoff: str, data_version: str) -> bool:
    if not isinstance(bundle, dict):
        return False
    expected = (ASOF_BUNDLE_VERSION, str(event_id), str(cutoff), str(data_version))
    actual = (
        bundle.get("bundle_version"), str(bundle.get("event_id")),
        str(bundle.get("cutoff_utc")), str(bundle.get("data_version")),
    )
    return actual == expected and bundle.get("lap_artifact") is not None and bundle.get("pit_artifact") is not None


@lru_cache(maxsize=16)
def _load_prediction_bundle(path_text: str, event_id: str, cutoff: str, data_version: str) -> dict[str, Any] | None:
    try:
        bundle = joblib.load(path_text)
    except Exception:
        return None
    return bundle if _bundle_metadata_matches(bundle, event_id, cutoff, data_version) else None


def load_prediction_bundle(event_id: str, cutoff: str, data_version: str | None = None) -> tuple[dict[str, Any] | None, bool]:
    """Load a prebuilt cutoff-safe model/profile bundle; this function never trains."""
    version = str(data_version or prediction_data_version(cutoff))
    path = _prediction_bundle_path(event_id, cutoff, version)
    before = _load_prediction_bundle.cache_info().hits
    bundle = _load_prediction_bundle(str(path), str(event_id), str(cutoff), version)
    hit = _load_prediction_bundle.cache_info().hits > before
    return bundle, hit


def prediction_bundle_status(event_id: str, cutoff: str, data_version: str | None = None) -> dict[str, Any]:
    """Validate a matching offline bundle and report missing/stale build lifecycle."""
    version = str(data_version or prediction_data_version(cutoff))
    bundle, _ = load_prediction_bundle(event_id, cutoff, version)
    base = {
        "event_id": str(event_id),
        "cutoff_utc": str(cutoff),
        "data_version": version,
        "bundle_version": ASOF_BUNDLE_VERSION,
        "available": bundle is not None,
    }
    if bundle is not None:
        return {**base, "status": "READY", "reason": None, "instruction": None, "stale": False}

    try:
        from startup_sync import bundle_lifecycle_status
        lifecycle = bundle_lifecycle_status(event_id, cutoff, version)
    except Exception:
        lifecycle = None
    if lifecycle and lifecycle.get("status") in {"BUILDING", "FAILED"}:
        return {
            **base, **lifecycle, "available": False,
            "instruction": f"Run python scripts/train_model.py --asof-event {event_id}" if lifecycle.get("status") == "FAILED" else None,
        }

    current_path = _prediction_bundle_path(event_id, cutoff, version)
    stale_bundle = None
    pattern = current_path.name.rsplit("_", 1)[0] + "_*.joblib"
    for candidate in current_path.parent.glob(pattern):
        if candidate == current_path:
            continue
        try:
            old = joblib.load(candidate)
        except Exception:
            continue
        if not isinstance(old, dict) or str(old.get("event_id")) != str(event_id) or str(old.get("cutoff_utc")) != str(cutoff):
            continue
        stale_bundle = old
        break
    if stale_bundle is not None:
        compatible = stale_bundle.get("bundle_version") == ASOF_BUNDLE_VERSION
        reason = "Bundle data version no longer matches the SQLite cutoff data." if compatible else "Bundle model/schema version is incompatible with the current application."
        return {**base, "status": "STALE", "reason": reason, "instruction": f"Run python scripts/train_model.py --asof-event {event_id}", "stale": True}
    return {**base, "status": "MISSING", "reason": "No validated offline bundle exists for this event cutoff and data version.", "instruction": f"Run python scripts/train_model.py --asof-event {event_id}", "stale": False}


def _latest_six_circuit_events(
    laps: pd.DataFrame,
    circuit: str,
    cutoff: str,
    *,
    exclude_event_id: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select laps from the latest six observed circuit races before cutoff."""
    if laps.empty or "Circuit" not in laps:
        return laps.iloc[0:0].copy(), []
    wanted_circuit = circuit_identity_key(circuit)
    circuit_rows = laps[laps["Circuit"].astype(str).map(circuit_identity_key).eq(wanted_circuit)].copy()
    if "EventDate" in circuit_rows:
        dates = pd.to_datetime(circuit_rows["EventDate"], errors="coerce", utc=True)
        cutoff_date = pd.to_datetime(str(cutoff)[:10], errors="coerce", utc=True)
        circuit_rows = circuit_rows[dates.notna() & dates.lt(cutoff_date)].copy()
    if exclude_event_id and "event_id" in circuit_rows:
        circuit_rows = circuit_rows[circuit_rows["event_id"].astype(str).ne(str(exclude_event_id))].copy()
    if circuit_rows.empty:
        return circuit_rows, []

    if "event_id" in circuit_rows and circuit_rows["event_id"].notna().any():
        event_keys = ["event_id"]
    else:
        event_keys = [column for column in ("Season", "Round", "Race") if column in circuit_rows]
        if not event_keys:
            return circuit_rows.iloc[0:0].copy(), []
    metadata = []
    for key_values, group in circuit_rows.groupby(event_keys, dropna=False, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {column: value for column, value in zip(event_keys, key_values)}
        for source, target in (("Season", "season"), ("Round", "round"), ("Race", "race"), ("Circuit", "circuit"), ("event_id", "event_id")):
            if source in group:
                row[target] = group[source].iloc[0]
        row["event_date"] = str(group["EventDate"].dropna().max())[:10] if "EventDate" in group and group["EventDate"].notna().any() else None
        if row.get("event_date") and row["event_date"] >= str(cutoff)[:10]:
            continue
        row["_key"] = tuple(row.get(column) for column in event_keys)
        metadata.append(row)
    metadata.sort(key=lambda row: (str(row.get("event_date") or ""), int(row.get("season") or 0), int(row.get("round") or 0)), reverse=True)
    chosen = metadata[:6]
    chosen_keys = {row["_key"] for row in chosen}
    mask = pd.Series(False, index=circuit_rows.index)
    for key in chosen_keys:
        key_mask = pd.Series(True, index=circuit_rows.index)
        for column, value in zip(event_keys, key):
            key_mask &= circuit_rows[column].eq(value)
        mask |= key_mask
    selected = circuit_rows.loc[mask].copy()
    for row in chosen:
        row.pop("_key", None)
    return selected, chosen


def _strategy_context(laps: pd.DataFrame, circuit: str, target_season: int, cutoff: str, exclude_event_id: str | None = None) -> dict[str, Any]:
    """Summarize six-race local history and recent global strategy priors offline."""
    season = pd.to_numeric(laps.get("Season", pd.Series(dtype=float)), errors="coerce")
    recent, circuit_events = _latest_six_circuit_events(laps, circuit, cutoff, exclude_event_id=exclude_event_id)
    recent_ids = [str(row["event_id"]) for row in circuit_events if row.get("event_id") is not None]
    global_recent = laps[season.ge(int(target_season) - 2)].copy()
    strategy_counts: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    driver_counts: dict[str, Counter[str]] = {}

    def collect(frame: pd.DataFrame, counts: Counter[str], by_driver: dict[str, Counter[str]] | None = None, *, local: bool = False) -> None:
        keys = [column for column in ("Season", "Round", "Race", "Driver") if column in frame]
        if not keys or "Compound" not in frame:
            return
        order = [column for column in ("LapNumber", "Stint") if column in frame]
        for values, group in frame.groupby(keys, dropna=True, sort=False):
            group = group.sort_values(order) if order else group
            if "Stint" in group and group["Stint"].notna().any():
                sequence = group.drop_duplicates("Stint", keep="first")["Compound"].astype(str).str.upper().tolist()
            else:
                sequence = group["Compound"].astype(str).str.upper().tolist()
                sequence = [value for index, value in enumerate(sequence) if index == 0 or value != sequence[index - 1]]
            sequence = [value for value in sequence if value in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}]
            if len(sequence) < 2:
                continue
            key = " → ".join(sequence)
            row_season = pd.to_numeric(
                pd.Series([group["Season"].iloc[0] if "Season" in group else target_season]),
                errors="coerce",
            ).iloc[0]
            if local:
                race_id = str(group["event_id"].iloc[0]) if "event_id" in group else None
                rank = next((index for index, event_row in enumerate(circuit_events) if race_id and str(event_row.get("event_id")) == race_id), None)
                season_weight = [1.0, .84, .70, .58, .48, .40][rank] if rank is not None else 1.0
            else:
                season_weight = {0: 1.0, 1: 0.68, 2: 0.46}.get(int(target_season) - int(row_season), 0.0) if pd.notna(row_season) else 0.0
            if not season_weight:
                continue
            counts[key] += season_weight
            if by_driver is not None:
                driver = str(group["Driver"].iloc[0]).upper()
                by_driver.setdefault(driver, Counter())[key] += season_weight

    collect(recent, strategy_counts, driver_counts, local=True)
    collect(global_recent, global_counts)
    pit_rows: list[dict[str, Any]] = []
    try:
        with connect() as connection:
            if recent_ids:
                marks = ",".join("?" for _ in recent_ids)
                pit_rows = [dict(row) for row in connection.execute(
                    "SELECT p.pit_lap,p.race_laps,p.driver,e.season,e.event_id "
                    "FROM pit_transitions p JOIN events e ON e.season=p.season AND e.round=p.round AND e.race=p.race "
                    f"WHERE e.event_id IN ({marks}) AND e.event_date<? AND p.pit_lap IS NOT NULL AND p.race_laps>0",
                    (*recent_ids, str(cutoff)[:10]),
                )]
    except Exception:
        pit_rows = []
    valid_pits = [row for row in pit_rows if row.get("race_laps") and float(row["race_laps"]) > 0]
    progresses = [float(row["pit_lap"]) / float(row["race_laps"]) for row in valid_pits]
    event_rank = {str(row.get("event_id")): index for index, row in enumerate(circuit_events)}
    rank_weights = [1.0, .84, .70, .58, .48, .40]
    progress_weights = [rank_weights[event_rank[str(row.get("event_id"))]] for row in valid_pits if str(row.get("event_id")) in event_rank]
    driver_progress: dict[str, list[float]] = {}
    for row in pit_rows:
        try:
            progress = float(row["pit_lap"]) / float(row["race_laps"])
            driver_progress.setdefault(str(row.get("driver", "")).upper(), []).append(progress)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return {
        "strategy_counts_circuit": dict(strategy_counts),
        "strategy_counts_recent_global": dict(global_counts),
        "driver_strategy_counts_circuit": {driver: dict(counts) for driver, counts in driver_counts.items()},
        "pit_progress_circuit": progresses,
        "pit_progress_weights": progress_weights,
        "driver_pit_progress_circuit": driver_progress,
        "recent_seasons": sorted({int(row["season"]) for row in circuit_events if row.get("season") is not None}),
        "circuit_history": {
            "races_requested": 6,
            "race_count": len(circuit_events),
            "races": circuit_events,
            "message": f"Only {len(circuit_events)} completed circuit races available" if len(circuit_events) < 6 else "Six most recent completed circuit races",
        },
    }


def build_prediction_bundle(
    event: dict[str, Any], cutoff: str, *, data_version: str | None = None, laps: pd.DataFrame | None = None
) -> Path:
    """Offline-only training and feature precompute for one event's strict cutoff."""
    version = str(data_version or prediction_data_version(cutoff))
    event_id = str(event["event_id"])
    full_laps = laps if laps is not None else laps_frame(before=cutoff)
    cutoff_laps = filter_laps_before_cutoff(full_laps, cutoff)
    season = int(event["season"])
    race = str(event["race"])
    if {"Season", "Race"}.issubset(cutoff_laps.columns):
        cutoff_laps = cutoff_laps[
            ~(pd.to_numeric(cutoff_laps["Season"], errors="coerce").eq(season)
              & cutoff_laps["Race"].astype(str).eq(race))
        ].copy()
    if "event_id" in cutoff_laps:
        cutoff_laps = cutoff_laps[cutoff_laps["event_id"].astype(str).ne(event_id)].copy()
    if cutoff_laps.empty:
        raise ValueError("No source lap rows remain before the requested artifact cutoff.")
    fingerprint = _frame_fingerprint(cutoff_laps)
    lap_artifact, pit_artifact, metrics = _asof_models(cutoff_laps, cutoff, fingerprint=fingerprint)
    summary, curve = _performance_profile(cutoff_laps, fingerprint, season)
    summary = _circuit_adjusted_performance(
        cutoff_laps, summary, str(event.get("circuit") or "Unknown"), season, fingerprint,
        cutoff=cutoff, exclude_event_id=event_id,
    )
    curve = _circuit_adjusted_curve(
        cutoff_laps, curve, str(event.get("circuit") or "Unknown"), season, fingerprint,
        cutoff=cutoff, exclude_event_id=event_id,
    )
    historical_context = _strategy_context(
        cutoff_laps, str(event.get("circuit") or "Unknown"), season, cutoff, exclude_event_id=event_id,
    )
    bundle = {
        "bundle_version": ASOF_BUNDLE_VERSION,
        "event_id": event_id,
        "cutoff_utc": str(cutoff),
        "data_version": version,
        "fingerprint": fingerprint,
        "lap_artifact": lap_artifact,
        "pit_artifact": pit_artifact,
        "metrics": metrics,
        "performance_summary": summary.to_dict(orient="records"),
        "performance_curve": curve.to_dict(orient="records"),
        "historical_context": historical_context,
        "circuit_history": historical_context.get("circuit_history", {}),
        "training_rows": int(len(cutoff_laps)),
        "training_event_count": int(cutoff_laps[["Season", "Race"]].drop_duplicates().shape[0]),
        "available_seasons": sorted(pd.to_numeric(cutoff_laps["Season"], errors="coerce").dropna().astype(int).unique().tolist()),
        "season_event_counts": {
            str(int(year)): int(group[["Season", "Race"]].drop_duplicates().shape[0])
            for year, group in cutoff_laps.groupby("Season", dropna=True)
        },
        "season_driver_counts": {
            str(int(year)): int(group["Driver"].nunique())
            for year, group in cutoff_laps.groupby("Season", dropna=True)
        },
        "training_completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path = _prediction_bundle_path(event_id, cutoff, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(bundle, temporary)
        written = joblib.load(temporary)
        if not _bundle_metadata_matches(written, event_id, cutoff, version):
            raise RuntimeError("The written model bundle failed event/cutoff/data-version/schema validation.")
        temporary.replace(path)
        _load_prediction_bundle.cache_clear()
        verified, _ = load_prediction_bundle(event_id, cutoff, version)
        if verified is None:
            raise RuntimeError("The atomically installed model bundle failed final validation.")
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _weighted_tyre_performance(laps: pd.DataFrame, target_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine per-season profiles with weights that favor the current technical era."""
    if laps.empty or "Season" not in laps:
        return analyze_tyre_performance(laps)
    weighted_summaries: list[pd.DataFrame] = []
    weighted_curves: list[pd.DataFrame] = []
    season_values = pd.to_numeric(laps["Season"], errors="coerce")
    for year, season_rows in laps.groupby(season_values, dropna=True, sort=True):
        age = int(target_season) - int(year)
        if age < 0 or age > 8:
            continue
        weight = 0.70 ** age
        summary, curve = analyze_tyre_performance(season_rows)
        summary = summary.copy()
        summary["SeasonWeight"] = weight
        weighted_summaries.append(summary)
        if not curve.empty:
            curve = curve.copy()
            curve["SeasonWeight"] = weight
            weighted_curves.append(curve)
    if not weighted_summaries:
        return analyze_tyre_performance(laps)

    combined = pd.concat(weighted_summaries, ignore_index=True)
    records: list[dict[str, Any]] = []
    for compound, group in combined.groupby("Compound", dropna=True, sort=False):
        effective_points = pd.to_numeric(group["DataPoints"], errors="coerce").fillna(0.0) * group["SeasonWeight"]
        record: dict[str, Any] = {
            "Compound": compound,
            "DataPoints": int(round(float(effective_points.sum()))),
            "StintsUsed": int(round(float(pd.to_numeric(group["StintsUsed"], errors="coerce").fillna(0).mul(group["SeasonWeight"]).sum()))),
        }
        for metric in ("CompoundOffset", "DegradationRate", "DegradationStd", "R2"):
            values = pd.to_numeric(group[metric], errors="coerce")
            source = group["OffsetSource"] if metric == "CompoundOffset" else group["DegradationSource"]
            eligible = source.eq("historical") if metric in {"CompoundOffset", "DegradationRate"} else group["DegradationSource"].eq("historical")
            valid = values.notna() & eligible & effective_points.gt(0)
            record[metric] = float(np.average(values[valid], weights=effective_points[valid])) if valid.any() else np.nan
        record["OffsetSource"] = "season-weighted historical" if pd.notna(record["CompoundOffset"]) else str(group["OffsetSource"].iloc[-1])
        record["DegradationSource"] = "season-weighted historical" if pd.notna(record["DegradationRate"]) else str(group["DegradationSource"].iloc[-1])
        records.append(record)
    summary = pd.DataFrame(records)
    if weighted_curves:
        curves = pd.concat(weighted_curves, ignore_index=True)
        curves["LapTime"] = pd.to_numeric(curves["LapTime"], errors="coerce")
        curves["WeightedLapTime"] = curves["LapTime"] * curves["SeasonWeight"]
        curve = curves.groupby(["Compound", "TyreLife"], as_index=False, dropna=True, sort=False).agg(
            WeightedLapTime=("WeightedLapTime", "sum"), SeasonWeight=("SeasonWeight", "sum")
        )
        curve["LapTime"] = curve["WeightedLapTime"] / curve["SeasonWeight"].replace(0, np.nan)
        curve = curve[["Compound", "TyreLife", "LapTime"]]
    else:
        curve = pd.DataFrame(columns=["Compound", "TyreLife", "LapTime"])
    return summary, curve


def _performance_profile(laps: pd.DataFrame, fingerprint: str, target_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_key = f"tyre-profile-v4:{fingerprint}:{int(target_season)}"
    cached = get_feature_cache(cache_key)
    if cached and isinstance(cached.get("summary"), list) and isinstance(cached.get("curve"), list):
        return pd.DataFrame(cached["summary"]), pd.DataFrame(cached["curve"])
    summary, curve = _weighted_tyre_performance(laps, int(target_season))
    put_feature_cache(cache_key, {
        "summary": summary.to_dict(orient="records"),
        "curve": curve.to_dict(orient="records"),
    })
    return summary, curve


def _circuit_adjusted_performance(
    laps: pd.DataFrame,
    summary: pd.DataFrame,
    circuit: str,
    target_season: int,
    fingerprint: str,
    *,
    cutoff: str | None = None,
    exclude_event_id: str | None = None,
) -> pd.DataFrame:
    """Blend the six latest completed circuit races with recent global F1 data."""
    circuit_key = hashlib.sha256(str(circuit).casefold().encode("utf-8")).hexdigest()[:12]
    cache_key = f"tyre-circuit-profile-v3:{fingerprint}:{target_season}:{circuit_key}"
    cached = get_feature_cache(cache_key)
    if cached and isinstance(cached.get("summary"), list):
        return pd.DataFrame(cached["summary"])

    output = summary.copy().set_index("Compound", drop=False)
    cutoff = str(cutoff or "9999-12-31")
    recent, circuit_events = _latest_six_circuit_events(laps, circuit, cutoff, exclude_event_id=exclude_event_id)
    output["CircuitRecentLaps"] = 0
    output["CircuitHistoryRaces"] = len(circuit_events)
    output["CircuitBlendWeight"] = 0.0
    output["CircuitDegradationRate"] = np.nan
    output["CircuitCompoundOffset"] = np.nan
    if recent.empty:
        output = output.reset_index(drop=True)
        put_feature_cache(cache_key, {"summary": output.to_dict(orient="records")})
        return output

    # Local observations are weighted by their measured sample size, not by
    # season age; this preserves useful history after a circuit calendar gap.
    local_summary, _ = analyze_tyre_performance(recent)
    local_by_compound = local_summary.set_index("Compound")
    for compound in output.index:
        if compound not in local_by_compound.index:
            continue
        local = local_by_compound.loc[compound]
        count = int(local.get("DataPoints", 0) or 0)
        weight = min(0.82, count / (count + 2200.0)) if count > 0 else 0.0
        output.loc[compound, "CircuitRecentLaps"] = count
        output.loc[compound, "CircuitBlendWeight"] = weight
        for metric, circuit_metric in (("CompoundOffset", "CircuitCompoundOffset"),
                                       ("DegradationRate", "CircuitDegradationRate")):
            local_value = pd.to_numeric(pd.Series([local.get(metric)]), errors="coerce").iloc[0]
            global_value = pd.to_numeric(pd.Series([output.loc[compound, metric]]), errors="coerce").iloc[0]
            local_degradation_source = str(local.get("DegradationSource", ""))
            is_measured_degradation = metric != "DegradationRate" or local_degradation_source == "historical"
            if pd.notna(local_value) and is_measured_degradation:
                output.loc[compound, circuit_metric] = float(local_value)
                output.loc[compound, metric] = float(local_value if pd.isna(global_value) else (1 - weight) * global_value + weight * local_value)
        local_spread = pd.to_numeric(pd.Series([local.get("DegradationStd")]), errors="coerce").iloc[0]
        if pd.notna(local_spread) and str(local.get("DegradationSource", "")) == "historical":
            output.loc[compound, "DegradationStd"] = float(local_spread)
            output.loc[compound, "DegradationSource"] = "recent circuit + global blend"
        elif count:
            output.loc[compound, "OffsetSource"] = "recent circuit + global blend"
    output = output.reset_index(drop=True)
    put_feature_cache(cache_key, {"summary": output.to_dict(orient="records")})
    return output


def _circuit_adjusted_curve(
    laps: pd.DataFrame,
    global_curve: pd.DataFrame,
    circuit: str,
    target_season: int,
    fingerprint: str,
    *,
    cutoff: str | None = None,
    exclude_event_id: str | None = None,
) -> pd.DataFrame:
    """Shrink age-specific local tyre pace from the same six races to global pace."""
    cache_key = f"tyre-circuit-curve-v2:{fingerprint}:{target_season}:{hashlib.sha256(str(circuit).casefold().encode()).hexdigest()[:12]}"
    cached = get_feature_cache(cache_key)
    if cached and isinstance(cached.get("curve"), list):
        return pd.DataFrame(cached["curve"])
    cutoff = str(cutoff or "9999-12-31")
    local, _events = _latest_six_circuit_events(laps, circuit, cutoff, exclude_event_id=exclude_event_id)
    if local.empty:
        return global_curve.copy()
    _, local_curve = analyze_tyre_performance(local)
    if local_curve.empty:
        return global_curve.copy()
    global_curve = global_curve.copy()
    local_curve = local_curve.copy()
    for frame in (global_curve, local_curve):
        frame["Compound"] = frame["Compound"].astype(str).str.upper()
        frame["TyreLife"] = pd.to_numeric(frame["TyreLife"], errors="coerce").round(0)
        frame["LapTime"] = pd.to_numeric(frame["LapTime"], errors="coerce")
    local_counts = (
        local.dropna(subset=["Compound", "TyreLife"])
        .assign(Compound=lambda frame: frame["Compound"].astype(str).str.upper(),
                TyreLife=lambda frame: pd.to_numeric(frame["TyreLife"], errors="coerce").round(0))
        .groupby(["Compound", "TyreLife"], dropna=True).size().to_dict()
    )
    combined = global_curve.set_index(["Compound", "TyreLife"])["LapTime"].to_dict() if not global_curve.empty else {}
    for row in local_curve.to_dict(orient="records"):
        key = (str(row["Compound"]), float(row["TyreLife"]))
        local_value = float(row["LapTime"])
        count = int(local_counts.get(key, 0))
        weight = min(0.78, count / (count + 55.0)) if count else 0.0
        global_value = combined.get(key)
        combined[key] = local_value if global_value is None or not np.isfinite(float(global_value)) else (1 - weight) * float(global_value) + weight * local_value
    result = pd.DataFrame([
        {"Compound": key[0], "TyreLife": key[1], "LapTime": value}
        for key, value in combined.items() if np.isfinite(float(value))
    ]).sort_values(["Compound", "TyreLife"]).reset_index(drop=True)
    put_feature_cache(cache_key, {"curve": result.to_dict(orient="records")})
    return result


def _degradation_slope_sd(
    laps: pd.DataFrame, summary: pd.DataFrame | None = None
) -> dict[str, float]:
    """Reuse the robust cross-stint spread in the cached tyre profile."""
    if summary is None:
        summary, _ = analyze_tyre_performance(laps)
    if "DegradationStd" not in summary:
        return {}
    spread: dict[str, float] = {}
    for row in summary.to_dict(orient="records"):
        value = pd.to_numeric(pd.Series([row.get("DegradationStd")]), errors="coerce").iloc[0]
        if pd.notna(value) and np.isfinite(float(value)):
            spread[str(row.get("Compound", "")).upper()] = float(value)
    return spread


def _strategy_degradation_sd(strategy: dict[str, Any], slope_sd: dict[str, float]) -> float:
    compounds = str(strategy["strategy"]).split(" → ")
    lengths = strategy["stint_lengths"]
    variance = 0.0
    for compound, length in zip(compounds, lengths):
        sigma = slope_sd.get(str(compound).upper())
        if sigma is not None:
            variance += (sigma * int(length) * (int(length) - 1) / 2.0) ** 2
    return float(np.sqrt(variance))


def _scenario_override(condition: str, race_info: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": condition,
        "impact": "HIGH" if condition == "WET" else "MEDIUM" if condition == "DAMP" else "LOW",
        "temperature": forecast.get("temperature"),
        "precipitation": forecast.get("precipitation"),
        "rain": forecast.get("rain"),
        "precipitation_probability": forecast.get("precipitation_probability"),
        "wind_speed": forecast.get("wind_speed"),
        "wind_direction": forecast.get("wind_direction"),
        "weather_certainty": "MEDIUM" if forecast.get("available") else "LOW",
        "source": "Open-Meteo forecast scenario" if forecast.get("available") else "forecast unavailable; dry baseline assumption",
        "reason": f"{condition} scenario generated from event-date forecast variables; scenario weights are heuristic and not calibrated.",
        "hourly": forecast.get("hourly", []),
    }


def run_pre_race_analysis(
    event: dict[str, Any] | pd.Series,
    driver: str,
    laps: pd.DataFrame,
    *,
    fetch_weather: bool = True,
    output_dir: str | Path | None = None,
    nomination_path: str | Path | None = None,
    request_get: Any = None,
    forecast_days: int = 16,
    weather_package_override: dict[str, Any] | None = None,
    precomputed_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a strict as-of strategy using an offline-trained artifact bundle."""

    target = dict(event)
    season = int(target["Season"])
    race = str(target["Race"])
    round_number = int(target.get("Round") or 0)
    race_start = str(target.get("CutoffTimeUTC") or f"{target['EventDate']}T00:00:00Z")
    cutoff = str(target.get("DataCutoffTimeUTC") or f"{target['EventDate']}T00:00:00Z")
    race_info = {
        "season": season,
        "round": round_number,
        "race": race,
        "official_name": target.get("OfficialEventName", race),
        "circuit": target.get("Circuit", "Unknown"),
        "location": target.get("Location", target.get("Circuit", "Unknown")),
        "country": target.get("Country", "Unknown"),
        "date": str(target.get("EventDate", "")),
        "laps": target.get("RaceLaps"),
        "latitude": target.get("Latitude"),
        "longitude": target.get("Longitude"),
    }
    if race_info["laps"] is None or pd.isna(race_info["laps"]):
        raise ValueError("Pre-race strategy needs a race distance. No source-backed or same-event historical estimate is available.")
    race_info["laps"] = int(race_info["laps"])

    cutoff_laps = filter_laps_before_cutoff(laps, cutoff)
    # Stronger than date filtering alone: never train on target-event rows even if
    # a source later changes EventDate granularity or contains duplicate metadata.
    cutoff_laps = cutoff_laps[
        ~(
            pd.to_numeric(cutoff_laps.get("Season"), errors="coerce").eq(season)
            & cutoff_laps.get("Race", pd.Series(index=cutoff_laps.index, dtype=str)).astype(str).eq(race)
        )
    ].copy()
    if cutoff_laps.empty:
        raise ValueError("No source lap rows remain before the strict pre-race cutoff.")
    if ((pd.to_numeric(cutoff_laps["Season"], errors="coerce").eq(season))
            & cutoff_laps["Race"].astype(str).eq(race)).any():
        raise AssertionError("Leakage guard failed: selected-event laps remained in the pre-race training set.")

    driver_code = str(driver).strip().upper()
    if not cutoff_laps["Driver"].astype(str).str.upper().eq(driver_code).any():
        raise ValueError(f"{driver_code} has no historical laps available before the selected race cutoff.")

    if not precomputed_artifact:
        raise RuntimeError(
            "No offline cutoff-safe model and feature bundle is ready for this event. "
            "Run scripts/train_model.py --asof-event <event_id> before requesting a prediction."
        )
    if str(precomputed_artifact.get("event_id")) != str(target.get("event_id")):
        raise RuntimeError("The saved prediction artifact belongs to a different event.")
    if str(precomputed_artifact.get("cutoff_utc")) != str(cutoff):
        raise RuntimeError("The saved prediction artifact does not match this race cutoff.")
    lap_artifact = precomputed_artifact.get("lap_artifact")
    pit_artifact = precomputed_artifact.get("pit_artifact")
    metrics = precomputed_artifact.get("metrics") or {}
    if lap_artifact is None or pit_artifact is None:
        raise RuntimeError("The saved cutoff-safe model bundle is incomplete. Rebuild it offline.")

    nomination = load_event_nomination(season, race, round_number, path=nomination_path) if nomination_path else load_event_nomination(season, race, round_number)
    inventory, inventory_source = allocation_for_event(nomination)
    weather_package = weather_package_override if weather_package_override is not None else (get_current_and_forecast_weather(race, race_info, request_get=request_get, forecast_days=forecast_days) if fetch_weather else {
        "current": {"available": False, "reason": "Weather lookup disabled."},
        "forecast": {"available": False, "reason": "Weather lookup disabled.", "hourly": []},
        "source": "disabled",
    })
    scenario_model = weather_scenario_probabilities(weather_package)
    weights = scenario_model.get("weights", {"DRY": 1.0})
    ranked_conditions = sorted(
        ((str(condition).upper(), float(probability)) for condition, probability in weights.items() if float(probability) > 0),
        key=lambda pair: pair[1], reverse=True,
    )
    if not ranked_conditions:
        ranked_conditions = [("DRY", 1.0)]
    scenarios = [ranked_conditions[0][0]]
    # One contingency is useful only when the forecast has meaningful ambiguity.
    if len(ranked_conditions) > 1 and ranked_conditions[0][1] < 0.80 and ranked_conditions[1][1] >= 0.20:
        scenarios.append(ranked_conditions[1][0])
    performance_summary = pd.DataFrame(precomputed_artifact.get("performance_summary") or [])
    performance_curve = pd.DataFrame(precomputed_artifact.get("performance_curve") or [])
    historical_context = precomputed_artifact.get("historical_context") or {}
    base_output = Path(output_dir) if output_dir else None
    if base_output is not None:
        base_output.mkdir(parents=True, exist_ok=True)

    scenario_results: list[dict[str, Any]] = []
    strategy_generation_ms = 0.0
    strategy_simulation_ms = 0.0
    for condition in scenarios:
        override = _scenario_override(condition, race_info, weather_package.get("forecast", {}))
        scenario_output = base_output / f"scenario_{condition.casefold()}" if base_output is not None else None
        response = run_analysis(
            race,
            driver_code,
            season=season,
            weather_override=override,
            fetch_weather=False,
            output_dir=scenario_output,
            analysis_mode="pre-race",
            cutoff_time=cutoff,
            event_info_override=race_info,
            training_laps=laps,
            lap_artifact_override=lap_artifact,
            pit_artifact_override=pit_artifact,
            metrics_override=metrics,
            inventory=inventory,
            performance_summary_override=performance_summary,
            precomputed_context=precomputed_artifact,
            historical_strategy_context=historical_context,
            persist_outputs=False,
        )
        result = response["result"]
        stage_timings = result.get("prediction_timings", {})
        strategy_generation_ms += float(stage_timings.get("strategy_generation_ms", 0.0))
        strategy_simulation_ms += float(stage_timings.get("strategy_simulation_ms", 0.0))
        pit_values = [int(value) for value in (result.get("pit_laps") or [])]
        strategy_name = str(result.get("strategy", ""))
        matching_candidates = [
            candidate for candidate in result.get("top_feasible_strategies", [])
            if str(candidate.get("Strategy", "")) == strategy_name
        ]
        best_score = float(result.get("strategy_score", 0.0))
        window_laps = [
            int(lap)
            for candidate in matching_candidates
            if float(candidate.get("StrategyScore", best_score)) <= best_score + 1.5
            for lap in (candidate.get("PitLaps") or [])
        ]
        if pit_values:
            window = {
                "start_lap": min(window_laps) if window_laps else pit_values[0],
                "end_lap": max(window_laps) if window_laps else pit_values[0],
                "central_lap": pit_values[0],
                "basis": "Nearby feasible candidates within 1.5 estimated seconds of the selected plan.",
            }
        else:
            window = {"start_lap": None, "end_lap": None, "central_lap": None, "basis": "No stop in the selected plan."}
        alternatives = []
        seen_strategies = {strategy_name}
        for candidate in result.get("top_feasible_strategies", []):
            candidate_name = str(candidate.get("Strategy", ""))
            if candidate_name and candidate_name not in seen_strategies:
                seen_strategies.add(candidate_name)
                alternatives.append({
                    "strategy": candidate_name,
                    "estimated_race_time": candidate.get("EstimatedRaceTime"),
                    "pit_laps": candidate.get("PitLaps", []),
                    "strategy_type": candidate.get("StrategyType"),
                    "score": candidate.get("StrategyScore"),
                    "score_breakdown": candidate.get("ScoreBreakdown", {}),
                })
            if len(alternatives) >= 4:
                break
        scenario_results.append({
            "condition": condition,
            "weight": float(weights.get(condition, 0.0)),
            "strategy": result["strategy"],
            "strategy_type": result["strategy_type"],
            "estimated_race_time": float(result["estimated_race_time"]),
            "strategy_score": float(result["strategy_score"]),
            "pit_laps": result["pit_laps"],
            "pit_window": window,
            "stint_lengths": result["stint_lengths"],
            "tyre_usage": result["tyre_usage"],
            "confidence_score": float(result["confidence_score"]),
            "candidate_count": int(result["feasible_candidate_count"]),
            "selected_event_laps_used_for_training": result["selected_event_laps_used_for_training"],
            "score_breakdown": result.get("strategy_score_breakdown", {}),
            "top_alternatives": alternatives,
            "reason": result.get("reason", ""),
        })

    total_weight = sum(max(0.0, row["weight"]) for row in scenario_results)
    if total_weight <= 0.0:
        scenario_results[0]["weight"] = 1.0
        total_weight = 1.0
    for row in scenario_results:
        row["weight"] = max(0.0, row["weight"]) / total_weight
    slope_sd = _degradation_slope_sd(cutoff_laps, performance_summary)
    for row in scenario_results:
        row["degradation_sensitivity_sd_seconds"] = _strategy_degradation_sd(row, slope_sd)
    values = np.asarray([row["estimated_race_time"] for row in scenario_results], dtype=float)
    scenario_weights = np.asarray([row["weight"] for row in scenario_results], dtype=float)
    expected = float(np.average(values, weights=scenario_weights))
    variance = float(np.average((values - expected) ** 2, weights=scenario_weights))
    monte_carlo_runs = 500
    rng = np.random.default_rng(RANDOM_STATE)
    draw_indices = rng.choice(len(scenario_results), size=monte_carlo_runs, p=scenario_weights)
    simulated_times = np.asarray([scenario_results[index]["estimated_race_time"] for index in draw_indices], dtype=float)
    lap_mae = metrics.get("lap_time", {}).get("MAE")
    if lap_mae is not None and np.isfinite(float(lap_mae)):
        # Under an independent, approximately normal per-lap residual assumption,
        # sigma ≈ MAE*sqrt(pi/2), then independent lap errors aggregate by sqrt(N).
        model_sigma = float(lap_mae) * np.sqrt(np.pi / 2.0) * np.sqrt(race_info["laps"])
        simulated_times += rng.normal(0.0, model_sigma, size=monte_carlo_runs)
    else:
        model_sigma = None
    degradation_sigma_by_scenario = np.asarray([
        scenario_results[index]["degradation_sensitivity_sd_seconds"] for index in draw_indices
    ], dtype=float)
    if np.any(degradation_sigma_by_scenario > 0.0):
        simulated_times += rng.normal(0.0, degradation_sigma_by_scenario, size=monte_carlo_runs)
    mc_summary = {
        "simulations": monte_carlo_runs,
        "seed": RANDOM_STATE,
        "mean_race_time_seconds": float(np.mean(simulated_times)),
        "standard_deviation_seconds": float(np.std(simulated_times, ddof=1)),
        "percentile_05_seconds": float(np.percentile(simulated_times, 5)),
        "percentile_50_seconds": float(np.percentile(simulated_times, 50)),
        "percentile_95_seconds": float(np.percentile(simulated_times, 95)),
        "lap_model_error_sigma_seconds": model_sigma,
        "degradation_uncertainty_source": "robust cross-stint slope spread (1.4826 × median absolute deviation), using plausible slopes from 0.002 to 0.20 s/lap and at least three historical stints per compound",
        "degradation_compounds_with_measured_spread": sorted(slope_sd),
        "assumptions": "weather scenario sampled by heuristic weights; per-lap model errors treated as independent normal residuals; degradation slope error uses a robust spread estimate from plausible historical cross-stint slopes",
    }
    strategy_weights: dict[str, float] = {}
    for row in scenario_results:
        strategy_weights[row["strategy"]] = strategy_weights.get(row["strategy"], 0.0) + row["weight"]
    expected_scenario = max(scenario_results, key=lambda row: row["weight"])
    dry_scenario = next((row for row in scenario_results if row["condition"] == "DRY"), expected_scenario)
    contingency = {
        row["condition"]: {
            "strategy": row["strategy"],
            "pit_laps": row["pit_laps"],
            "stint_lengths": row["stint_lengths"],
            "scenario_weight": round(row["weight"], 5),
        }
        for row in scenario_results
    }
    forecast_available = bool(weather_package.get("forecast", {}).get("available"))
    if forecast_available:
        wet_share = sum(row["weight"] for row in scenario_results if row["condition"] == "WET")
        damp_share = sum(row["weight"] for row in scenario_results if row["condition"] == "DAMP")
        contingency_rules = [
            f"Keep {dry_scenario['strategy']} as the dry-plan baseline."
        ]
        for row in scenario_results:
            if row["condition"] in {"DAMP", "WET"}:
                contingency_rules.append(
                    f"If conditions match the {row['condition'].lower()} scenario (heuristic weight {row['weight']:.0%}), switch to {row['strategy']}."
                )
        if not damp_share and not wet_share:
            contingency_rules.append("No damp or wet contingency was generated because the event-date forecast assigned no rain scenario weight.")
    else:
        contingency_rules = ["Forecast unavailable. Use the dry baseline as an explicit assumption and refresh closer to race day."]

    result = {
        "analysis_mode": "PRE-RACE",
        "season": season,
        "round": round_number,
        "race": race,
        "event_status": target.get("EventStatus", "UPCOMING"),
        "official_event_name": race_info["official_name"],
        "circuit": race_info["circuit"],
        "country": race_info["country"],
        "race_date": race_info["date"],
        "race_distance": race_info["laps"],
        "race_distance_source": target.get("RaceLapsSource", "source metadata"),
        "driver": driver_code,
        "selected_race_start_time_utc": pd.to_datetime(race_start, utc=True).isoformat(),
        "training_cutoff_time_utc": pd.to_datetime(cutoff, utc=True).isoformat(),
        "training_cutoff_policy": f"{target.get('RaceStartTimeSource', 'race date only; conservative midnight UTC cutoff')}; lap history has date precision only, so training excludes the full event date",
        "training_rows": int(precomputed_artifact.get("training_rows", len(cutoff_laps))),
        "training_event_count": int(precomputed_artifact.get("training_event_count", cutoff_laps[["Season", "Race"]].drop_duplicates().shape[0])),
        "driver_history_rows": int(len(cutoff_laps)),
        "model_training_rows": int(precomputed_artifact.get("training_rows", 0)),
        "model_training_event_count": int(precomputed_artifact.get("training_event_count", 0)),
        "models_loaded_from_offline_artifact": True,
        "selected_event_laps_used_for_training": False,
        "models_trained_only_on_pre_cutoff_data": True,
        "model_metrics": metrics,
        "nomination": nomination,
        "inventory": inventory,
        "inventory_source": inventory_source,
        "weather": weather_package,
        "weather_scenarios": scenario_model,
        "scenarios": scenario_results,
        "expected_strategy": expected_scenario,
        "strategy_alternatives": expected_scenario.get("top_alternatives", []),
        "strategy_score_breakdown": expected_scenario.get("score_breakdown", {}),
        "track_specific_learning": {
            "circuit": str(race_info.get("circuit") or "Unknown"),
            "recent_lap_samples": int(pd.to_numeric(performance_summary.get("CircuitRecentLaps", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "circuit_races_used": int(pd.to_numeric(performance_summary.get("CircuitHistoryRaces", pd.Series(dtype=float)), errors="coerce").fillna(0).max()),
            "largest_compound_blend_weight": float(pd.to_numeric(performance_summary.get("CircuitBlendWeight", pd.Series(dtype=float)), errors="coerce").fillna(0).max()),
            "recent_seasons": ", ".join(str(year) for year in historical_context.get("recent_seasons", [])) or "none available",
            "circuit_history": precomputed_artifact.get("circuit_history", historical_context.get("circuit_history", {})),
            "global_context": "the six latest completed circuit races are shrunk toward recent all-circuit history according to local sample size",
        },
        "primary_strategy": dry_scenario,
        "robust_strategy": {
            "type": "adaptive contingency policy",
            "by_condition": contingency,
            "rules": contingency_rules,
            "interpretation": "Scenario-best plans are evaluated separately by condition. Expected/worst/best times summarize this adaptive policy, not one fixed tyre plan held unchanged across every weather state.",
        },
        "robustness": {
            "expected_race_time_seconds": expected,
            "best_scenario_time_seconds": float(values.min()),
            "worst_scenario_time_seconds": float(values.max()),
            "scenario_standard_deviation_seconds": float(np.sqrt(variance)),
            "scenario_variance_seconds_squared": variance,
            "monte_carlo": mc_summary,
            "strategy_stability": {"scenario_weight_by_strategy": strategy_weights, "most_frequent_plan_share": max(strategy_weights.values()) if strategy_weights else 0.0},
            "metric_scope": "best feasible plan per included weather scenario; adaptive-policy summary",
        },
        "confidence_score": float(np.average([row["confidence_score"] for row in scenario_results], weights=scenario_weights)),
        "confidence_level": "HIGH" if np.average([row["confidence_score"] for row in scenario_results], weights=scenario_weights) >= 75 else "MEDIUM" if np.average([row["confidence_score"] for row in scenario_results], weights=scenario_weights) >= 50 else "LOW",
        "confidence_is_win_probability": False,
        "historical_strategy_context": historical_context,
        "prediction_timings": {
            "strategy_generation_ms": round(strategy_generation_ms, 2),
            "strategy_simulation_ms": round(strategy_simulation_ms, 2),
        },
    }
    if base_output is not None:
        (base_output / "pre_race_result.json").write_text(json.dumps(_record_for_json(result), indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(scenario_results).to_csv(base_output / "pre_race_scenarios.csv", index=False)
    return {"result": result, "scenarios": pd.DataFrame(scenario_results), "output_dir": base_output}
