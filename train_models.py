"""Train and persist leak-aware Random Forest models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    CLEAN_LAP_DATA_PATH,
    LAP_TIME_MODEL_PATH,
    MIN_LAP_MODEL_ROWS,
    MIN_PIT_MODEL_ROWS,
    MODEL_METRICS_PATH,
    PIT_STOP_MODEL_PATH,
    PIT_TRAINING_DATA_PATH,
    RANDOM_STATE,
    ensure_project_directories,
)
from data_preparation import prepare_data


LAP_CATEGORICAL = ["Season", "Race", "Circuit", "Driver", "Team", "Compound"]
LAP_NUMERIC = ["LapNumber", "TyreLife", "RaceProgress"]
PIT_CATEGORICAL = ["Season", "Race", "Circuit", "Driver", "Team", "PreviousTyre"]
PIT_NUMERIC = ["PreviousTyreAge", "CurrentLap", "RaceProgress"]


def _preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )


def _group_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    classification: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Create a deterministic group holdout without row-level leakage."""

    if groups.nunique() < 2 or len(features) < 10:
        return None
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=RANDOM_STATE)
    for train_idx, test_idx in splitter.split(features, target, groups):
        if len(train_idx) < 2 or len(test_idx) < 2:
            continue
        if classification and target.iloc[train_idx].nunique() < 2:
            continue
        return train_idx, test_idx
    return None


def _temporal_holdout(features: pd.DataFrame, target: pd.Series, seasons: pd.Series,
                      *, classification: bool = False) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Hold out the latest non-current season; fall back to grouped events if needed."""

    year_values = pd.to_numeric(seasons, errors="coerce")
    years = sorted(year_values.dropna().astype(int).unique().tolist())
    completed_years = [year for year in years if year < pd.Timestamp.now().year]
    candidate_years = completed_years or years
    for held_out in reversed(candidate_years):
        train_idx = np.flatnonzero(year_values.to_numpy() < held_out)
        test_idx = np.flatnonzero(year_values.to_numpy() == held_out)
        if len(train_idx) < 10 or len(test_idx) < 10:
            continue
        if classification and target.iloc[train_idx].nunique() < 2:
            continue
        return train_idx, test_idx, int(held_out)
    return None


def _lap_features(laps: pd.DataFrame) -> pd.DataFrame:
    data = laps.copy()
    for column, fallback in (("Season", 2024), ("Circuit", "Unknown"), ("Team", "Unknown")):
        if column not in data:
            data[column] = fallback
    for column in LAP_CATEGORICAL:
        data[column] = data[column].fillna("Unknown").astype(str)
    race_laps = pd.to_numeric(data.get("RaceLaps"), errors="coerce") if "RaceLaps" in data else pd.Series(np.nan, index=data.index)
    inferred = data.groupby(["Season", "Race"])["LapNumber"].transform("max")
    race_laps = race_laps.fillna(inferred).clip(lower=1)
    features = data[LAP_CATEGORICAL + ["LapNumber", "TyreLife"]].copy()
    features["RaceProgress"] = (pd.to_numeric(data["LapNumber"], errors="coerce") / race_laps).fillna(0.5)
    return features


def train_lap_time_model(laps: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    required = LAP_CATEGORICAL + ["LapNumber", "TyreLife", "LapTime"]
    valid = laps.dropna(subset=required).copy()
    if len(valid) < MIN_LAP_MODEL_ROWS or valid["Compound"].nunique() < 2:
        return None, {"available": False, "reason": "Insufficient clean lap data."}

    features = _lap_features(valid)
    target = valid["LapTime"].astype(float)
    group_columns = [column for column in ("Season", "Race", "Driver", "Stint") if column in valid]
    stint_groups = (
        valid[group_columns].astype(str).agg("|".join, axis=1)
        if "Stint" in valid.columns
        else valid[[column for column in ("Season", "Race", "Driver") if column in valid]].astype(str).agg("|".join, axis=1)
    )
    metrics: dict[str, Any] = {
        "available": True,
        "training_rows": int(len(valid)),
        "evaluation_method": "temporal season holdout when multiple seasons are available; final model refit on all available data",
    }
    temporal = _temporal_holdout(features, target, valid.get("Season", pd.Series(2024, index=valid.index)))
    if temporal is not None:
        train_idx, test_idx, held_out_season = temporal
        split = (train_idx, test_idx)
        metrics["held_out_season"] = held_out_season
    else:
        split = _group_holdout(features, target, stint_groups)
        metrics["evaluation_method"] = "race-driver-stint group holdout; no laps from a stint cross train and test"
    if split is not None:
        train_idx, test_idx = split
        evaluation_model = Pipeline(
            [
                ("preprocessor", _preprocessor(LAP_CATEGORICAL, LAP_NUMERIC)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=60,
                        max_depth=10,
                        min_samples_leaf=6,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        evaluation_model.fit(features.iloc[train_idx], target.iloc[train_idx])
        predictions = evaluation_model.predict(features.iloc[test_idx])
        truth = target.iloc[test_idx].to_numpy(dtype=float)
        median_baseline = np.repeat(float(target.iloc[train_idx].median()), len(test_idx))
        metrics.update(
            {
                "MAE": float(mean_absolute_error(truth, predictions)),
                "RMSE": float(np.sqrt(mean_squared_error(truth, predictions))),
                "R2": float(r2_score(truth, predictions)) if len(truth) >= 2 else None,
                "baseline": {
                    "name": "training-set historical median lap time",
                    "MAE": float(mean_absolute_error(truth, median_baseline)),
                    "RMSE": float(np.sqrt(mean_squared_error(truth, median_baseline))),
                    "R2": float(r2_score(truth, median_baseline)) if len(truth) >= 2 else None,
                },
                "test_rows": int(len(test_idx)),
                "test_races": sorted(valid.iloc[test_idx]["Race"].unique().tolist()),
                "test_groups": int(stint_groups.iloc[test_idx].nunique()),
                "test_seasons": sorted(valid.iloc[test_idx].get("Season", pd.Series(dtype=int)).unique().tolist()),
            }
        )
    else:
        metrics.update(
            {"MAE": None, "RMSE": None, "R2": None, "evaluation_method": "unavailable: fewer than two groups"}
        )

    final_model = Pipeline(
        [
            ("preprocessor", _preprocessor(LAP_CATEGORICAL, LAP_NUMERIC)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=10,
                    min_samples_leaf=6,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_model.fit(features, target)
    artifact = {
        "model": final_model,
        "categorical_features": LAP_CATEGORICAL,
        "numeric_features": LAP_NUMERIC,
        "metrics": metrics,
        "training_rows": int(len(valid)),
        "model_type": "RandomForestRegressor",
    }
    return artifact, metrics


def train_pit_stop_model(pit_data: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    required = PIT_CATEGORICAL + PIT_NUMERIC + ["NewTyre"]
    missing = set(required) - set(pit_data.columns)
    if missing:
        return None, {"available": False, "reason": f"Missing columns: {', '.join(sorted(missing))}"}
    valid = pit_data.dropna(subset=required).copy()
    valid["NewTyre"] = valid["NewTyre"].astype(str).str.upper().str.strip()
    valid = valid[valid["NewTyre"].ne("")]
    if len(valid) < MIN_PIT_MODEL_ROWS or valid["NewTyre"].nunique() < 2:
        return None, {"available": False, "reason": "Insufficient observed pit transitions."}

    for column, fallback in (("Season", 2024), ("Circuit", "Unknown"), ("Team", "Unknown"),
                             ("CurrentLap", None), ("RaceProgress", None)):
        if column not in valid:
            valid[column] = fallback
    if valid["CurrentLap"].isna().all():
        valid["CurrentLap"] = valid["PitLap"]
    if valid["RaceProgress"].isna().all():
        distances = pd.to_numeric(valid.get("RaceLaps"), errors="coerce") if "RaceLaps" in valid else pd.Series(np.nan, index=valid.index)
        distances = distances.fillna(valid.groupby(["Season", "Race"])["PitLap"].transform("max")).clip(lower=1)
        valid["RaceProgress"] = valid["CurrentLap"] / distances
    for column in PIT_CATEGORICAL:
        valid[column] = valid[column].fillna("Unknown").astype(str)
    features = valid[PIT_CATEGORICAL + PIT_NUMERIC].copy()
    target = valid["NewTyre"]
    temporal = _temporal_holdout(features, target, valid.get("Season", pd.Series(2024, index=valid.index)), classification=True)
    if temporal is not None:
        train_idx, test_idx, held_out_season = temporal
        split = (train_idx, test_idx)
        evaluation_method = "temporal season holdout; final model refit on all available data"
    else:
        event_groups = valid[[column for column in ("Season", "Race") if column in valid]].astype(str).agg("|".join, axis=1)
        split = _group_holdout(features, target, event_groups, classification=True)
        held_out_season = None
        evaluation_method = "race-group holdout; metrics are from unseen races"
    metrics: dict[str, Any] = {
        "available": True,
        "training_rows": int(len(valid)),
        "classes": sorted(target.unique().tolist()),
        "evaluation_method": evaluation_method,
    }
    if held_out_season is not None:
        metrics["held_out_season"] = int(held_out_season)
    if split is not None:
        train_idx, test_idx = split
        evaluation_model = Pipeline(
            [
                ("preprocessor", _preprocessor(PIT_CATEGORICAL, PIT_NUMERIC)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=8,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        evaluation_model.fit(features.iloc[train_idx], target.iloc[train_idx])
        predictions = evaluation_model.predict(features.iloc[test_idx])
        truth = target.iloc[test_idx]
        majority_class = target.iloc[train_idx].mode().iloc[0]
        baseline_predictions = np.repeat(majority_class, len(test_idx))
        matrix = confusion_matrix(truth, predictions, labels=sorted(target.unique().tolist()))
        metrics.update(
            {
                "accuracy": float(accuracy_score(truth, predictions)),
                "baseline_accuracy": float(accuracy_score(truth, baseline_predictions)),
                "baseline_name": "most common training-set next tyre",
                "confusion_matrix": matrix.tolist(),
                "confusion_matrix_labels": sorted(target.unique().tolist()),
                "test_rows": int(len(test_idx)),
                "test_races": sorted(valid.iloc[test_idx]["Race"].unique().tolist()),
                "test_seasons": sorted(valid.iloc[test_idx].get("Season", pd.Series(dtype=int)).unique().tolist()),
            }
        )
    else:
        metrics.update(
            {"accuracy": None, "evaluation_method": "unavailable: fewer than two races or classes"}
        )

    final_model = Pipeline(
        [
            ("preprocessor", _preprocessor(PIT_CATEGORICAL, PIT_NUMERIC)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=8,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_model.fit(features, target)
    artifact = {
        "model": final_model,
        "categorical_features": PIT_CATEGORICAL,
        "numeric_features": PIT_NUMERIC,
        "metrics": metrics,
        "training_rows": int(len(valid)),
        "model_type": "RandomForestClassifier",
    }
    return artifact, metrics


def train_models(*, force_preparation: bool = False) -> dict[str, Any]:
    ensure_project_directories()
    if force_preparation or not CLEAN_LAP_DATA_PATH.exists() or not PIT_TRAINING_DATA_PATH.exists():
        prepared = prepare_data()
        laps, pit_data = prepared["laps"], prepared["pit_training"]
    else:
        try:
            laps = pd.read_csv(CLEAN_LAP_DATA_PATH)
            pit_data = pd.read_csv(PIT_TRAINING_DATA_PATH)
        except Exception:
            prepared = prepare_data()
            laps, pit_data = prepared["laps"], prepared["pit_training"]
        if set(PIT_CATEGORICAL + PIT_NUMERIC + ["NewTyre"]) - set(pit_data.columns):
            prepared = prepare_data()
            laps, pit_data = prepared["laps"], prepared["pit_training"]

    lap_artifact, lap_metrics = train_lap_time_model(laps)
    pit_artifact, pit_metrics = train_pit_stop_model(pit_data)

    if lap_artifact is not None:
        joblib.dump(lap_artifact, LAP_TIME_MODEL_PATH)
    elif LAP_TIME_MODEL_PATH.exists():
        LAP_TIME_MODEL_PATH.unlink()
    if pit_artifact is not None:
        joblib.dump(pit_artifact, PIT_STOP_MODEL_PATH)
    elif PIT_STOP_MODEL_PATH.exists():
        PIT_STOP_MODEL_PATH.unlink()

    metrics = {"lap_time": lap_metrics, "pit_stop": pit_metrics}
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model_artifact(path: str | Path) -> dict[str, Any] | None:
    """Load a project-generated model artifact, returning None if unavailable."""

    model_path = Path(path)
    if not model_path.exists():
        return None
    try:
        loaded = joblib.load(model_path)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) and "model" in loaded else None


if __name__ == "__main__":
    summary = train_models(force_preparation=True)
    print(json.dumps(summary, indent=2))
    print(f"Model files written under: {LAP_TIME_MODEL_PATH.parent}")
