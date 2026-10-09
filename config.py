"""Shared paths and modeling assumptions for the F1 strategy project."""

import os
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SEASON_DATA_DIR = RAW_DATA_DIR / "fastf1"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
PRE_RACE_MODEL_DIR = MODEL_DIR / "pre_race"
DOCS_DIR = PROJECT_ROOT / "docs"
FASTF1_CACHE_DIR = PROJECT_ROOT / "cache" / "fastf1"

DEFAULT_SEASON_START = int(os.getenv("F1_SEASON_START", "2019"))
DEFAULT_SEASON_END = int(os.getenv("F1_SEASON_END", "2026"))
LATEST_COMPLETED_CALENDAR_YEAR = date.today().year

# New multi-season paths; legacy paths remain available for existing project data.
RAW_LAP_DATA_PATH = RAW_DATA_DIR / "f1_laps_2019_2026.csv"
CLEAN_LAP_DATA_PATH = PROCESSED_DATA_DIR / "all_seasons_laps_clean.csv"
LEGACY_RAW_LAP_DATA_PATH = RAW_DATA_DIR / "2024_multi_race_laps.csv"
LEGACY_CLEAN_LAP_DATA_PATH = PROCESSED_DATA_DIR / "2024_multi_race_laps_clean.csv"
DATA_COVERAGE_PATH = PROCESSED_DATA_DIR / "data_coverage.json"
DOWNLOAD_STATUS_PATH = PROCESSED_DATA_DIR / "download_status.csv"
EVENT_CATALOG_PATH = PROCESSED_DATA_DIR / "event_catalog.csv"
COORDINATE_CACHE_PATH = PROJECT_ROOT / "cache" / "location_coordinates.json"
WEATHER_CACHE_DIR = PROJECT_ROOT / "cache" / "historical_weather"
FORECAST_CACHE_DIR = PROJECT_ROOT / "cache" / "forecast_weather"
PIT_TRAINING_DATA_PATH = PROCESSED_DATA_DIR / "pit_strategy_training_data.csv"
STRATEGY_RESULTS_PATH = PROCESSED_DATA_DIR / "strategy_simulation_results.csv"
WEATHER_ANALYSIS_PATH = PROCESSED_DATA_DIR / "weather_analysis.csv"
EVENT_TYRE_NOMINATIONS_PATH = PROCESSED_DATA_DIR / "event_tyre_nominations.csv"
TYRE_ALLOCATIONS_PATH = PROCESSED_DATA_DIR / "tyre_allocations.csv"
EVENT_METADATA_OVERRIDE_PATH = PROCESSED_DATA_DIR / "event_metadata_overrides.csv"
CALENDAR_UPDATES_PATH = PROCESSED_DATA_DIR / "calendar_updates.csv"
HISTORICAL_CIRCUIT_LAPS_PATH = RAW_DATA_DIR / "historical_circuit_laps.csv"
HISTORICAL_CIRCUIT_PIT_STOPS_PATH = RAW_DATA_DIR / "historical_circuit_pit_stops.csv"
DATA_QUALITY_JSON_PATH = PROCESSED_DATA_DIR / "data_quality_report.json"
DATA_QUALITY_CSV_PATH = PROCESSED_DATA_DIR / "data_quality_report.csv"
BACKEND_RESULT_PATH = PROCESSED_DATA_DIR / "backend_result.json"
TYRE_PERFORMANCE_PATH = PROCESSED_DATA_DIR / "tyre_performance.csv"
TYRE_DEGRADATION_PATH = PROCESSED_DATA_DIR / "tyre_degradation.csv"
TYRE_MODEL_RESULTS_PATH = PROCESSED_DATA_DIR / "tyre_model_results.csv"
MODEL_METRICS_PATH = PROCESSED_DATA_DIR / "model_metrics.json"
LAP_TIME_MODEL_PATH = MODEL_DIR / "lap_time_model.pkl"
PIT_STOP_MODEL_PATH = MODEL_DIR / "pit_stop_model.pkl"


RACES = {
    "Bahrain Grand Prix": {
        "laps": 57,
        "circuit": "Sakhir",
        "location": "Sakhir, Bahrain",
        "latitude": 26.0325,
        "longitude": 50.5106,
        "date": "2024-03-02",
    },
    "Emilia-Romagna Grand Prix": {
        "laps": 63,
        "circuit": "Imola",
        "location": "Imola, Italy",
        "latitude": 44.3439,
        "longitude": 11.7167,
        "date": "2024-05-19",
    },
    "British Grand Prix": {
        "laps": 52,
        "circuit": "Silverstone",
        "location": "Silverstone, United Kingdom",
        "latitude": 52.0786,
        "longitude": -1.0169,
        "date": "2024-07-07",
    },
    "Belgian Grand Prix": {
        "laps": 44,
        "circuit": "Spa-Francorchamps",
        "location": "Spa-Francorchamps, Belgium",
        "latitude": 50.4372,
        "longitude": 5.9714,
        "date": "2024-07-28",
    },
    "Italian Grand Prix": {
        "laps": 53,
        "circuit": "Monza",
        "location": "Monza, Italy",
        "latitude": 45.6156,
        "longitude": 9.2811,
        "date": "2024-09-01",
    },
}

SLICK_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
ALL_COMPOUNDS = SLICK_COMPOUNDS + WET_COMPOUNDS

# Baseline available slick sets for a standard, non-Sprint weekend. Actual
# per-driver remaining inventory depends on sets used throughout the weekend.
TYRE_SET_LIMITS = {"SOFT": 8, "MEDIUM": 3, "HARD": 2}

# Separate illustrative wet-set limits; these are modeling assumptions, not
# a claim about a particular driver's remaining FIA/Pirelli inventory.
WET_TYRE_SET_LIMITS = {"INTERMEDIATE": 4, "WET": 4}

TYRE_MIN_AGE = {"SOFT": 8, "MEDIUM": 10, "HARD": 12}
TYRE_MAX_AGE = {"SOFT": 25, "MEDIUM": 35, "HARD": 45}
WET_TYRE_MIN_AGE = {"INTERMEDIATE": 8, "WET": 8}
WET_TYRE_MAX_AGE = {"INTERMEDIATE": 35, "WET": 35}
MIN_STINT = 10
PIT_LOSS = 22.0

FALLBACK_DEGRADATION = {"SOFT": 0.085, "MEDIUM": 0.055, "HARD": 0.030}
FALLBACK_OFFSET = {"SOFT": -0.90, "MEDIUM": 0.00, "HARD": 0.70}

RANDOM_STATE = 42
MIN_LAP_MODEL_ROWS = 50
MIN_PIT_MODEL_ROWS = 20


def ensure_project_directories() -> None:
    """Create only the runtime directories used by the application."""

    for directory in (PROCESSED_DATA_DIR, MODEL_DIR, PRE_RACE_MODEL_DIR, DOCS_DIR, SEASON_DATA_DIR, FASTF1_CACHE_DIR, WEATHER_CACHE_DIR, FORECAST_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configured_seasons() -> tuple[int, ...]:
    """Return the configured season range, capped at the current calendar year."""

    start = max(2018, DEFAULT_SEASON_START)
    end = min(DEFAULT_SEASON_END, LATEST_COMPLETED_CALENDAR_YEAR)
    if start > end:
        return ()
    return tuple(range(start, end + 1))
