"""Command-line entry point for the modular strategy engine.

The original interactive prototype is retained below for reference, but the
supported executable path now dispatches to the tested, constraint-aware
engine.  The dashboard calls the engine directly.
"""

if __name__ == "__main__":
    from strategy_engine import main as _strategy_engine_main

    raise SystemExit(_strategy_engine_main())

import os
import sys
import json
import argparse
import warnings
import requests
import numpy as np
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

# ============================================================
# OPTIONAL COMMAND-LINE INPUT
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--race",
    type=str,
    default=None
)

parser.add_argument(
    "--driver",
    type=str,
    default=None
)

args, _ = parser.parse_known_args()

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ============================================================
# F1 PRE-RACE STRATEGY SYSTEM
# ============================================================

print("\n========================================")
print("      F1 PRE-RACE STRATEGY SYSTEM")
print("========================================")


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/raw/2024_multi_race_laps.csv"
PIT_DATA_PATH = "data/processed/pit_strategy_training_data.csv"
RESULT_PATH = "data/processed/strategy_simulation_results.csv"
WEATHER_PATH = "data/processed/weather_analysis.csv"

os.makedirs("data/processed", exist_ok=True)


RACES = {

    "Bahrain Grand Prix": {
        "laps": 57,
        "circuit": "Sakhir, Bahrain",
        "lat": 26.0325,
        "lon": 50.5106,
        "date": "2024-03-02"
    },

    "Emilia-Romagna Grand Prix": {
        "laps": 63,
        "circuit": "Imola, Italy",
        "lat": 44.3439,
        "lon": 11.7167,
        "date": "2024-05-19"
    },

    "British Grand Prix": {
        "laps": 52,
        "circuit": "Silverstone, UK",
        "lat": 52.0786,
        "lon": -1.0169,
        "date": "2024-07-07"
    },

    "Belgian Grand Prix": {
        "laps": 44,
        "circuit": "Spa-Francorchamps, Belgium",
        "lat": 50.4372,
        "lon": 5.9714,
        "date": "2024-07-28"
    },

    "Italian Grand Prix": {
        "laps": 53,
        "circuit": "Monza, Italy",
        "lat": 45.6156,
        "lon": 9.2811,
        "date": "2024-09-01"
    }
}


DRY_COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD"
]

PIT_LOSS = 22.0

RANDOM_STATE = 42


# ============================================================
# REALISTIC TYRE AGE LIMITS
# ============================================================

# These are simulator constraints, not claims about official
# Pirelli maximum tyre life. They prevent the optimizer from
# selecting obviously unrealistic strategies.

TYRE_MIN_AGE = {
    "SOFT": 8,
    "MEDIUM": 10,
    "HARD": 12
}


TYRE_MAX_AGE = {
    "SOFT": 25,
    "MEDIUM": 35,
    "HARD": 45
}


# Minimum stint length for any strategy.
MIN_STINT = 10


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nLap dataset loaded!")
print("Records:", len(df))


for column in [
    "LapNumber",
    "TyreLife",
    "LapTime"
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "Race",
        "Driver",
        "Compound",
        "LapNumber",
        "TyreLife",
        "LapTime"
    ]
).copy()


df["Driver"] = (
    df["Driver"]
    .astype(str)
    .str.upper()
    .str.strip()
)


df["Compound"] = (
    df["Compound"]
    .astype(str)
    .str.upper()
    .str.strip()
)


df = df[
    (df["LapTime"] > 50) &
    (df["LapTime"] < 150) &
    (df["TyreLife"] >= 2)
].copy()


print(
    "Records after cleaning:",
    len(df)
)


# ============================================================
# SELECT RACE
# ============================================================

print("\n========================================")
print("          AVAILABLE RACES")
print("========================================")


race_names = list(RACES.keys())


for i, race in enumerate(
    race_names,
    start=1
):

    print(
        f"{i} - {race}"
    )


while True:

    try:

        race_choice = int(
            input(
                "\nSelect race number: "
            )
        )

        if (
            1 <= race_choice <= len(race_names)
        ):

            break

    except ValueError:

        pass


    print(
        "Please enter a valid race number."
    )


selected_race = race_names[
    race_choice - 1
]


race_info = RACES[
    selected_race
]


race_laps = race_info[
    "laps"
]


print(
    "\nSelected race:",
    selected_race
)

print(
    "Circuit:",
    race_info["circuit"]
)

print(
    "Race date:",
    race_info["date"]
)

print(
    "Race distance:",
    race_laps,
    "laps"
)


# ============================================================
# SELECT DRIVER
# ============================================================

race_data = df[
    df["Race"] == selected_race
].copy()


drivers = sorted(
    race_data[
        "Driver"
    ].unique()
)


print(
    "\nAvailable drivers:"
)


for i, driver in enumerate(
    drivers,
    start=1
):

    print(
        f"{i} - {driver}"
    )


while True:

    try:

        driver_choice = int(
            input(
                "\nSelect driver number: "
            )
        )

        if (
            1 <= driver_choice <= len(drivers)
        ):

            break

    except ValueError:

        pass


    print(
        "Please enter a valid driver number."
    )


selected_driver = drivers[
    driver_choice - 1
]


driver_history = df[
    df["Driver"] == selected_driver
].copy()


print(
    "\nSelected driver:",
    selected_driver
)


print(
    "\nHistorical records for driver:",
    len(driver_history)
)


print(
    "\nAvailable tyres:"
)


for compound in sorted(
    driver_history[
        "Compound"
    ].unique()
):

    count = len(
        driver_history[
            driver_history[
                "Compound"
            ] == compound
        ]
    )

    print(
        f"{compound:<14} → {count} records"
    )


# ============================================================
# HISTORICAL WEATHER
# ============================================================

def get_weather_name(code):

    code = int(code)

    if code == 0:
        return "Clear"

    if code in [1, 2, 3]:
        return "Cloudy"

    if code in [45, 48]:
        return "Fog"

    if code in [
        51, 53, 55, 56, 57
    ]:
        return "Drizzle"

    if code in [
        61, 63, 65, 66, 67,
        80, 81, 82
    ]:
        return "Rain"

    if code in [
        71, 73, 75, 77,
        85, 86
    ]:
        return "Snow"

    if code in [
        95, 96, 99
    ]:
        return "Thunderstorm"

    return "Unknown"


def get_weather():

    print(
        "\n========================================"
    )

    print(
        "       AUTOMATIC HISTORICAL WEATHER"
    )

    print(
        "========================================"
    )

    print(
        "\nConnecting to Open-Meteo historical data..."
    )


    params = {

        "latitude":
            race_info["lat"],

        "longitude":
            race_info["lon"],

        "hourly":
            "temperature_2m,"
            "precipitation,"
            "rain,"
            "weather_code",

        "timezone":
            "auto",

        "start_date":
            race_info["date"],

        "end_date":
            race_info["date"]
    }


    try:

        response = requests.get(

            "https://archive-api.open-meteo.com/v1/archive",

            params=params,

            timeout=15
        )


        response.raise_for_status()

        data = response.json()

        hourly = pd.DataFrame(
            data["hourly"]
        )


        hourly[
            "weather_name"
        ] = hourly[
            "weather_code"
        ].apply(
            get_weather_name
        )


        times = pd.to_datetime(
            hourly["time"]
        )


        race_window = hourly[
            times.dt.hour.between(
                11,
                22
            )
        ].copy()


        if race_window.empty:

            race_window = hourly.copy()


        average_temperature = float(
            race_window[
                "temperature_2m"
            ].mean()
        )


        total_rain = float(
            race_window[
                "rain"
            ]
            .fillna(0)
            .sum()
        )


        total_precipitation = float(
            race_window[
                "precipitation"
            ]
            .fillna(0)
            .sum()
        )


        dominant_weather = (
            race_window[
                "weather_name"
            ]
            .value_counts()
            .idxmax()
        )


        if (
            total_rain >= 1.0
            or
            total_precipitation >= 1.0
        ):

            track_condition = "WET"

        elif (
            total_rain > 0
            or
            total_precipitation > 0
        ):

            track_condition = "DAMP"

        else:

            track_condition = "DRY"


        if track_condition == "WET":

            weather_impact = "HIGH"

        elif (
            track_condition == "DAMP"
            or
            average_temperature >= 32
        ):

            weather_impact = "MEDIUM"

        else:

            weather_impact = "LOW"


        hourly.to_csv(
            WEATHER_PATH,
            index=False
        )


        print(
            "Historical weather data received."
        )


        print(
            "\nAverage temperature:",
            f"{average_temperature:.2f} °C"
        )

        print(
            "Total rain:",
            f"{total_rain:.1f} mm"
        )

        print(
            "Total precipitation:",
            f"{total_precipitation:.1f} mm"
        )

        print(
            "Dominant weather:",
            dominant_weather
        )

        print(
            "Track condition:",
            track_condition
        )

        print(
            "Weather impact:",
            weather_impact
        )


        return {

            "temperature":
                average_temperature,

            "rain":
                total_rain,

            "precipitation":
                total_precipitation,

            "condition":
                track_condition,

            "impact":
                weather_impact
        }


    except Exception as error:

        print(
            "\nUnable to retrieve historical weather."
        )

        print(
            "Using dry-weather fallback."
        )

        print(
            "Error:",
            error
        )


        return {

            "temperature":
                25.0,

            "rain":
                0.0,

            "precipitation":
                0.0,

            "condition":
                "DRY",

            "impact":
                "LOW"
        }


weather = get_weather()


# ============================================================
# WEATHER-AWARE TYRE SUPPORT
# ============================================================

# The main strategy simulator uses dry compounds because the lap dataset
# is primarily a dry-tyre dataset.  For damp/wet conditions, we calculate
# an additional weather strategy using the driver's historical
# INTERMEDIATE performance when available.

INTERMEDIATE = "INTERMEDIATE"
WET = "WET"


def estimate_weather_tyre_performance():

    intermediate_data = driver_history[
        driver_history["Compound"] == INTERMEDIATE
    ].copy()

    if intermediate_data.empty:
        intermediate_data = df[
            df["Compound"] == INTERMEDIATE
        ].copy()

    if intermediate_data.empty:
        return {
            "INTERMEDIATE": driver_overall + 4.0,
            "WET": driver_overall + 7.0
        }, 0.06

    intermediate_baseline = float(
        intermediate_data["LapTime"].median()
    )

    # Keep the estimate bounded so a small number of wet laps cannot
    # produce an unrealistic race strategy.
    intermediate_baseline = float(
        np.clip(
            intermediate_baseline,
            driver_overall - 2.0,
            driver_overall + 8.0
        )
    )

    degradation = 0.06

    if len(intermediate_data) >= 12:

        degradation_data = intermediate_data[
            (intermediate_data["TyreLife"] >= 2)
            &
            (intermediate_data["TyreLife"] <= 25)
        ].copy()

        if (
            len(degradation_data) >= 10
            and
            degradation_data["TyreLife"].nunique() >= 4
        ):

            try:
                slope = float(
                    np.polyfit(
                        degradation_data["TyreLife"].to_numpy(dtype=float),
                        degradation_data["LapTime"].to_numpy(dtype=float),
                        1
                    )[0]
                )

                if np.isfinite(slope) and slope > 0:
                    degradation = float(
                        np.clip(slope, 0.025, 0.12)
                    )

            except Exception:
                pass

    # Full-wet performance is not present in the current dataset.
    # Therefore WET is a conservative estimate derived from INTERMEDIATE.
    wet_baseline = intermediate_baseline + 2.5

    return {
        "INTERMEDIATE": intermediate_baseline,
        "WET": wet_baseline
    }, degradation


weather_tyre_baseline = {}
intermediate_degradation = 0.06


def build_weather_strategy(dry_best):

    if weather["condition"] == "DRY":
        return {
            "strategy": (
                f"{dry_best['FirstTyre']} → {dry_best['SecondTyre']}"
                if dry_best["StrategyType"] == "1-STOP"
                else
                f"{dry_best['FirstTyre']} → "
                f"{dry_best['SecondTyre']} → "
                f"{dry_best['ThirdTyre']}"
            ),
            "pit1": int(dry_best["PitLap1"]),
            "pit2": (
                None
                if pd.isna(dry_best["PitLap2"])
                else int(dry_best["PitLap2"])
            ),
            "primary": dry_best["FirstTyre"],
            "backup": dry_best["SecondTyre"],
            "estimated_time": float(dry_best["EstimatedRaceTime"]),
            "reason": "Dry conditions allow the dry strategy simulator to determine the recommendation."
        }

    # Weather pit timing is based on the race progress and is deliberately
    # kept conservative.  The actual wet/damp compound is not mixed into
    # the dry optimizer because the current dataset does not contain a
    # reliable full wet compound model.
    if weather["condition"] == "WET":

        pit1 = int(
            np.clip(
                round(race_laps * 0.30),
                10,
                max(10, race_laps - 20)
            )
        )

        pit2 = int(
            np.clip(
                round(race_laps * 0.65),
                pit1 + 10,
                race_laps - 10
            )
        )

        return {
            "strategy": "INTERMEDIATE → INTERMEDIATE → DRY TRANSITION",
            "pit1": pit1,
            "pit2": pit2,
            "primary": "INTERMEDIATE",
            "backup": "MEDIUM",
            "estimated_time": None,
            "reason": (
                "Wet conditions override the dry model. "
                "Use Intermediate tyres initially and prepare for a switch "
                "to dry tyres as the track improves. Full Wet performance "
                "is not available in the current historical dataset."
            )
        }

    # DAMP
    pit1 = int(
        np.clip(
            round(race_laps * 0.30),
            10,
            max(10, race_laps - 20)
        )
    )

    return {
        "strategy": "INTERMEDIATE → DRY TRANSITION",
        "pit1": pit1,
        "pit2": None,
        "primary": "INTERMEDIATE",
        "backup": "MEDIUM",
        "estimated_time": None,
        "reason": (
            "Damp conditions require Intermediate tyres initially. "
            "Monitor the drying line and switch to Medium/Hard when "
            "dry-tyre conditions become favourable."
        )
    }


# ============================================================
# TYRE PERFORMANCE MODEL
# ============================================================

print(
    "\n========================================"
)

print(
    "       TRAINING TYRE PERFORMANCE"
)

print(
    "========================================"
)


driver_overall = float(
    driver_history[
        "LapTime"
    ].median()
)


weather_tyre_baseline, intermediate_degradation = (
    estimate_weather_tyre_performance()
)


# ============================================================
# IMPORTANT:
# Use a driver-normalized approach.
#
# Each tyre's pace is measured against the same driver's
# overall median. This is more stable than comparing raw
# lap times across different races.
# ============================================================

driver_history[
    "DriverMedian"
] = (

    driver_history
    .groupby(
        "Driver"
    )[
        "LapTime"
    ]
    .transform(
        "median"
    )
)


driver_history[
    "NormalizedLapTime"
] = (

    driver_history[
        "LapTime"
    ]

    -

    driver_history[
        "DriverMedian"
    ]
)


compound_offsets = {}
degradation_rates = {}


# ============================================================
# MANUAL FALLBACK PHYSICAL DEGRADATION
# ============================================================

# Used only when historical regression is too weak.
#
# The values preserve the expected relationship:
#
# SOFT   = fastest but highest degradation
# MEDIUM = balanced
# HARD   = slower initial pace but lowest degradation

FALLBACK_DEGRADATION = {

    "SOFT":
        0.085,

    "MEDIUM":
        0.055,

    "HARD":
        0.030
}


FALLBACK_OFFSET = {

    "SOFT":
        -0.90,

    "MEDIUM":
        0.00,

    "HARD":
        0.70
}


for compound in DRY_COMPOUNDS:

    compound_data = driver_history[
        driver_history[
            "Compound"
        ] == compound
    ].copy()


    if len(compound_data) < 12:

        compound_data = df[
            df["Compound"] == compound
        ].copy()


        compound_data[
            "DriverMedian"
        ] = (

            compound_data
            .groupby(
                "Driver"
            )[
                "LapTime"
            ]
            .transform(
                "median"
            )
        )


        compound_data[
            "NormalizedLapTime"
        ] = (

            compound_data[
                "LapTime"
            ]

            -

            compound_data[
                "DriverMedian"
            ]
        )


    # --------------------------------------------------------
    # PACE OFFSET
    # --------------------------------------------------------

    if len(compound_data) >= 10:

        offset = float(
            compound_data[
                "NormalizedLapTime"
            ].median()
        )


        offset = float(
            np.clip(
                offset,
                -1.5,
                1.5
            )
        )

    else:

        offset = FALLBACK_OFFSET[
            compound
        ]


    # --------------------------------------------------------
    # DEGRADATION
    # --------------------------------------------------------

    degradation = None


    if (
        len(compound_data) >= 15
    ):

        # Use only realistic tyre ages.
        degradation_data = compound_data[
            (
                compound_data[
                    "TyreLife"
                ] >= 2
            )
            &
            (
                compound_data[
                    "TyreLife"
                ] <= 30
            )
        ].copy()


        if (
            len(degradation_data) >= 12
            and
            degradation_data[
                "TyreLife"
            ].nunique() >= 5
        ):

            x = degradation_data[
                "TyreLife"
            ].to_numpy(
                dtype=float
            )


            y = degradation_data[
                "NormalizedLapTime"
            ].to_numpy(
                dtype=float
            )


            try:

                slope = float(
                    np.polyfit(
                        x,
                        y,
                        1
                    )[0]
                )


                if (
                    np.isfinite(
                        slope
                    )
                    and
                    slope > 0.015
                ):

                    degradation = slope

            except Exception:

                degradation = None


    if degradation is None:

        degradation = (
            FALLBACK_DEGRADATION[
                compound
            ]
        )


    # --------------------------------------------------------
    # Keep degradation physically meaningful.
    # --------------------------------------------------------

    degradation = float(
        np.clip(
            degradation,
            0.025,
            0.12
        )
    )


    compound_offsets[
        compound
    ] = offset


    degradation_rates[
        compound
    ] = degradation


# ============================================================
# FORCE COMPOUND ORDER IF DATA IS TOO NOISY
# ============================================================

# If the historical data produces a physically impossible
# relationship, gently correct it.

if not (
    degradation_rates["SOFT"]
    >=
    degradation_rates["MEDIUM"]
    >=
    degradation_rates["HARD"]
):

    degradation_rates[
        "SOFT"
    ] = max(
        degradation_rates["SOFT"],
        0.075
    )


    degradation_rates[
        "MEDIUM"
    ] = max(
        min(
            degradation_rates["MEDIUM"],
            0.070
        ),
        0.045
    )


    degradation_rates[
        "HARD"
    ] = min(
        degradation_rates["HARD"],
        0.045
    )


# ============================================================
# DISPLAY TYRE PERFORMANCE
# ============================================================

print(
    "\nDriver overall median lap:",
    round(
        driver_overall,
        3
    ),
    "seconds"
)


print(
    "\nCompound pace offsets:"
)


for compound in DRY_COMPOUNDS:

    print(
        f"{compound:<14}"
        f"{compound_offsets[compound]:+.3f}"
        " sec/lap"
    )


print(
    "\nDegradation rates:"
)


for compound in DRY_COMPOUNDS:

    print(
        f"{compound:<14}"
        f"{degradation_rates[compound]:.4f}"
        " sec/lap"
    )


# ============================================================
# LAP-TIME MODEL
# ============================================================

print(
    "\n========================================"
)

print(
    "       TRAINING LAP-TIME MODEL"
)

print(
    "========================================"
)


model_data = df[
    df["Compound"].isin(
        DRY_COMPOUNDS
    )
].copy()


driver_encoder = LabelEncoder()

compound_encoder = LabelEncoder()


driver_encoder.fit(
    model_data["Driver"]
)

compound_encoder.fit(
    model_data["Compound"]
)


model_data[
    "DriverCode"
] = driver_encoder.transform(
    model_data["Driver"]
)


model_data[
    "CompoundCode"
] = compound_encoder.transform(
    model_data["Compound"]
)


race_lap_map = {

    race:
        info["laps"]

    for race, info in RACES.items()
}


model_data[
    "RaceProgress"
] = (

    model_data[
        "LapNumber"
    ]

    /

    model_data[
        "Race"
    ]
    .map(
        race_lap_map
    )
    .fillna(
        60
    )
)


MODEL_FEATURES = [

    "DriverCode",
    "CompoundCode",
    "TyreLife",
    "LapNumber",
    "RaceProgress"
]


lap_model = RandomForestRegressor(

    n_estimators=60,

    max_depth=10,

    min_samples_leaf=6,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


lap_model.fit(

    model_data[
        MODEL_FEATURES
    ],

    model_data[
        "LapTime"
    ]
)


print(
    "Lap-time prediction model trained."
)


# ============================================================
# BUILD PREDICTION CACHE
# ============================================================

print(
    "\n========================================"
)

print(
    "       BUILDING PREDICTION CACHE"
)

print(
    "========================================"
)


driver_code = int(
    driver_encoder.transform(
        [
            selected_driver
        ]
    )[0]
)


cache_rows = []


for compound in DRY_COMPOUNDS:

    compound_code = int(
        compound_encoder.transform(
            [
                compound
            ]
        )[0]
    )


    for lap in range(
        1,
        race_laps + 1
    ):

        cache_rows.append({

            "DriverCode":
                driver_code,

            "Compound":
                compound,

            "CompoundCode":
                compound_code,

            "TyreLife":
                lap,

            "LapNumber":
                lap,

            "RaceProgress":
                lap / race_laps
        })


cache = pd.DataFrame(
    cache_rows
)


cache[
    "MLPrediction"
] = lap_model.predict(
    cache[
        MODEL_FEATURES
    ]
)


# ============================================================
# FINAL TYRE PREDICTION
# ============================================================

for compound in DRY_COMPOUNDS:

    mask = (
        cache[
            "Compound"
        ]
        ==
        compound
    )


    tyre_age = cache.loc[
        mask,
        "TyreLife"
    ].to_numpy(
        dtype=float
    )


    lap_number = cache.loc[
        mask,
        "LapNumber"
    ].to_numpy(
        dtype=float
    )


    statistical_prediction = (

        driver_overall

        +

        compound_offsets[
            compound
        ]

        +

        degradation_rates[
            compound
        ]
        *
        np.maximum(
            tyre_age - 2,
            0
        )

        -

        0.025
        *
        np.maximum(
            lap_number - 1,
            0
        )
    )


    ml_prediction = cache.loc[
        mask,
        "MLPrediction"
    ].to_numpy(
        dtype=float
    )


    # Statistical model is dominant.
    final_prediction = (

        0.85
        *
        statistical_prediction

        +

        0.15
        *
        ml_prediction
    )


    cache.loc[
        mask,
        "PredictedLapTime"
    ] = final_prediction


cache[
    "PredictedLapTime"
] = cache[
    "PredictedLapTime"
].clip(
    70,
    130
)


prediction_cache = {

    (
        row.Compound,
        int(row.TyreLife),
        int(row.LapNumber)
    ):

        float(
            row.PredictedLapTime
        )

    for row in cache.itertuples()
}


print(
    "Prediction cache ready."
)


# ============================================================
# DISPLAY PREDICTED TYRE PERFORMANCE
# ============================================================

print(
    "\n========================================"
)

print(
    "       PREDICTED TYRE PERFORMANCE"
)

print(
    "========================================"
)


for compound in DRY_COMPOUNDS:

    predicted = prediction_cache[
        (
            compound,
            10,
            10
        )
    ]


    print(
        f"{compound:<14}"
        f"{predicted:.3f}"
        " sec/lap"
    )


# ============================================================
# STINT CACHE
# ============================================================

print(
    "\n========================================"
)

print(
    "       BUILDING STINT-TIME CACHE"
)

print(
    "========================================"
)


stint_prefix = {}


for compound in DRY_COMPOUNDS:

    prefix = [
        0.0
    ]


    for age in range(
        1,
        race_laps + 1
    ):

        lap_time = prediction_cache[
            (
                compound,
                age,
                age
            )
        ]


        prefix.append(
            prefix[-1]
            +
            lap_time
        )


    stint_prefix[
        compound
    ] = prefix


def stint_time(
    compound,
    stint_length
):

    if stint_length <= 0:

        return 0.0


    return stint_prefix[
        compound
    ][
        stint_length
    ]


print(
    "Stint-time cache ready."
)


# ============================================================
# PIT-STOP ML MODEL
# ============================================================

print(
    "\nTraining pit-stop prediction model..."
)


pit_model = None


pit_driver_encoder = None
pit_previous_encoder = None
pit_new_encoder = None


if os.path.exists(
    PIT_DATA_PATH
):

    pit_data = pd.read_csv(
        PIT_DATA_PATH
    )


    required_columns = [

        "Driver",
        "PreviousTyre",
        "PreviousTyreAge",
        "PitLap",
        "NewTyre"
    ]


    if all(
        column in pit_data.columns
        for column in required_columns
    ):

        pit_data = pit_data.dropna(
            subset=required_columns
        ).copy()


        if len(pit_data) >= 20:

            pit_data[
                "Driver"
            ] = (
                pit_data[
                    "Driver"
                ]
                .astype(str)
                .str.upper()
            )


            pit_data[
                "PreviousTyre"
            ] = (
                pit_data[
                    "PreviousTyre"
                ]
                .astype(str)
                .str.upper()
            )


            pit_data[
                "NewTyre"
            ] = (
                pit_data[
                    "NewTyre"
                ]
                .astype(str)
                .str.upper()
            )


            pit_driver_encoder = (
                LabelEncoder()
            )


            pit_previous_encoder = (
                LabelEncoder()
            )


            pit_new_encoder = (
                LabelEncoder()
            )


            pit_driver_encoder.fit(
                pit_data[
                    "Driver"
                ]
            )


            pit_previous_encoder.fit(
                pit_data[
                    "PreviousTyre"
                ]
            )


            pit_new_encoder.fit(
                pit_data[
                    "NewTyre"
                ]
            )


            X_pit = pd.DataFrame({

                "DriverCode":
                    pit_driver_encoder.transform(
                        pit_data[
                            "Driver"
                        ]
                    ),

                "PreviousTyreCode":
                    pit_previous_encoder.transform(
                        pit_data[
                            "PreviousTyre"
                        ]
                    ),

                "PreviousTyreAge":
                    pd.to_numeric(
                        pit_data[
                            "PreviousTyreAge"
                        ]
                    ),

                "PitLap":
                    pd.to_numeric(
                        pit_data[
                            "PitLap"
                        ]
                    )
            })


            y_pit = (
                pit_new_encoder.transform(
                    pit_data[
                        "NewTyre"
                    ]
                )
            )


            pit_model = RandomForestClassifier(

                n_estimators=80,

                max_depth=8,

                min_samples_leaf=2,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1
            )


            pit_model.fit(
                X_pit,
                y_pit
            )


print(
    "Pit-stop ML model trained."
)


# ============================================================
# ML TYRE PREDICTION
# ============================================================

def predict_next_tyre(
    current_tyre,
    tyre_age,
    race_lap
):

    if (
        pit_model is None
        or pit_driver_encoder is None
        or pit_previous_encoder is None
        or pit_new_encoder is None
    ):
        return None, {}


    if (
        selected_driver
        not in pit_driver_encoder.classes_
    ):

        return None, {}


    if (
        current_tyre
        not in pit_previous_encoder.classes_
    ):

        return None, {}


    driver_code = int(
        pit_driver_encoder.transform(
            [
                selected_driver
            ]
        )[0]
    )


    tyre_code = int(
        pit_previous_encoder.transform(
            [
                current_tyre
            ]
        )[0]
    )


    input_data = pd.DataFrame({

        "DriverCode":
            [driver_code],

        "PreviousTyreCode":
            [tyre_code],

        "PreviousTyreAge":
            [float(tyre_age)],

        "PitLap":
            [float(race_lap)]
    })


    probabilities = (
        pit_model.predict_proba(
            input_data
        )[0]
    )


    classes = (
        pit_new_encoder.inverse_transform(
            pit_model.classes_
        )
    )


    probability_dict = {

        str(compound):
            float(
                probability * 100
            )

        for compound, probability
        in zip(
            classes,
            probabilities
        )
    }


    prediction = max(
        probability_dict,
        key=probability_dict.get
    )


    return (
        prediction,
        probability_dict
    )


# ============================================================
# SEARCH STRATEGIES
# ============================================================

print(
    "\n========================================"
)

print(
    "        SEARCHING PIT STRATEGIES"
)

print(
    "========================================"
)


strategies = []


# ============================================================
# ONE STOP
# ============================================================

for first_tyre in DRY_COMPOUNDS:

    for second_tyre in DRY_COMPOUNDS:

        if first_tyre == second_tyre:

            continue


        first_min = max(
            MIN_STINT,
            TYRE_MIN_AGE[
                first_tyre
            ]
        )


        second_min = max(
            MIN_STINT,
            TYRE_MIN_AGE[
                second_tyre
            ]
        )


        first_max = min(
            TYRE_MAX_AGE[
                first_tyre
            ],
            race_laps - second_min
        )


        for pit_lap in range(

            first_min,

            first_max + 1
        ):


            first_stint = pit_lap

            second_stint = (
                race_laps
                -
                pit_lap
            )


            if (
                second_stint
                <
                second_min
            ):

                continue


            if (
                second_stint
                >
                TYRE_MAX_AGE[
                    second_tyre
                ]
            ):

                continue


            total_time = (

                stint_time(
                    first_tyre,
                    first_stint
                )

                +

                PIT_LOSS

                +

                stint_time(
                    second_tyre,
                    second_stint
                )
            )


            penalty = 0.0


            # Penalize extreme stint imbalance.
            imbalance = abs(
                first_stint
                -
                second_stint
            )


            if imbalance > 22:

                penalty += 2.0


            # Small realism penalty for very late pit.
            if pit_lap > race_laps * 0.75:

                penalty += 2.0


            strategies.append({

                "StrategyType":
                    "1-STOP",

                "FirstTyre":
                    first_tyre,

                "SecondTyre":
                    second_tyre,

                "ThirdTyre":
                    "-",

                "PitLap1":
                    pit_lap,

                "PitLap2":
                    np.nan,

                "EstimatedRaceTime":
                    total_time,

                "StrategyPenalty":
                    penalty,

                "StrategyScore":
                    total_time + penalty
            })


# ============================================================
# TWO STOP
# ============================================================

for first_tyre in DRY_COMPOUNDS:

    for second_tyre in DRY_COMPOUNDS:

        for third_tyre in DRY_COMPOUNDS:

            first_min = max(
                MIN_STINT,
                TYRE_MIN_AGE[
                    first_tyre
                ]
            )


            second_min = max(
                MIN_STINT,
                TYRE_MIN_AGE[
                    second_tyre
                ]
            )


            third_min = max(
                MIN_STINT,
                TYRE_MIN_AGE[
                    third_tyre
                ]
            )


            first_max = min(
                TYRE_MAX_AGE[
                    first_tyre
                ],
                race_laps
                -
                second_min
                -
                third_min
            )


            for first_pit in range(

                first_min,

                first_max + 1
            ):


                second_start = (
                    first_pit
                    +
                    second_min
                )


                second_max = min(

                    first_pit
                    +
                    TYRE_MAX_AGE[
                        second_tyre
                    ],

                    race_laps
                    -
                    third_min
                )


                for second_pit in range(

                    second_start,

                    second_max + 1
                ):


                    stint1 = first_pit

                    stint2 = (
                        second_pit
                        -
                        first_pit
                    )

                    stint3 = (
                        race_laps
                        -
                        second_pit
                    )


                    if (
                        stint1
                        >
                        TYRE_MAX_AGE[
                            first_tyre
                        ]
                    ):

                        continue


                    if (
                        stint2
                        >
                        TYRE_MAX_AGE[
                            second_tyre
                        ]
                    ):

                        continue


                    if (
                        stint3
                        >
                        TYRE_MAX_AGE[
                            third_tyre
                        ]
                    ):

                        continue


                    total_time = (

                        stint_time(
                            first_tyre,
                            stint1
                        )

                        +

                        PIT_LOSS

                        +

                        stint_time(
                            second_tyre,
                            stint2
                        )

                        +

                        PIT_LOSS

                        +

                        stint_time(
                            third_tyre,
                            stint3
                        )
                    )


                    penalty = 4.0


                    if min(
                        stint1,
                        stint2,
                        stint3
                    ) < 12:

                        penalty += 2.0


                    if max(
                        stint1,
                        stint2,
                        stint3
                    ) > 30:

                        penalty += 1.5


                    strategies.append({

                        "StrategyType":
                            "2-STOP",

                        "FirstTyre":
                            first_tyre,

                        "SecondTyre":
                            second_tyre,

                        "ThirdTyre":
                            third_tyre,

                        "PitLap1":
                            first_pit,

                        "PitLap2":
                            second_pit,

                        "EstimatedRaceTime":
                            total_time,

                        "StrategyPenalty":
                            penalty,

                        "StrategyScore":
                            total_time + penalty
                    })


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    strategies
)


results = (
    results
    .sort_values(
        [
            "StrategyScore",
            "EstimatedRaceTime"
        ]
    )
    .reset_index(
        drop=True
    )
)


best = results.iloc[0]


best_one_stop = (
    results[
        results[
            "StrategyType"
        ] == "1-STOP"
    ]
    .sort_values(
        "StrategyScore"
    )
    .iloc[0]
)


best_two_stop = (
    results[
        results[
            "StrategyType"
        ] == "2-STOP"
    ]
    .sort_values(
        "StrategyScore"
    )
    .iloc[0]
)


# ============================================================
# WEATHER-ADJUSTED STRATEGY
# ============================================================

weather_strategy = build_weather_strategy(best)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "       FINAL PRE-RACE STRATEGY"
)

print(
    "========================================"
)


print(
    "\nDriver:",
    selected_driver
)

print(
    "Race:",
    selected_race
)

print(
    "Circuit:",
    race_info[
        "circuit"
    ]
)

print(
    "Race distance:",
    race_laps,
    "laps"
)


# ============================================================
# WEATHER
# ============================================================

print(
    "\n----------------------------------------"
)

print(
    "AUTOMATIC WEATHER"
)

print(
    "----------------------------------------"
)


print(
    f"Average temperature: "
    f"{weather['temperature']:.2f} °C"
)

print(
    f"Rain: "
    f"{weather['rain']:.1f} mm"
)

print(
    f"Precipitation: "
    f"{weather['precipitation']:.1f} mm"
)

print(
    "Track condition:",
    weather[
        "condition"
    ]
)

print(
    "Weather impact:",
    weather[
        "impact"
    ]
)


# ============================================================
# ONE STOP VS TWO STOP
# ============================================================

print(
    "\n========================================"
)

print(
    "          1-STOP vs 2-STOP"
)

print(
    "========================================"
)


print(
    "\nBEST 1-STOP"
)

print(
    "Strategy:",
    best_one_stop[
        "FirstTyre"
    ],
    "→",
    best_one_stop[
        "SecondTyre"
    ]
)

print(
    "Pit lap:",
    int(
        best_one_stop[
            "PitLap1"
        ]
    )
)

print(
    "Estimated time:",
    round(
        best_one_stop[
            "EstimatedRaceTime"
        ],
        2
    )
)


print(
    "\nBEST 2-STOP"
)

print(
    "Strategy:",
    best_two_stop[
        "FirstTyre"
    ],
    "→",
    best_two_stop[
        "SecondTyre"
    ],
    "→",
    best_two_stop[
        "ThirdTyre"
    ]
)

print(
    "First pit:",
    int(
        best_two_stop[
            "PitLap1"
        ]
    )
)

print(
    "Second pit:",
    int(
        best_two_stop[
            "PitLap2"
        ]
    )
)

print(
    "Estimated time:",
    round(
        best_two_stop[
            "EstimatedRaceTime"
        ],
        2
    )
)


difference = abs(

    best_one_stop[
        "StrategyScore"
    ]

    -

    best_two_stop[
        "StrategyScore"
    ]
)


preferred = (

    "1-STOP"

    if
    best_one_stop[
        "StrategyScore"
    ]
    <
    best_two_stop[
        "StrategyScore"
    ]

    else
    "2-STOP"
)


print(
    "\nStrategy score difference:",
    round(
        difference,
        2
    ),
    "seconds"
)


print(
    "Preferred strategy:",
    preferred
)


# ============================================================
# BEST DRY STRATEGY
# ============================================================

print(
    "\n========================================"
)

print(
    "       BEST DRY STRATEGY"
)

print(
    "========================================"
)


if (
    best[
        "StrategyType"
    ]
    ==
    "1-STOP"
):

    print(
        "\nStrategy:",
        best[
            "FirstTyre"
        ],
        "→",
        best[
            "SecondTyre"
        ]
    )


    print(
        "Pit lap:",
        int(
            best[
                "PitLap1"
            ]
        )
    )


else:

    print(
        "\nStrategy:",
        best[
            "FirstTyre"
        ],
        "→",
        best[
            "SecondTyre"
        ],
        "→",
        best[
            "ThirdTyre"
        ]
    )


    print(
        "First pit:",
        int(
            best[
                "PitLap1"
            ]
        )
    )


    print(
        "Second pit:",
        int(
            best[
                "PitLap2"
            ]
        )
    )


print(
    "Estimated race time:",
    round(
        best[
            "EstimatedRaceTime"
        ],
        2
    ),
    "seconds"
)


# ============================================================
# STINT ANALYSIS
# ============================================================

print(
    "\n----------------------------------------"
)

print(
    "STINT ANALYSIS"
)

print(
    "----------------------------------------"
)


if (
    best[
        "StrategyType"
    ]
    ==
    "1-STOP"
):

    pit1 = int(
        best[
            "PitLap1"
        ]
    )


    print(
        f"{best['FirstTyre']} stint:",
        pit1,
        "laps"
    )


    print(
        f"{best['SecondTyre']} stint:",
        race_laps - pit1,
        "laps"
    )


else:

    pit1 = int(
        best[
            "PitLap1"
        ]
    )


    pit2 = int(
        best[
            "PitLap2"
        ]
    )


    print(
        f"{best['FirstTyre']} stint:",
        pit1,
        "laps"
    )


    print(
        f"{best['SecondTyre']} stint:",
        pit2 - pit1,
        "laps"
    )


    print(
        f"{best['ThirdTyre']} stint:",
        race_laps - pit2,
        "laps"
    )


# ============================================================
# DEGRADATION
# ============================================================

print(
    "\n----------------------------------------"
)

print(
    "TYRE DEGRADATION"
)

print(
    "----------------------------------------"
)


used_tyres = [

    best[
        "FirstTyre"
    ],

    best[
        "SecondTyre"
    ]
]


if (
    best[
        "StrategyType"
    ]
    ==
    "2-STOP"
):

    used_tyres.append(
        best[
            "ThirdTyre"
        ]
    )


for tyre in used_tyres:

    print(
        f"{tyre:<14}"
        f"{degradation_rates[tyre]:.4f}"
        " sec/lap"
    )


# ============================================================
# ML PREDICTION
# ============================================================

print(
    "\n========================================"
)

print(
    "        ML TYRE PREDICTION"
)

print(
    "========================================"
)


prediction_lap = int(
    best[
        "PitLap1"
    ]
)


current_tyre = best[
    "FirstTyre"
]


ml_tyre, probabilities = (
    predict_next_tyre(

        current_tyre,

        prediction_lap,

        prediction_lap
    )
)


print(
    "\nCurrent tyre:",
    current_tyre
)

print(
    "Tyre age:",
    prediction_lap
)

print(
    "Race lap:",
    prediction_lap
)


if ml_tyre is not None:

    print(
        "\nML predicted next tyre:",
        ml_tyre
    )


    print(
        "\nTyre probabilities:"
    )


    for tyre, probability in sorted(

        probabilities.items(),

        key=lambda x: x[1],

        reverse=True
    ):

        print(
            f"{tyre:<15}"
            f"{probability:.2f}%"
        )


    simulator_next = best[
        "SecondTyre"
    ]


    print(
        "\nSimulator next tyre:",
        simulator_next
    )


    if (
        ml_tyre
        ==
        simulator_next
    ):

        print(
            "\n✓ ML and simulator agree."
        )

    else:

        print(
            "\n⚠ ML and simulator disagree."
        )


# ============================================================
# CONFIDENCE
# ============================================================

confidence = 65.0


if difference >= 30:

    confidence += 15

elif difference >= 15:

    confidence += 8

elif difference < 5:

    confidence -= 10


if (
    ml_tyre
    ==
    best[
        "SecondTyre"
    ]
):

    confidence += 10

elif ml_tyre is not None:

    confidence -= 5


if weather["condition"] != "DRY":

    confidence -= 20


confidence = max(
    20.0,
    min(
        95.0,
        confidence
    )
)


if confidence >= 75:

    confidence_level = "HIGH"

elif confidence >= 55:

    confidence_level = "MEDIUM"

else:

    confidence_level = "LOW"


print(
    "\n========================================"
)

print(
    "          MODEL CONFIDENCE"
)

print(
    "========================================"
)


print(
    f"\nConfidence score: "
    f"{confidence:.1f} %"
)

print(
    "Confidence level:",
    confidence_level
)


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print(
    "\n========================================"
)

print(
    "       FINAL RECOMMENDATION"
)

print(
    "========================================"
)

print(
    "\nDry-model strategy:"
)

if best["StrategyType"] == "1-STOP":

    print(
        f"{best['FirstTyre']} → {best['SecondTyre']}"
    )

else:

    print(
        f"{best['FirstTyre']} → "
        f"{best['SecondTyre']} → "
        f"{best['ThirdTyre']}"
    )


print(
    "\nWeather-adjusted strategy:"
)

print(
    weather_strategy["strategy"]
)

print(
    "\nPrimary tyre:",
    weather_strategy["primary"]
)

print(
    "Backup tyre:",
    weather_strategy["backup"]
)

print(
    "\nReason:"
)

print(
    weather_strategy["reason"]
)

if weather_strategy["pit1"] is not None:

    print(
        "\nWeather pit recommendation:"
    )

    print(
        "First pit around lap",
        weather_strategy["pit1"]
    )

    if weather_strategy["pit2"] is not None:

        print(
            "Second pit around lap",
            weather_strategy["pit2"]
        )


# ============================================================
# PIT WINDOW
# ============================================================

print(
    "\n========================================"
)

print(
    "             PIT WINDOW"
)

print(
    "========================================"
)

if weather["condition"] == "DRY":

    pit1 = int(
        best["PitLap1"]
    )

    print(
        "Recommended first pit window:",
        f"{max(MIN_STINT, pit1 - 2)}",
        f" - ",
        f"{min(race_laps - MIN_STINT, pit1 + 2)}"
    )

    if best["StrategyType"] == "2-STOP":

        pit2 = int(
            best["PitLap2"]
        )

        print(
            "Recommended second pit window:",
            f"{max(pit1 + MIN_STINT, pit2 - 2)}",
            f" - ",
            f"{min(race_laps - MIN_STINT, pit2 + 2)}"
        )

else:

    pit1 = int(
        weather_strategy["pit1"]
    )

    print(
        "Weather-adjusted first pit window:",
        f"{max(MIN_STINT, pit1 - 2)}",
        f" - ",
        f"{min(race_laps - MIN_STINT, pit1 + 2)}"
    )

    if weather_strategy["pit2"] is not None:

        pit2 = int(
            weather_strategy["pit2"]
        )

        print(
            "Weather-adjusted second pit window:",
            f"{max(pit1 + MIN_STINT, pit2 - 2)}",
            f" - ",
            f"{min(race_laps - MIN_STINT, pit2 + 2)}"
        )


# ============================================================
# TOP 10
# ============================================================

print(
    "\n========================================"
)

print(
    "          TOP 10 STRATEGIES"
)

print(
    "========================================"
)


display_columns = [

    "StrategyType",
    "FirstTyre",
    "SecondTyre",
    "ThirdTyre",
    "PitLap1",
    "PitLap2",
    "EstimatedRaceTime",
    "StrategyPenalty",
    "StrategyScore"
]


top10 = results.head(
    10
)[
    display_columns
].copy()


print(
    top10.to_string(
        index=False,
        formatters={

            "EstimatedRaceTime":
                "{:.2f}".format,

            "StrategyPenalty":
                "{:.2f}".format,

            "StrategyScore":
                "{:.2f}".format
        }
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    RESULT_PATH,
    index=False
)


print(
    "\nWeather model:"
)

print(
    "Historical INTERMEDIATE support:",
    "YES" if not driver_history[driver_history["Compound"] == "INTERMEDIATE"].empty else "NO"
)

print(
    "Weather-adjusted strategy applied:",
    "YES" if weather["condition"] != "DRY" else "NO"
)


print(
    "\nResults saved to:"
)

print(
    RESULT_PATH
)


# ============================================================
# READY
# ============================================================

print(
    "\n========================================"
)

print(
    "     F1 PRE-RACE SYSTEM READY!"
)

print(
    "========================================"
)

# ============================================================
# SAVE STRUCTURED BACKEND RESULT
# ============================================================

import json

if best["StrategyType"] == "1-STOP":

    dry_strategy_text = (
        f"{best['FirstTyre']} → {best['SecondTyre']}"
    )

else:

    dry_strategy_text = (
        f"{best['FirstTyre']} → "
        f"{best['SecondTyre']} → "
        f"{best['ThirdTyre']}"
    )


backend_result = {

    "race": str(selected_race),

    "driver": str(selected_driver),

    "circuit": str(
        race_info["circuit"]
    ),

    "race_date": str(
        race_info["date"]
    ),

    "race_distance": int(
        race_laps
    ),


    # --------------------------------------------------------
    # FINAL STRATEGY
    # --------------------------------------------------------

    "dry_strategy":
        dry_strategy_text,

    "weather_strategy":
        str(
            weather_strategy["strategy"]
        ),

    "strategy_type":
        str(
            best["StrategyType"]
        ),


    # --------------------------------------------------------
    # TYRES
    # --------------------------------------------------------

    "primary_tyre":
        str(
            weather_strategy["primary"]
        ),

    "backup_tyre":
        str(
            weather_strategy["backup"]
        ),


    # --------------------------------------------------------
    # PIT STOPS
    # --------------------------------------------------------

    "pit_lap_1":
        int(
            best["PitLap1"]
        ),

    "pit_lap_2":
        (
            None
            if pd.isna(
                best["PitLap2"]
            )
            else int(
                best["PitLap2"]
            )
        ),

    "weather_pit_lap_1":
        (
            None
            if weather_strategy["pit1"] is None
            else int(
                weather_strategy["pit1"]
            )
        ),

    "weather_pit_lap_2":
        (
            None
            if weather_strategy["pit2"] is None
            else int(
                weather_strategy["pit2"]
            )
        ),


    # --------------------------------------------------------
    # RACE TIME
    # --------------------------------------------------------

    "estimated_race_time":
        float(
            best["EstimatedRaceTime"]
        ),

    "strategy_score":
        float(
            best["StrategyScore"]
        ),


    # --------------------------------------------------------
    # 1-STOP
    # --------------------------------------------------------

    "one_stop_strategy": (
        f"{best_one_stop['FirstTyre']} → "
        f"{best_one_stop['SecondTyre']}"
    ),

    "one_stop_pit_lap":
        int(
            best_one_stop["PitLap1"]
        ),

    "one_stop_time":
        float(
            best_one_stop[
                "EstimatedRaceTime"
            ]
        ),


    # --------------------------------------------------------
    # 2-STOP
    # --------------------------------------------------------

    "two_stop_strategy": (
        f"{best_two_stop['FirstTyre']} → "
        f"{best_two_stop['SecondTyre']} → "
        f"{best_two_stop['ThirdTyre']}"
    ),

    "two_stop_pit_lap_1":
        int(
            best_two_stop["PitLap1"]
        ),

    "two_stop_pit_lap_2":
        int(
            best_two_stop["PitLap2"]
        ),

    "two_stop_time":
        float(
            best_two_stop[
                "EstimatedRaceTime"
            ]
        ),


    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    "preferred_strategy":
        str(preferred),

    "strategy_score_difference":
        float(difference),


    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    "ml_predicted_tyre":
        (
            None
            if ml_tyre is None
            else str(ml_tyre)
        ),

    "ml_probabilities": {
        str(key): float(value)
        for key, value
        in probabilities.items()
    },


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    "confidence_score":
        float(confidence),

    "confidence_level":
        str(confidence_level),


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    "temperature":
        float(
            weather["temperature"]
        ),

    "rain":
        float(
            weather["rain"]
        ),

    "precipitation":
        float(
            weather["precipitation"]
        ),

    "track_condition":
        str(
            weather["condition"]
        ),

    "weather_impact":
        str(
            weather["impact"]
        ),


    # --------------------------------------------------------
    # DEGRADATION
    # --------------------------------------------------------

    "degradation_rates": {
        str(key): float(value)
        for key, value
        in degradation_rates.items()
    },


    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    "reason":
        str(
            weather_strategy["reason"]
        )
}


# ============================================================
# SAVE JSON
# ============================================================

BACKEND_RESULT_PATH = (
    "data/processed/backend_result.json"
)


with open(
    BACKEND_RESULT_PATH,
    "w",
    encoding="utf-8"
) as result_file:

    json.dump(
        backend_result,
        result_file,
        indent=4
    )


print(
    "\nStructured backend result saved to:"
)

print(
    BACKEND_RESULT_PATH
)
