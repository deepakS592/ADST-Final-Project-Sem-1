import pandas as pd
import numpy as np
import requests

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ============================================================
# F1 PRE-RACE STRATEGY SYSTEM
# ============================================================

print("\n========================================")
print("      F1 PRE-RACE STRATEGY SYSTEM")
print("========================================")


# ============================================================
# RACE INFORMATION
# ============================================================

race_information = {

    "Bahrain Grand Prix": {
        "date": "2024-03-02",
        "location": "Sakhir, Bahrain",
        "latitude": 26.0325,
        "longitude": 50.5106,
        "laps": 57
    },

    "Emilia-Romagna Grand Prix": {
        "date": "2024-05-19",
        "location": "Imola, Italy",
        "latitude": 44.3439,
        "longitude": 11.7167,
        "laps": 63
    },

    "British Grand Prix": {
        "date": "2024-07-07",
        "location": "Silverstone, United Kingdom",
        "latitude": 52.0733,
        "longitude": -1.0147,
        "laps": 52
    },

    "Belgian Grand Prix": {
        "date": "2024-07-28",
        "location": "Spa-Francorchamps, Belgium",
        "latitude": 50.4372,
        "longitude": 5.9714,
        "laps": 44
    },

    "Italian Grand Prix": {
        "date": "2024-09-01",
        "location": "Monza, Italy",
        "latitude": 45.6156,
        "longitude": 9.2811,
        "laps": 53
    }
}


# ============================================================
# SETTINGS
# ============================================================

DRY_COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD"
]

PIT_STOP_LOSS = 22.0

MIN_ONE_STOP_STINT = 15

MIN_TWO_STOP_STINT = 12

SECOND_STOP_RISK = 4.0


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(
        "data/raw/2024_multi_race_laps.csv"
    )

except FileNotFoundError:

    print("\nERROR: Dataset not found.")

    raise SystemExit


print("\nLap dataset loaded!")
print("Records:", len(df))


# ============================================================
# CLEAN DATA
# ============================================================

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


df = df[
    (df["LapTime"] > 50) &
    (df["LapTime"] < 150)
].copy()


df = df[
    df["TyreLife"] >= 2
].copy()


print(
    "Records after cleaning:",
    len(df)
)


# ============================================================
# RACE PROGRESS
# ============================================================

df["RaceDistance"] = df.groupby(
    "Race"
)["LapNumber"].transform("max")


df["RaceProgress"] = (
    df["LapNumber"] /
    df["RaceDistance"]
)


# ============================================================
# SELECT RACE
# ============================================================

print("\n========================================")
print("          AVAILABLE RACES")
print("========================================")


races = list(
    race_information.keys()
)


for i, race in enumerate(
    races,
    start=1
):

    print(
        i,
        "-",
        race
    )


while True:

    try:

        race_choice = int(
            input(
                "\nSelect race number: "
            )
        )

        if 1 <= race_choice <= len(races):

            break

        print(
            "Enter a valid race number."
        )

    except ValueError:

        print(
            "Enter a number."
        )


selected_race = races[
    race_choice - 1
]


race_info = race_information[
    selected_race
]


race_laps = race_info["laps"]


print(
    "\nSelected race:",
    selected_race
)

print(
    "Circuit:",
    race_info["location"]
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
    race_data["Driver"].unique()
)


print("\nAvailable drivers:")


for i, driver in enumerate(
    drivers,
    start=1
):

    print(
        i,
        "-",
        driver
    )


while True:

    try:

        driver_choice = int(
            input(
                "\nSelect driver number: "
            )
        )

        if 1 <= driver_choice <= len(drivers):

            break

        print(
            "Enter a valid driver number."
        )

    except ValueError:

        print(
            "Enter a number."
        )


selected_driver = drivers[
    driver_choice - 1
]


print(
    "\nSelected driver:",
    selected_driver
)


# ============================================================
# DRIVER HISTORY
# ============================================================

driver_history = df[
    df["Driver"] == selected_driver
].copy()


print(
    "\nHistorical records for driver:",
    len(driver_history)
)


available_compounds = [

    compound

    for compound in DRY_COMPOUNDS

    if compound in
    driver_history["Compound"].unique()

]


print("\nAvailable tyres:")


for compound in sorted(
    driver_history["Compound"].unique()
):

    count = len(
        driver_history[
            driver_history["Compound"] == compound
        ]
    )

    print(
        compound,
        "→",
        count,
        "records"
    )


# ============================================================
# AUTOMATIC HISTORICAL WEATHER
# ============================================================

print("\n========================================")
print("       AUTOMATIC WEATHER FORECAST")
print("========================================")


weather_temperature = 20.0
weather_rain = 0.0
weather_precipitation = 0.0
weather_condition = "DRY"
weather_impact = "LOW"


def weather_code_name(code):

    code = int(code)

    if code == 0:
        return "Clear"

    if code in [1, 2, 3]:
        return "Cloudy"

    if code in [45, 48]:
        return "Fog"

    if code in [51, 53, 55, 56, 57]:
        return "Drizzle"

    if code in [61, 63, 65, 66, 67]:
        return "Rain"

    if code in [71, 73, 75, 77]:
        return "Snow"

    if code in [80, 81, 82]:
        return "Rain Showers"

    if code in [95, 96, 99]:
        return "Thunderstorm"

    return "Unknown"


def fetch_weather():

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {

        "latitude":
            race_info["latitude"],

        "longitude":
            race_info["longitude"],

        "start_date":
            race_info["date"],

        "end_date":
            race_info["date"],

        "hourly":
            "temperature_2m,precipitation,rain,weather_code",

        "timezone":
            "auto",

        "temperature_unit":
            "celsius",

        "precipitation_unit":
            "mm"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


try:

    print(
        "\nConnecting to Open-Meteo..."
    )

    weather_data = fetch_weather()

    print(
        "Historical weather data received."
    )

    hourly = weather_data["hourly"]

    times = hourly["time"]

    temperatures = hourly[
        "temperature_2m"
    ]

    precipitation = hourly[
        "precipitation"
    ]

    rain = hourly[
        "rain"
    ]

    weather_codes = hourly[
        "weather_code"
    ]


    indexes = []


    for i, timestamp in enumerate(times):

        try:

            hour = int(
                timestamp[11:13]
            )

            if 11 <= hour <= 17:

                indexes.append(i)

        except:

            pass


    if not indexes:

        indexes = list(
            range(len(times))
        )


    temp_values = [

        temperatures[i]

        for i in indexes

        if temperatures[i] is not None

    ]


    rain_values = [

        rain[i]

        for i in indexes

        if rain[i] is not None

    ]


    precipitation_values = [

        precipitation[i]

        for i in indexes

        if precipitation[i] is not None

    ]


    weather_code_values = [

        weather_codes[i]

        for i in indexes

        if weather_codes[i] is not None

    ]


    if temp_values:

        weather_temperature = (
            sum(temp_values)
            /
            len(temp_values)
        )


    weather_rain = sum(
        rain_values
    )


    weather_precipitation = sum(
        precipitation_values
    )


    if weather_rain >= 5:

        weather_condition = "WET"

    elif weather_rain > 0:

        weather_condition = "DAMP"

    else:

        weather_condition = "DRY"


    if weather_condition == "WET":

        weather_impact = "HIGH"

    elif weather_condition == "DAMP":

        weather_impact = "MEDIUM"

    else:

        weather_impact = "LOW"


    weather_names = [

        weather_code_name(code)

        for code in weather_code_values

    ]


    if weather_names:

        dominant_weather = max(
            set(weather_names),
            key=weather_names.count
        )

    else:

        dominant_weather = "Unknown"


    print(
        "\nWeather analysis:"
    )

    print(
        "Average temperature:",
        round(
            weather_temperature,
            2
        ),
        "°C"
    )

    print(
        "Total rain:",
        round(
            weather_rain,
            2
        ),
        "mm"
    )

    print(
        "Total precipitation:",
        round(
            weather_precipitation,
            2
        ),
        "mm"
    )

    print(
        "Dominant weather:",
        dominant_weather
    )

    print(
        "Track condition:",
        weather_condition
    )

    print(
        "Weather impact:",
        weather_impact
    )


except Exception as error:

    print(
        "\nWARNING: Weather data unavailable."
    )

    print(
        "Using dry-weather assumption."
    )

    print(
        "Reason:",
        error
    )


# ============================================================
# TRAIN LAP-TIME MODEL
# ============================================================

print("\n========================================")
print("       TRAINING LAP-TIME MODEL")
print("========================================")


driver_encoder = LabelEncoder()

race_encoder = LabelEncoder()

compound_encoder = LabelEncoder()


df["DriverCode"] = driver_encoder.fit_transform(
    df["Driver"].astype(str)
)


df["RaceCode"] = race_encoder.fit_transform(
    df["Race"].astype(str)
)


df["CompoundCode"] = compound_encoder.fit_transform(
    df["Compound"].astype(str)
)


features = [

    "DriverCode",

    "RaceCode",

    "CompoundCode",

    "TyreLife",

    "LapNumber",

    "RaceProgress"

]


X = df[features]

y = df["LapTime"]


lap_model = RandomForestRegressor(

    n_estimators=150,

    max_depth=14,

    min_samples_leaf=5,

    random_state=42,

    n_jobs=-1

)


lap_model.fit(
    X,
    y
)


print(
    "Lap-time prediction model trained."
)


# ============================================================
# BUILD COMPLETE PREDICTION CACHE
# ============================================================

print("\n========================================")
print("       BUILDING TYRE PREDICTION CACHE")
print("========================================")


cache_rows = []


driver_code = driver_encoder.transform(
    [selected_driver]
)[0]


race_code = race_encoder.transform(
    [selected_race]
)[0]


# We only need ages that can realistically
# appear in our strategies.

max_age = race_laps


for compound in available_compounds:

    compound_code = compound_encoder.transform(
        [compound]
    )[0]


    rows = []


    for lap in range(
        1,
        race_laps + 1
    ):

        for tyre_age in range(
            1,
            min(max_age, 35) + 1
        ):

            rows.append({

                "DriverCode":
                    driver_code,

                "RaceCode":
                    race_code,

                "CompoundCode":
                    compound_code,

                "TyreLife":
                    tyre_age,

                "LapNumber":
                    lap,

                "RaceProgress":
                    lap / race_laps,

                "Compound":
                    compound,

                "Lap":
                    lap,

                "Age":
                    tyre_age

            })


    batch = pd.DataFrame(rows)


    predictions = lap_model.predict(
        batch[features]
    )


    batch["PredictedLapTime"] = predictions


    cache_rows.append(
        batch[
            [
                "Compound",
                "Lap",
                "Age",
                "PredictedLapTime"
            ]
        ]
    )


prediction_cache = pd.concat(
    cache_rows,
    ignore_index=True
)


prediction_cache = prediction_cache.set_index(
    [
        "Compound",
        "Lap",
        "Age"
    ]
)


print(
    "Prediction cache ready."
)


# ============================================================
# FAST LAP-TIME LOOKUP
# ============================================================

def get_cached_lap_time(

    compound,

    lap,

    tyre_age

):

    try:

        value = prediction_cache.loc[
            (
                compound,
                lap,
                tyre_age
            ),
            "PredictedLapTime"
        ]

        return float(value)

    except:

        return None


# ============================================================
# DISPLAY PREDICTED TYRE PERFORMANCE
# ============================================================

print("\n========================================")
print("       PREDICTED TYRE PERFORMANCE")
print("========================================")


for compound in available_compounds:

    values = []


    for lap in range(
        10,
        min(
            race_laps,
            30
        ) + 1
    ):

        value = get_cached_lap_time(

            compound,

            lap,

            5

        )


        if value is not None:

            values.append(value)


    if values:

        print(

            f"{compound:<15}",

            f"{np.mean(values):.3f}",

            "sec/lap"

        )


# ============================================================
# STINT SIMULATION
# ============================================================

def simulate_stint(

    compound,

    start_lap,

    end_lap

):

    if end_lap < start_lap:

        return None


    total_time = 0.0


    for lap in range(

        start_lap,

        end_lap + 1

    ):

        tyre_age = (
            lap
            -
            start_lap
            +
            1
        )


        if tyre_age > 35:

            return None


        lap_time = get_cached_lap_time(

            compound,

            lap,

            tyre_age

        )


        if lap_time is None:

            return None


        total_time += lap_time


    return total_time


# ============================================================
# ONE STOP
# ============================================================

def simulate_one_stop(

    first,

    second,

    pit_lap

):

    first_length = pit_lap

    second_length = (
        race_laps
        -
        pit_lap
    )


    if (

        first_length
        <
        MIN_ONE_STOP_STINT

        or

        second_length
        <
        MIN_ONE_STOP_STINT

    ):

        return None


    first_time = simulate_stint(

        first,

        1,

        pit_lap

    )


    second_time = simulate_stint(

        second,

        pit_lap + 1,

        race_laps

    )


    if (

        first_time is None

        or

        second_time is None

    ):

        return None


    return (

        first_time

        +

        second_time

        +

        PIT_STOP_LOSS

    )


# ============================================================
# TWO STOP
# ============================================================

def simulate_two_stop(

    first,

    second,

    third,

    pit_1,

    pit_2

):

    # At least two different compounds
    if len(
        set([
            first,
            second,
            third
        ])
    ) < 2:

        return None


    stint_1 = pit_1

    stint_2 = pit_2 - pit_1

    stint_3 = race_laps - pit_2


    if (

        stint_1
        <
        MIN_TWO_STOP_STINT

        or

        stint_2
        <
        MIN_TWO_STOP_STINT

        or

        stint_3
        <
        MIN_TWO_STOP_STINT

    ):

        return None


    first_time = simulate_stint(

        first,

        1,

        pit_1

    )


    second_time = simulate_stint(

        second,

        pit_1 + 1,

        pit_2

    )


    third_time = simulate_stint(

        third,

        pit_2 + 1,

        race_laps

    )


    if (

        first_time is None

        or

        second_time is None

        or

        third_time is None

    ):

        return None


    return (

        first_time

        +

        second_time

        +

        third_time

        +

        (2 * PIT_STOP_LOSS)

        +

        SECOND_STOP_RISK

    )


# ============================================================
# SEARCH STRATEGIES
# ============================================================

print("\n========================================")
print("        SEARCHING PIT STRATEGIES")
print("========================================")


strategies = []


# ============================================================
# ONE STOP
# ============================================================

for first in available_compounds:

    for second in available_compounds:

        if first == second:

            continue


        for pit_lap in range(

            MIN_ONE_STOP_STINT,

            race_laps
            -
            MIN_ONE_STOP_STINT
            +
            1

        ):

            result = simulate_one_stop(

                first,

                second,

                pit_lap

            )


            if result is None:

                continue


            strategies.append({

                "StrategyType":
                    "1-STOP",

                "FirstTyre":
                    first,

                "SecondTyre":
                    second,

                "ThirdTyre":
                    "-",

                "PitLap1":
                    pit_lap,

                "PitLap2":
                    np.nan,

                "EstimatedRaceTime":
                    result

            })


# ============================================================
# TWO STOP
# ============================================================

for first in available_compounds:

    for second in available_compounds:

        for third in available_compounds:

            # Need at least two different compounds
            if len(
                set([
                    first,
                    second,
                    third
                ])
            ) < 2:

                continue


            for pit_1 in range(

                MIN_TWO_STOP_STINT,

                race_laps
                -
                (2 * MIN_TWO_STOP_STINT)
                +
                1

            ):

                for pit_2 in range(

                    pit_1
                    +
                    MIN_TWO_STOP_STINT,

                    race_laps
                    -
                    MIN_TWO_STOP_STINT
                    +
                    1

                ):

                    result = simulate_two_stop(

                        first,

                        second,

                        third,

                        pit_1,

                        pit_2

                    )


                    if result is None:

                        continue


                    strategies.append({

                        "StrategyType":
                            "2-STOP",

                        "FirstTyre":
                            first,

                        "SecondTyre":
                            second,

                        "ThirdTyre":
                            third,

                        "PitLap1":
                            pit_1,

                        "PitLap2":
                            pit_2,

                        "EstimatedRaceTime":
                            result

                    })


results = pd.DataFrame(
    strategies
)


if results.empty:

    print(
        "\nNo suitable strategy found."
    )

    raise SystemExit


# ============================================================
# STRATEGY PENALTY
# ============================================================

def strategy_penalty(row):

    penalty = 0.0


    if row["StrategyType"] == "2-STOP":

        penalty += SECOND_STOP_RISK


        pit_1 = int(
            row["PitLap1"]
        )

        pit_2 = int(
            row["PitLap2"]
        )


        stint_lengths = [

            pit_1,

            pit_2 - pit_1,

            race_laps - pit_2

        ]


        if max(stint_lengths) > 32:

            penalty += 3.0


        if min(stint_lengths) < 14:

            penalty += 3.0


    return penalty


results[
    "StrategyPenalty"
] = results.apply(

    strategy_penalty,

    axis=1

)


results[
    "StrategyScore"
] = (

    results["EstimatedRaceTime"]

    +

    results["StrategyPenalty"]

)


results = results.sort_values(
    "StrategyScore"
).reset_index(
    drop=True
)


# ============================================================
# BEST ONE STOP
# ============================================================

one_stop_results = results[
    results["StrategyType"]
    == "1-STOP"
]


two_stop_results = results[
    results["StrategyType"]
    == "2-STOP"
]


best_one_stop = (

    one_stop_results.iloc[0]

    if not one_stop_results.empty

    else None

)


best_two_stop = (

    two_stop_results.iloc[0]

    if not two_stop_results.empty

    else None

)


# ============================================================
# SELECT FINAL STRATEGY
# ============================================================

if (

    best_one_stop is not None

    and

    best_two_stop is not None

):

    one_score = float(
        best_one_stop[
            "StrategyScore"
        ]
    )


    two_score = float(
        best_two_stop[
            "StrategyScore"
        ]
    )


    # 2-stop must be at least
    # 8 seconds better.

    if two_score < one_score - 8:

        best_strategy = best_two_stop

    else:

        best_strategy = best_one_stop


else:

    best_strategy = results.iloc[0]


# ============================================================
# PIT STOP ML MODEL
# ============================================================

print(
    "\nTraining pit-stop prediction model..."
)


ml_model = None


try:

    pit_data = pd.read_csv(

        "data/processed/"
        "pit_strategy_training_data.csv"

    )


    pit_data = pit_data.dropna(

        subset=[

            "Driver",

            "PreviousTyre",

            "PreviousTyreAge",

            "PitLap",

            "NewTyre"

        ]

    ).copy()


    driver_ml_encoder = LabelEncoder()

    tyre_ml_encoder = LabelEncoder()

    target_ml_encoder = LabelEncoder()


    pit_data[
        "DriverCode"
    ] = driver_ml_encoder.fit_transform(

        pit_data[
            "Driver"
        ].astype(str)

    )


    pit_data[
        "TyreCode"
    ] = tyre_ml_encoder.fit_transform(

        pit_data[
            "PreviousTyre"
        ].astype(str)

    )


    pit_data[
        "Target"
    ] = target_ml_encoder.fit_transform(

        pit_data[
            "NewTyre"
        ].astype(str)

    )


    X_ml = pit_data[

        [

            "DriverCode",

            "TyreCode",

            "PreviousTyreAge",

            "PitLap"

        ]

    ]


    y_ml = pit_data[
        "Target"
    ]


    ml_model = RandomForestClassifier(

        n_estimators=200,

        max_depth=8,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42

    )


    ml_model.fit(
        X_ml,
        y_ml
    )


    print(
        "Pit-stop ML model trained."
    )


except Exception as error:

    print(
        "Pit-stop ML model unavailable."
    )

    print(
        "Reason:",
        error
    )


# ============================================================
# ML PREDICTION
# ============================================================

def predict_next_tyre(

    current_tyre,

    tyre_age,

    race_lap

):

    if ml_model is None:

        return None, {}


    if selected_driver not in \
       driver_ml_encoder.classes_:

        return None, {}


    if current_tyre not in \
       tyre_ml_encoder.classes_:

        return None, {}


    driver_code = driver_ml_encoder.transform(
        [selected_driver]
    )[0]


    tyre_code = tyre_ml_encoder.transform(
        [current_tyre]
    )[0]


    input_data = pd.DataFrame({

        "DriverCode":
            [driver_code],

        "TyreCode":
            [tyre_code],

        "PreviousTyreAge":
            [tyre_age],

        "PitLap":
            [race_lap]

    })


    prediction = ml_model.predict(
        input_data
    )[0]


    predicted_tyre = (

        target_ml_encoder.inverse_transform(
            [prediction]
        )[0]

    )


    probabilities = ml_model.predict_proba(
        input_data
    )[0]


    probability_dict = {}


    for class_number, probability in zip(

        ml_model.classes_,

        probabilities

    ):

        tyre_name = (

            target_ml_encoder.inverse_transform(
                [class_number]
            )[0]

        )


        probability_dict[
            tyre_name
        ] = float(
            probability
        )


    return (
        predicted_tyre,
        probability_dict
    )


# ============================================================
# ML PREDICTION AT BEST PIT
# ============================================================

ml_current_tyre = best_strategy[
    "FirstTyre"
]


ml_pit_lap = int(
    best_strategy[
        "PitLap1"
    ]
)


ml_prediction, ml_probabilities = (

    predict_next_tyre(

        ml_current_tyre,

        ml_pit_lap,

        ml_pit_lap

    )

)


# ============================================================
# CONFIDENCE
# ============================================================

confidence = 60.0


if len(driver_history) >= 250:

    confidence += 10

elif len(driver_history) >= 150:

    confidence += 5

else:

    confidence -= 5


if (

    best_one_stop is not None

    and

    best_two_stop is not None

):

    gap = abs(

        best_one_stop[
            "StrategyScore"
        ]

        -

        best_two_stop[
            "StrategyScore"
        ]

    )


    if gap >= 10:

        confidence += 10

    elif gap < 3:

        confidence -= 10


simulator_next_tyre = best_strategy[
    "SecondTyre"
]


if ml_prediction is not None:

    if ml_prediction == simulator_next_tyre:

        confidence += 10

    else:

        confidence -= 5


if weather_data is not None:

    confidence += 5


confidence = float(
    np.clip(
        confidence,
        0,
        100
    )
)


if confidence >= 75:

    confidence_level = "HIGH"

elif confidence >= 55:

    confidence_level = "MEDIUM"

else:

    confidence_level = "LOW"


# ============================================================
# WEATHER OVERRIDE
# ============================================================

if weather_condition == "WET":

    final_strategy = (
        "INTERMEDIATE / FULL WET"
    )

    weather_override = True

    final_reason = (
        "Wet conditions override "
        "the normal dry strategy."
    )


elif weather_condition == "DAMP":

    final_strategy = (
        "INTERMEDIATE → DRY TRANSITION"
    )

    weather_override = True

    final_reason = (
        "Damp conditions require "
        "intermediate tyres initially."
    )


else:

    weather_override = False


    if best_strategy[
        "StrategyType"
    ] == "1-STOP":

        final_strategy = (

            best_strategy[
                "FirstTyre"
            ]

            + " → "

            + best_strategy[
                "SecondTyre"
            ]

        )

    else:

        final_strategy = (

            best_strategy[
                "FirstTyre"
            ]

            + " → "

            + best_strategy[
                "SecondTyre"
            ]

            + " → "

            + best_strategy[
                "ThirdTyre"
            ]

        )


    final_reason = (
        "Dry conditions allow the "
        "tyre strategy model to determine "
        "the recommended pit-stop plan."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("       FINAL PRE-RACE STRATEGY")
print("========================================")


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
    race_info["location"]
)

print(
    "Race distance:",
    race_laps,
    "laps"
)


# ============================================================
# WEATHER OUTPUT
# ============================================================

print("\n----------------------------------------")
print("AUTOMATIC WEATHER")
print("----------------------------------------")


print(
    "Average temperature:",
    round(
        weather_temperature,
        2
    ),
    "°C"
)


print(
    "Rain:",
    round(
        weather_rain,
        2
    ),
    "mm"
)


print(
    "Precipitation:",
    round(
        weather_precipitation,
        2
    ),
    "mm"
)


print(
    "Track condition:",
    weather_condition
)


print(
    "Weather impact:",
    weather_impact
)


# ============================================================
# 1 STOP VS 2 STOP
# ============================================================

print("\n========================================")
print("          1-STOP vs 2-STOP")
print("========================================")


if best_one_stop is not None:

    print(
        "\nBEST 1-STOP"
    )


    print(
        "Strategy:",
        best_one_stop["FirstTyre"],
        "→",
        best_one_stop["SecondTyre"]
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


if best_two_stop is not None:

    print(
        "\nBEST 2-STOP"
    )


    print(
        "Strategy:",
        best_two_stop["FirstTyre"],
        "→",
        best_two_stop["SecondTyre"],
        "→",
        best_two_stop["ThirdTyre"]
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


if (

    best_one_stop is not None

    and

    best_two_stop is not None

):

    score_difference = abs(

        best_one_stop[
            "StrategyScore"
        ]

        -

        best_two_stop[
            "StrategyScore"
        ]

    )


    print(
        "\nStrategy score difference:",
        round(
            score_difference,
            2
        ),
        "seconds"
    )


    print(
        "Preferred strategy:",

        best_strategy[
            "StrategyType"
        ]
    )


# ============================================================
# BEST STRATEGY
# ============================================================

print("\n========================================")
print("       BEST DRY STRATEGY")
print("========================================")


if best_strategy[
    "StrategyType"
] == "1-STOP":

    print(
        "\nStrategy:",
        best_strategy[
            "FirstTyre"
        ],
        "→",
        best_strategy[
            "SecondTyre"
        ]
    )


    print(
        "Pit lap:",
        int(
            best_strategy[
                "PitLap1"
            ]
        )
    )


else:

    print(
        "\nStrategy:",
        best_strategy[
            "FirstTyre"
        ],
        "→",
        best_strategy[
            "SecondTyre"
        ],
        "→",
        best_strategy[
            "ThirdTyre"
        ]
    )


    print(
        "First pit:",
        int(
            best_strategy[
                "PitLap1"
            ]
        )
    )


    print(
        "Second pit:",
        int(
            best_strategy[
                "PitLap2"
            ]
        )
    )


print(
    "Estimated race time:",
    round(
        best_strategy[
            "EstimatedRaceTime"
        ],
        2
    ),
    "seconds"
)


# ============================================================
# STINT ANALYSIS
# ============================================================

print("\n----------------------------------------")
print("STINT ANALYSIS")
print("----------------------------------------")


if best_strategy[
    "StrategyType"
] == "1-STOP":

    pit = int(
        best_strategy[
            "PitLap1"
        ]
    )


    print(
        best_strategy[
            "FirstTyre"
        ],
        "stint:",
        pit,
        "laps"
    )


    print(
        best_strategy[
            "SecondTyre"
        ],
        "stint:",
        race_laps - pit,
        "laps"
    )


else:

    pit_1 = int(
        best_strategy[
            "PitLap1"
        ]
    )


    pit_2 = int(
        best_strategy[
            "PitLap2"
        ]
    )


    print(
        best_strategy[
            "FirstTyre"
        ],
        "stint:",
        pit_1,
        "laps"
    )


    print(
        best_strategy[
            "SecondTyre"
        ],
        "stint:",
        pit_2 - pit_1,
        "laps"
    )


    print(
        best_strategy[
            "ThirdTyre"
        ],
        "stint:",
        race_laps - pit_2,
        "laps"
    )


# ============================================================
# ML PREDICTION
# ============================================================

print("\n========================================")
print("        ML TYRE PREDICTION")
print("========================================")


if ml_prediction is not None:

    print(
        "\nCurrent tyre:",
        ml_current_tyre
    )


    print(
        "Tyre age:",
        ml_pit_lap
    )


    print(
        "Race lap:",
        ml_pit_lap
    )


    print(
        "\nML predicted next tyre:",
        ml_prediction
    )


    print(
        "\nTyre probabilities:"
    )


    for tyre, probability in sorted(

        ml_probabilities.items(),

        key=lambda item: item[1],

        reverse=True

    ):

        print(

            f"{tyre:<15}",

            f"{probability * 100:.2f}%"

        )


    print(
        "\nSimulator next tyre:",
        simulator_next_tyre
    )


    if ml_prediction == simulator_next_tyre:

        print(
            "\n✓ ML and simulator agree."
        )

    else:

        print(
            "\n⚠ ML and simulator disagree."
        )


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print("\n========================================")
print("       FINAL RECOMMENDATION")
print("========================================")


print(
    "\nFinal strategy:"
)


print(
    final_strategy
)


print(
    "\nReason:"
)


print(
    final_reason
)


if weather_override:

    print(
        "\n⚠ WEATHER OVERRIDE ACTIVE"
    )

else:

    print(
        "\n✓ Historical dry strategy retained."
    )


# ============================================================
# CONFIDENCE
# ============================================================

print("\n========================================")
print("          MODEL CONFIDENCE")
print("========================================")


print(
    "\nConfidence score:",
    round(
        confidence,
        1
    ),
    "%"
)


print(
    "Confidence level:",
    confidence_level
)


# ============================================================
# PIT WINDOW
# ============================================================

print("\n========================================")
print("             PIT WINDOW")
print("========================================")


first_pit = int(
    best_strategy[
        "PitLap1"
    ]
)


print(

    "Recommended first pit window:",

    max(
        1,
        first_pit - 2
    ),

    "-",

    min(
        race_laps,
        first_pit + 2
    )

)


if best_strategy[
    "StrategyType"
] == "2-STOP":

    second_pit = int(
        best_strategy[
            "PitLap2"
        ]
    )


    print(

        "Recommended second pit window:",

        max(
            1,
            second_pit - 2
        ),

        "-",

        min(
            race_laps,
            second_pit + 2
        )

    )


# ============================================================
# TOP 10
# ============================================================

print("\n========================================")
print("          TOP 10 STRATEGIES")
print("========================================")


columns = [

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


print(

    results[
        columns
    ].head(10).to_string(
        index=False
    )

)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    "data/processed/"
    "strategy_simulation_results.csv"
)


results.to_csv(
    output_file,
    index=False
)


print(
    "\nResults saved to:"
)


print(
    output_file
)


# ============================================================
# EXPLANATION
# ============================================================

print("\n========================================")
print("       STRATEGY EXPLANATION")
print("========================================")


print(
    "\nThe system considered:"
)

print(
    "✓ Driver historical performance"
)

print(
    "✓ ML lap-time prediction"
)

print(
    "✓ Tyre compound"
)

print(
    "✓ Tyre age"
)

print(
    "✓ Race lap"
)

print(
    "✓ Race progress"
)

print(
    "✓ One-stop strategies"
)

print(
    "✓ Two-stop strategies"
)

print(
    "✓ Pit-stop time loss"
)

print(
    "✓ Strategy realism"
)

print(
    "✓ Minimum stint lengths"
)

print(
    "✓ Two-compound dry-race rule"
)

print(
    "✓ ML pit-stop prediction"
)

print(
    "✓ Automatic historical weather"
)

print(
    "✓ Model confidence"
)


print("\n========================================")
print("     F1 PRE-RACE SYSTEM READY!")
print("========================================")