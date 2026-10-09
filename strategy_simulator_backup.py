import os
import warnings
import requests
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================
# F1 PRE-RACE STRATEGY SYSTEM - OPTIMIZED VERSION
# ============================================================

print("\n========================================")
print("      F1 PRE-RACE STRATEGY SYSTEM")
print("========================================")

DATA_PATH = "data/raw/2024_multi_race_laps.csv"
PIT_DATA_PATH = "data/processed/pit_strategy_training_data.csv"
RESULT_PATH = "data/processed/strategy_simulation_results.csv"
WEATHER_PATH = "data/processed/weather_analysis.csv"

os.makedirs("data/processed", exist_ok=True)

RACES = {
    "Bahrain Grand Prix": {
        "laps": 57, "circuit": "Sakhir, Bahrain",
        "lat": 26.0325, "lon": 50.5106, "date": "2024-03-02"
    },
    "Emilia-Romagna Grand Prix": {
        "laps": 63, "circuit": "Imola, Italy",
        "lat": 44.3439, "lon": 11.7167, "date": "2024-05-19"
    },
    "British Grand Prix": {
        "laps": 52, "circuit": "Silverstone, UK",
        "lat": 52.0786, "lon": -1.0169, "date": "2024-07-07"
    },
    "Belgian Grand Prix": {
        "laps": 44, "circuit": "Spa-Francorchamps, Belgium",
        "lat": 50.4372, "lon": 5.9714, "date": "2024-07-28"
    },
    "Italian Grand Prix": {
        "laps": 53, "circuit": "Monza, Italy",
        "lat": 45.6156, "lon": 9.2811, "date": "2024-09-01"
    }
}

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
ALL_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]

PIT_LOSS = 22.0
MIN_STINT = 8
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nLap dataset loaded!")
print("Records:", len(df))

for col in ["LapNumber", "TyreLife", "LapTime"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["Race", "Driver", "Compound",
            "LapNumber", "TyreLife", "LapTime"]
).copy()

df["Compound"] = df["Compound"].astype(str).str.upper().str.strip()
df["Driver"] = df["Driver"].astype(str).str.upper().str.strip()

df = df[
    (df["LapTime"] > 50) &
    (df["LapTime"] < 150) &
    (df["TyreLife"] >= 2)
].copy()

print("Records after cleaning:", len(df))


# ============================================================
# SELECT RACE
# ============================================================

print("\n========================================")
print("          AVAILABLE RACES")
print("========================================")

race_names = list(RACES.keys())

for i, race in enumerate(race_names, 1):
    print(f"{i} - {race}")

while True:
    try:
        race_choice = int(input("\nSelect race number: "))
        if 1 <= race_choice <= len(race_names):
            break
    except ValueError:
        pass
    print("Please enter a valid race number.")

selected_race = race_names[race_choice - 1]
race_info = RACES[selected_race]
race_laps = race_info["laps"]

print("\nSelected race:", selected_race)
print("Circuit:", race_info["circuit"])
print("Race date:", race_info["date"])
print("Race distance:", race_laps, "laps")


# ============================================================
# SELECT DRIVER
# ============================================================

race_data = df[df["Race"] == selected_race].copy()
drivers = sorted(race_data["Driver"].unique())

print("\nAvailable drivers:")

for i, driver in enumerate(drivers, 1):
    print(f"{i} - {driver}")

while True:
    try:
        driver_choice = int(input("\nSelect driver number: "))
        if 1 <= driver_choice <= len(drivers):
            break
    except ValueError:
        pass
    print("Please enter a valid driver number.")

selected_driver = drivers[driver_choice - 1]
driver_history = df[df["Driver"] == selected_driver].copy()

print("\nSelected driver:", selected_driver)
print("\nHistorical records for driver:", len(driver_history))

available = sorted(driver_history["Compound"].unique())

print("\nAvailable tyres:")

for compound in available:
    count = len(
        driver_history[
            driver_history["Compound"] == compound
        ]
    )
    print(f"{compound:<14} → {count} records")


# ============================================================
# AUTOMATIC HISTORICAL WEATHER
# ============================================================

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
        return "Rain showers"
    if code in [85, 86]:
        return "Snow showers"
    if code in [95, 96, 99]:
        return "Thunderstorm"

    return "Unknown"


def get_weather():

    print("\n========================================")
    print("       AUTOMATIC WEATHER FORECAST")
    print("========================================")

    print("\nConnecting to Open-Meteo...")

    params = {
        "latitude": race_info["lat"],
        "longitude": race_info["lon"],
        "hourly": (
            "temperature_2m,"
            "precipitation_probability,"
            "precipitation,"
            "rain,"
            "weather_code"
        ),
        "timezone": "auto",
        "start_date": race_info["date"],
        "end_date": race_info["date"]
    }

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()
        hourly = data["hourly"]

        weather = pd.DataFrame(hourly)
        weather["weather_name"] = weather["weather_code"].apply(
            weather_code_name
        )

        times = pd.to_datetime(weather["time"])

        # Focus on daytime/race hours.
        mask = times.dt.hour.between(10, 20)
        race_weather = weather[mask].copy()

        if race_weather.empty:
            race_weather = weather.copy()

        avg_temp = float(
            race_weather["temperature_2m"].mean()
        )

        total_rain = float(
            race_weather["rain"].fillna(0).sum()
        )

        total_precip = float(
            race_weather["precipitation"].fillna(0).sum()
        )

        rain_probability = float(
            race_weather[
                "precipitation_probability"
            ].fillna(0).max()
        )

        dominant = (
            race_weather["weather_name"]
            .value_counts()
            .idxmax()
        )

        if total_rain >= 2.0 or total_precip >= 2.0:
            track_condition = "WET"
        elif (
            total_rain > 0
            or total_precip > 0
            or rain_probability >= 40
        ):
            track_condition = "DAMP"
        else:
            track_condition = "DRY"

        if track_condition == "WET":
            impact = "HIGH"
        elif track_condition == "DAMP":
            impact = "MEDIUM"
        elif rain_probability >= 40:
            impact = "MEDIUM"
        elif avg_temp >= 32:
            impact = "MEDIUM"
        else:
            impact = "LOW"

        weather.to_csv(
            WEATHER_PATH,
            index=False
        )

        print("Historical weather data received.")

        print("\nWeather analysis:")
        print(f"Average temperature: {avg_temp:.2f} °C")
        print(f"Total rain: {total_rain:.1f} mm")
        print(f"Total precipitation: {total_precip:.1f} mm")
        print("Dominant weather:", dominant)
        print("Track condition:", track_condition)
        print("Weather impact:", impact)

        return {
            "avg_temp": avg_temp,
            "total_rain": total_rain,
            "total_precip": total_precip,
            "rain_probability": rain_probability,
            "dominant": dominant,
            "track_condition": track_condition,
            "impact": impact
        }

    except Exception as exc:

        print("\nUnable to retrieve weather data.")
        print("Using dry-weather fallback.")
        print("Error:", exc)

        return {
            "avg_temp": 25.0,
            "total_rain": 0.0,
            "total_precip": 0.0,
            "rain_probability": 0.0,
            "dominant": "Unknown",
            "track_condition": "DRY",
            "impact": "LOW"
        }


weather = get_weather()


# ============================================================
# ROBUST TYRE PERFORMANCE MODEL
# ============================================================

print("\n========================================")
print("       TRAINING TYRE PERFORMANCE MODEL")
print("========================================")

model_df = df.copy()


def remove_outliers(group):

    if len(group) < 8:
        return group

    q1 = group["LapTime"].quantile(0.25)
    q3 = group["LapTime"].quantile(0.75)

    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    return group[
        (group["LapTime"] >= low) &
        (group["LapTime"] <= high)
    ]


model_df = (
    model_df
    .groupby(
        ["Driver", "Compound"],
        group_keys=False
    )
    .apply(remove_outliers)
    .reset_index(drop=True)
)

overall_driver_median = float(
    driver_history["LapTime"].median()
)

compound_stats = {}
degradation_rates = {}

for compound in DRY_COMPOUNDS:

    hist = driver_history[
        driver_history["Compound"] == compound
    ].copy()

    if len(hist) < 8:

        hist = model_df[
            model_df["Compound"] == compound
        ].copy()

    if len(hist) < 3:
        continue

    # Use early/mid-stint laps for compound pace.
    base_pool = hist[
        hist["TyreLife"].between(2, 8)
    ]

    if len(base_pool) < 3:
        base_pool = hist

    baseline = float(
        base_pool["LapTime"].median()
    )

    x = hist["TyreLife"].to_numpy(
        dtype=float
    )

    y = hist["LapTime"].to_numpy(
        dtype=float
    )

    if len(hist) >= 8 and np.ptp(x) > 0:
        slope = float(
            np.polyfit(x, y, 1)[0]
        )
    else:
        slope = 0.0

    # Degradation should not become negative.
    slope = max(0.0, slope)

    # Prevent noisy data from dominating the simulator.
    slope = min(slope, 0.20)

    compound_stats[compound] = {
        "baseline": baseline,
        "offset": baseline - overall_driver_median,
        "count": len(hist)
    }

    degradation_rates[compound] = slope


print(
    "\nDriver overall median lap:",
    round(overall_driver_median, 3),
    "seconds"
)

print("\nCompound pace:")

for compound in DRY_COMPOUNDS:

    if compound in compound_stats:

        print(
            f"{compound:<14}"
            f"{compound_stats[compound]['baseline']:.3f} sec/lap"
        )

print("\nDegradation rates:")

for compound in DRY_COMPOUNDS:

    print(
        f"{compound:<14}"
        f"{degradation_rates.get(compound, 0.0):.4f} sec/lap"
    )


# ============================================================
# RANDOM FOREST LAP-TIME MODEL
# ============================================================

train_df = model_df[
    model_df["Compound"].isin(ALL_COMPOUNDS)
].copy()

driver_encoder = LabelEncoder()
compound_encoder = LabelEncoder()

train_df["DriverCode"] = driver_encoder.fit_transform(
    train_df["Driver"].astype(str)
)

train_df["CompoundCode"] = compound_encoder.fit_transform(
    train_df["Compound"].astype(str)
)

race_lap_map = {
    race: info["laps"]
    for race, info in RACES.items()
}

train_df["RaceProgress"] = (
    train_df["LapNumber"]
    /
    train_df["Race"].map(race_lap_map).fillna(60)
)

features = [
    "DriverCode",
    "CompoundCode",
    "TyreLife",
    "LapNumber",
    "RaceProgress"
]

X = train_df[features]
y = train_df["LapTime"]

lap_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lap_model.fit(X, y)

print("\nLap-time model trained.")


# ============================================================
# FAST PREDICTION CACHE
# ============================================================

print("\n========================================")
print("       BUILDING TYRE PREDICTION CACHE")
print("========================================")

prediction_cache = {}


def encode_driver(driver):

    matches = np.where(
        driver_encoder.classes_ == driver
    )[0]

    if len(matches):
        return int(matches[0])

    return 0


def encode_compound(compound):

    matches = np.where(
        compound_encoder.classes_ == compound
    )[0]

    if len(matches):
        return int(matches[0])

    return 0


def statistical_prediction(
    compound,
    tyre_age,
    race_lap
):

    if compound not in compound_stats:

        all_data = model_df[
            model_df["Compound"] == compound
        ]

        if len(all_data) == 0:
            return overall_driver_median

        early = all_data[
            all_data["TyreLife"].between(2, 8)
        ]

        if len(early):
            base = float(
                early["LapTime"].median()
            )
        else:
            base = float(
                all_data["LapTime"].median()
            )

        rate = 0.0

    else:

        base = compound_stats[
            compound
        ]["baseline"]

        rate = degradation_rates.get(
            compound,
            0.0
        )

    age_effect = (
        max(0, tyre_age - 2)
        * rate
    )

    # Small fuel-load improvement over the race.
    fuel_effect = (
        -0.015
        * max(0, race_lap - 1)
    )

    return base + age_effect + fuel_effect


def predict_lap_time(
    compound,
    tyre_age,
    race_lap
):

    key = (
        compound,
        int(tyre_age),
        int(race_lap)
    )

    if key in prediction_cache:
        return prediction_cache[key]

    stat_prediction = statistical_prediction(
        compound,
        tyre_age,
        race_lap
    )

    try:

        input_data = pd.DataFrame([{
            "DriverCode": encode_driver(
                selected_driver
            ),
            "CompoundCode": encode_compound(
                compound
            ),
            "TyreLife": float(tyre_age),
            "LapNumber": float(race_lap),
            "RaceProgress": float(
                race_lap / race_laps
            )
        }])

        ml_prediction = float(
            lap_model.predict(input_data)[0]
        )

        # Statistical model remains the main anchor.
        # ML provides only a controlled correction.
        prediction = (
            0.75 * stat_prediction
            +
            0.25 * ml_prediction
        )

    except Exception:
        prediction = stat_prediction

    prediction = float(
        max(
            50.0,
            min(150.0, prediction)
        )
    )

    prediction_cache[key] = prediction

    return prediction


print("Prediction cache ready.")

print("\n========================================")
print("       PREDICTED TYRE PERFORMANCE")
print("========================================")

for compound in DRY_COMPOUNDS:

    value = predict_lap_time(
        compound,
        tyre_age=5,
        race_lap=5
    )

    print(
        f"{compound:<14}"
        f"{value:.3f} sec/lap"
    )


# ============================================================
# PIT-STOP ML MODEL
# ============================================================

pit_model = None
pit_label_encoder = None
pit_driver_encoder = None
pit_prev_encoder = None

if os.path.exists(PIT_DATA_PATH):

    print("\nTraining pit-stop prediction model...")

    pit_df = pd.read_csv(
        PIT_DATA_PATH
    )

    required = [
        "Driver",
        "PreviousTyre",
        "PreviousTyreAge",
        "PitLap",
        "NewTyre"
    ]

    if all(
        column in pit_df.columns
        for column in required
    ):

        pit_df = pit_df.dropna(
            subset=required
        ).copy()

        pit_driver_encoder = LabelEncoder()
        pit_prev_encoder = LabelEncoder()
        pit_label_encoder = LabelEncoder()

        pit_df["DriverCode"] = (
            pit_driver_encoder.fit_transform(
                pit_df["Driver"].astype(str)
            )
        )

        pit_df["PreviousTyreCode"] = (
            pit_prev_encoder.fit_transform(
                pit_df["PreviousTyre"].astype(str)
            )
        )

        pit_features = [
            "DriverCode",
            "PreviousTyreCode",
            "PreviousTyreAge",
            "PitLap"
        ]

        pit_model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        pit_model.fit(
            pit_df[pit_features],
            pit_label_encoder.fit_transform(
                pit_df["NewTyre"].astype(str)
            )
        )

        print("Pit-stop ML model trained.")


def predict_next_tyre(
    current_tyre,
    tyre_age,
    race_lap
):

    if pit_model is None:
        return None, {}

    if (
        selected_driver
        not in pit_driver_encoder.classes_
    ):
        return None, {}

    if (
        current_tyre
        not in pit_prev_encoder.classes_
    ):
        return None, {}

    driver_matches = np.where(
        pit_driver_encoder.classes_
        == selected_driver
    )[0]

    tyre_matches = np.where(
        pit_prev_encoder.classes_
        == current_tyre
    )[0]

    if not len(driver_matches) or not len(tyre_matches):
        return None, {}

    driver_code = int(
        driver_matches[0]
    )

    tyre_code = int(
        tyre_matches[0]
    )

    sample = pd.DataFrame([{
        "DriverCode": driver_code,
        "PreviousTyreCode": tyre_code,
        "PreviousTyreAge": tyre_age,
        "PitLap": race_lap
    }])

    probabilities = pit_model.predict_proba(
        sample
    )[0]

    result = {}

    for cls, probability in zip(
        pit_model.classes_,
        probabilities
    ):

        label = pit_label_encoder.inverse_transform(
            [cls]
        )[0]

        result[label] = float(
            probability * 100
        )

    best = max(
        result,
        key=result.get
    )

    return best, result


# ============================================================
# STRATEGY SIMULATION
# ============================================================

print("\n========================================")
print("        SEARCHING PIT STRATEGIES")
print("========================================")

current_race_lap = 1


def simulate_stint(
    compound,
    start_lap,
    end_lap
):

    if end_lap < start_lap:
        return None

    total = 0.0
    tyre_age = 0

    for race_lap in range(
        start_lap,
        end_lap + 1
    ):

        tyre_age += 1

        total += predict_lap_time(
            compound,
            tyre_age,
            race_lap
        )

    return total


def simulate_one_stop(
    first,
    second,
    pit_lap
):

    if pit_lap < MIN_STINT:
        return None

    second_length = (
        race_laps - pit_lap
    )

    if second_length < MIN_STINT:
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
        or second_time is None
    ):
        return None

    return (
        first_time
        +
        second_time
        +
        PIT_LOSS
    )


def simulate_two_stop(
    first,
    second,
    third,
    pit1,
    pit2
):

    if pit1 < MIN_STINT:
        return None

    if (
        pit2 - pit1
        < MIN_STINT
    ):
        return None

    if (
        race_laps - pit2
        < MIN_STINT
    ):
        return None

    first_time = simulate_stint(
        first,
        1,
        pit1
    )

    second_time = simulate_stint(
        second,
        pit1 + 1,
        pit2
    )

    third_time = simulate_stint(
        third,
        pit2 + 1,
        race_laps
    )

    if any(
        value is None
        for value in [
            first_time,
            second_time,
            third_time
        ]
    ):
        return None

    return (
        first_time
        +
        second_time
        +
        third_time
        +
        (2 * PIT_LOSS)
    )


strategies = []


# ============================================================
# ONE-STOP SEARCH
# ============================================================

for first in DRY_COMPOUNDS:

    for second in DRY_COMPOUNDS:

        if first == second:
            continue

        for pit in range(
            MIN_STINT,
            race_laps - MIN_STINT + 1
        ):

            result = simulate_one_stop(
                first,
                second,
                pit
            )

            if result is not None:

                strategies.append({
                    "StrategyType": "1-STOP",
                    "FirstTyre": first,
                    "SecondTyre": second,
                    "ThirdTyre": "-",
                    "PitLap1": pit,
                    "PitLap2": np.nan,
                    "EstimatedRaceTime": result
                })


# ============================================================
# TWO-STOP SEARCH
# ============================================================

for first in DRY_COMPOUNDS:

    for second in DRY_COMPOUNDS:

        for third in DRY_COMPOUNDS:

            for pit1 in range(
                MIN_STINT,
                race_laps - (2 * MIN_STINT) + 1
            ):

                for pit2 in range(
                    pit1 + MIN_STINT,
                    race_laps - MIN_STINT + 1
                ):

                    result = simulate_two_stop(
                        first,
                        second,
                        third,
                        pit1,
                        pit2
                    )

                    if result is not None:

                        strategies.append({
                            "StrategyType": "2-STOP",
                            "FirstTyre": first,
                            "SecondTyre": second,
                            "ThirdTyre": third,
                            "PitLap1": pit1,
                            "PitLap2": pit2,
                            "EstimatedRaceTime": result
                        })


results = pd.DataFrame(
    strategies
)

if results.empty:

    print("\nNo suitable strategy found.")
    raise SystemExit


# ============================================================
# STRATEGY REALISM
# ============================================================

def strategy_penalty(row):

    penalty = 0.0

    compounds = [
        row["FirstTyre"],
        row["SecondTyre"],
        row["ThirdTyre"]
    ]

    compounds = [
        c for c in compounds
        if c != "-"
    ]

    unique_compounds = len(
        set(compounds)
    )

    # Avoid unrealistic three-stints-on-one-compound
    # unless it is clearly faster by a large margin.
    if (
        row["StrategyType"] == "2-STOP"
        and unique_compounds == 1
    ):
        penalty += 8.0

    if row["StrategyType"] == "1-STOP":

        stint1 = int(
            row["PitLap1"]
        )

        stint2 = (
            race_laps - stint1
        )

        if (
            stint1 < 10
            or stint2 < 10
        ):
            penalty += 3.0

    else:

        p1 = int(
            row["PitLap1"]
        )

        p2 = int(
            row["PitLap2"]
        )

        if (
            p1 < 10
            or p2 - p1 < 10
            or race_laps - p2 < 10
        ):
            penalty += 3.0

    return penalty


results["StrategyPenalty"] = results.apply(
    strategy_penalty,
    axis=1
)

results["StrategyScore"] = (
    results["EstimatedRaceTime"]
    +
    results["StrategyPenalty"]
)

results = results.sort_values(
    "StrategyScore"
).reset_index(drop=True)


# ============================================================
# BEST STRATEGIES
# ============================================================

one_stop = results[
    results["StrategyType"] == "1-STOP"
]

two_stop = results[
    results["StrategyType"] == "2-STOP"
]

best_1 = (
    one_stop.iloc[0]
    if not one_stop.empty
    else None
)

best_2 = (
    two_stop.iloc[0]
    if not two_stop.empty
    else None
)

if (
    best_1 is not None
    and best_2 is not None
):

    if (
        best_1["StrategyScore"]
        <= best_2["StrategyScore"]
    ):
        best = best_1
    else:
        best = best_2

elif best_1 is not None:

    best = best_1

else:

    best = best_2


# ============================================================
# WEATHER
# ============================================================

if weather["track_condition"] == "WET":

    weather_strategy = "WET"

    weather_reason = (
        "Wet track conditions override the normal "
        "dry strategy."
    )

elif weather["track_condition"] == "DAMP":

    weather_strategy = "DAMP"

    weather_reason = (
        "Damp conditions create uncertainty. "
        "Intermediate tyres should be prepared."
    )

else:

    weather_strategy = "DRY"

    weather_reason = (
        "Dry conditions allow the tyre strategy "
        "model to determine the recommendation."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("       FINAL PRE-RACE STRATEGY")
print("========================================")

print("\nDriver:", selected_driver)
print("Race:", selected_race)
print("Circuit:", race_info["circuit"])
print("Race distance:", race_laps, "laps")

print("\n----------------------------------------")
print("AUTOMATIC WEATHER")
print("----------------------------------------")

print(
    f"Average temperature: "
    f"{weather['avg_temp']:.2f} °C"
)

print(
    f"Rain: "
    f"{weather['total_rain']:.1f} mm"
)

print(
    f"Precipitation: "
    f"{weather['total_precip']:.1f} mm"
)

print(
    "Track condition:",
    weather["track_condition"]
)

print(
    "Weather impact:",
    weather["impact"]
)


# ============================================================
# ONE STOP VS TWO STOP
# ============================================================

print("\n========================================")
print("          1-STOP vs 2-STOP")
print("========================================")

if best_1 is not None:

    print("\nBEST 1-STOP")

    print(
        "Strategy:",
        best_1["FirstTyre"],
        "→",
        best_1["SecondTyre"]
    )

    print(
        "Pit lap:",
        int(best_1["PitLap1"])
    )

    print(
        "Estimated time:",
        round(
            best_1["EstimatedRaceTime"],
            2
        )
    )

if best_2 is not None:

    print("\nBEST 2-STOP")

    print(
        "Strategy:",
        best_2["FirstTyre"],
        "→",
        best_2["SecondTyre"],
        "→",
        best_2["ThirdTyre"]
    )

    print(
        "First pit:",
        int(best_2["PitLap1"])
    )

    print(
        "Second pit:",
        int(best_2["PitLap2"])
    )

    print(
        "Estimated time:",
        round(
            best_2["EstimatedRaceTime"],
            2
        )
    )

if (
    best_1 is not None
    and best_2 is not None
):

    difference = abs(
        best_1["StrategyScore"]
        -
        best_2["StrategyScore"]
    )

    preferred = (
        "1-STOP"
        if (
            best_1["StrategyScore"]
            <
            best_2["StrategyScore"]
        )
        else "2-STOP"
    )

    print(
        "\nStrategy score difference:",
        round(difference, 2),
        "seconds"
    )

    print(
        "Preferred strategy:",
        preferred
    )


# ============================================================
# BEST DRY STRATEGY
# ============================================================

print("\n========================================")
print("       BEST DRY STRATEGY")
print("========================================")

if best["StrategyType"] == "1-STOP":

    print(
        "\nStrategy:",
        best["FirstTyre"],
        "→",
        best["SecondTyre"]
    )

    print(
        "Pit lap:",
        int(best["PitLap1"])
    )

else:

    print(
        "\nStrategy:",
        best["FirstTyre"],
        "→",
        best["SecondTyre"],
        "→",
        best["ThirdTyre"]
    )

    print(
        "First pit:",
        int(best["PitLap1"])
    )

    print(
        "Second pit:",
        int(best["PitLap2"])
    )

print(
    "Estimated race time:",
    round(
        best["EstimatedRaceTime"],
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

if best["StrategyType"] == "1-STOP":

    p1 = int(
        best["PitLap1"]
    )

    print(
        f"{best['FirstTyre']} stint:",
        p1,
        "laps"
    )

    print(
        f"{best['SecondTyre']} stint:",
        race_laps - p1,
        "laps"
    )

else:

    p1 = int(
        best["PitLap1"]
    )

    p2 = int(
        best["PitLap2"]
    )

    print(
        f"{best['FirstTyre']} stint:",
        p1,
        "laps"
    )

    print(
        f"{best['SecondTyre']} stint:",
        p2 - p1,
        "laps"
    )

    print(
        f"{best['ThirdTyre']} stint:",
        race_laps - p2,
        "laps"
    )


# ============================================================
# ML PIT-STOP PREDICTION
# ============================================================

print("\n========================================")
print("        ML TYRE PREDICTION")
print("========================================")

prediction_lap = int(
    best["PitLap1"]
)

prediction_tyre = best["FirstTyre"]

ml_tyre, ml_probs = predict_next_tyre(
    prediction_tyre,
    prediction_lap,
    prediction_lap
)

print(
    "\nCurrent tyre:",
    prediction_tyre
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

    print("\nTyre probabilities:")

    for tyre, probability in sorted(
        ml_probs.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        print(
            f"{tyre:<15}"
            f"{probability:.2f}%"
        )

    simulator_next = best["SecondTyre"]

    print(
        "\nSimulator next tyre:",
        simulator_next
    )

    if ml_tyre == simulator_next:
        print("\n✓ ML and simulator agree.")
    else:
        print("\n⚠ ML and simulator disagree.")


# ============================================================
# CONFIDENCE
# ============================================================

print("\n========================================")
print("          MODEL CONFIDENCE")
print("========================================")

confidence = 50.0

if (
    best_1 is not None
    and best_2 is not None
):

    separation = abs(
        best_1["StrategyScore"]
        -
        best_2["StrategyScore"]
    )

    if separation >= 30:
        confidence += 20

    elif separation >= 15:
        confidence += 10

    elif separation < 5:
        confidence -= 10


first_support = compound_stats.get(
    best["FirstTyre"],
    {"count": 0}
)["count"]

if first_support >= 50:
    confidence += 15

elif first_support >= 20:
    confidence += 8

else:
    confidence -= 5


if ml_tyre is not None:

    if ml_tyre == best["SecondTyre"]:
        confidence += 10
    else:
        confidence -= 5


if weather["track_condition"] in [
    "DAMP",
    "WET"
]:
    confidence -= 15


confidence = max(
    20.0,
    min(90.0, confidence)
)

if confidence >= 75:
    confidence_level = "HIGH"

elif confidence >= 55:
    confidence_level = "MEDIUM"

else:
    confidence_level = "LOW"

print(
    f"\nConfidence score: "
    f"{confidence:.1f} %"
)

print(
    "Confidence level:",
    confidence_level
)


# ============================================================
# PIT WINDOWS
# ============================================================

print("\n========================================")
print("             PIT WINDOW")
print("========================================")

p1 = int(
    best["PitLap1"]
)

print(
    "Recommended first pit window:",
    f"{max(MIN_STINT, p1 - 2)} - "
    f"{min(race_laps - MIN_STINT, p1 + 2)}"
)

if best["StrategyType"] == "2-STOP":

    p2 = int(
        best["PitLap2"]
    )

    print(
        "Recommended second pit window:",
        f"{max(p1 + MIN_STINT, p2 - 2)} - "
        f"{min(race_laps - MIN_STINT, p2 + 2)}"
    )


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print("\n========================================")
print("       FINAL RECOMMENDATION")
print("========================================")

if weather_strategy == "WET":

    print("\nFinal strategy:")
    print("FULL WET / INTERMEDIATE")

    print("\nReason:")
    print(weather_reason)

elif weather_strategy == "DAMP":

    print("\nFinal strategy:")
    print("INTERMEDIATE / DRY TRANSITION")

    print("\nReason:")
    print(weather_reason)

else:

    print("\nFinal strategy:")

    if best["StrategyType"] == "1-STOP":

        print(
            best["FirstTyre"],
            "→",
            best["SecondTyre"]
        )

    else:

        print(
            best["FirstTyre"],
            "→",
            best["SecondTyre"],
            "→",
            best["ThirdTyre"]
        )

    print("\nReason:")
    print(weather_reason)


# ============================================================
# TOP 10 STRATEGIES
# ============================================================

print("\n========================================")
print("          TOP 10 STRATEGIES")
print("========================================")

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

top10 = results.head(10)[
    display_columns
].copy()

print(
    top10.to_string(
        index=False,
        formatters={
            "EstimatedRaceTime": "{:.2f}".format,
            "StrategyPenalty": "{:.2f}".format,
            "StrategyScore": "{:.2f}".format
        }
    )
)

results.to_csv(
    RESULT_PATH,
    index=False
)

print("\nResults saved to:")
print(RESULT_PATH)

print("\n========================================")
print("     F1 PRE-RACE SYSTEM READY!")
print("========================================")

