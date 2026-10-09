import pandas as pd
import os


# ==========================================
# F1 PIT STOP TRAINING DATA CREATOR
# ==========================================

print("\n========================================")
print("     F1 PIT STOP TRAINING DATA CREATOR")
print("========================================")


# ==========================================
# LOAD LAP DATA
# ==========================================

file_path = "data/raw/2024_multi_race_laps.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded!")
print("Total records:", len(df))


# ==========================================
# CLEAN DATA
# ==========================================

df["LapNumber"] = pd.to_numeric(
    df["LapNumber"],
    errors="coerce"
)

df["TyreLife"] = pd.to_numeric(
    df["TyreLife"],
    errors="coerce"
)

df["LapTime"] = pd.to_numeric(
    df["LapTime"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Race",
        "Driver",
        "LapNumber",
        "TyreLife",
        "Compound",
        "LapTime"
    ]
)


# Remove abnormal lap times
df = df[
    (df["LapTime"] > 50) &
    (df["LapTime"] < 150)
].copy()


# Sort the data
df = df.sort_values(
    [
        "Race",
        "Driver",
        "LapNumber"
    ]
).reset_index(
    drop=True
)


print(
    "Records after cleaning:",
    len(df)
)


# ==========================================
# IDENTIFY PIT STOPS
# ==========================================
#
# A new tyre stint normally starts when
# TyreLife goes back to 1 or a lower value.
#
# Example:
#
# Lap 1   TyreLife 1
# Lap 2   TyreLife 2
# ...
# Lap 20  TyreLife 20
# Lap 21  TyreLife 1   <- PIT STOP
#
# ==========================================

df["PreviousTyreLife"] = (
    df.groupby(
        ["Race", "Driver"]
    )["TyreLife"]
    .shift(1)
)


df["PreviousCompound"] = (
    df.groupby(
        ["Race", "Driver"]
    )["Compound"]
    .shift(1)
)


df["PreviousLap"] = (
    df.groupby(
        ["Race", "Driver"]
    )["LapNumber"]
    .shift(1)
)


# ==========================================
# FIND PIT STOPS
# ==========================================

pit_stops = df[
    (
        df["TyreLife"] <= 2
    )
    &
    (
        df["PreviousTyreLife"] > 5
    )
].copy()


print(
    "\nPotential pit stops found:",
    len(pit_stops)
)


# ==========================================
# CREATE TRAINING RECORDS
# ==========================================

training_records = []


for index, row in pit_stops.iterrows():

    race = row["Race"]

    driver = row["Driver"]

    pit_lap = int(
        row["LapNumber"]
    )

    new_tyre = row["Compound"]

    tyre_age_before = int(
        row["PreviousTyreLife"]
    )

    previous_tyre = row[
        "PreviousCompound"
    ]


    # --------------------------------------
    # Find driver data before pit
    # --------------------------------------

    driver_race_data = df[
        (df["Race"] == race)
        &
        (df["Driver"] == driver)
    ]


    before_pit = driver_race_data[
        driver_race_data["LapNumber"]
        < pit_lap
    ]


    after_pit = driver_race_data[
        driver_race_data["LapNumber"]
        >= pit_lap
    ]


    # --------------------------------------
    # Average lap time before pit
    # --------------------------------------

    if len(before_pit) > 0:

        avg_before = (
            before_pit[
                "LapTime"
            ].tail(5).mean()
        )

    else:

        avg_before = None


    # --------------------------------------
    # Average lap time after pit
    # --------------------------------------

    if len(after_pit) > 0:

        avg_after = (
            after_pit[
                "LapTime"
            ].head(5).mean()
        )

    else:

        avg_after = None


    # --------------------------------------
    # Create training record
    # --------------------------------------

    training_records.append({

        "Race":
            race,

        "Driver":
            driver,

        "PitLap":
            pit_lap,

        "PreviousTyre":
            previous_tyre,

        "PreviousTyreAge":
            tyre_age_before,

        "NewTyre":
            new_tyre,

        "AverageLapTimeBefore":
            avg_before,

        "AverageLapTimeAfter":
            avg_after

    })


# ==========================================
# CREATE DATAFRAME
# ==========================================

strategy_df = pd.DataFrame(
    training_records
)


# ==========================================
# REMOVE EMPTY RECORDS
# ==========================================

strategy_df = strategy_df.dropna(
    subset=[
        "PreviousTyre",
        "NewTyre"
    ]
)


# ==========================================
# SORT DATA
# ==========================================

strategy_df = strategy_df.sort_values(
    [
        "Race",
        "Driver",
        "PitLap"
    ]
).reset_index(
    drop=True
)


# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

output_directory = (
    "data/processed"
)


os.makedirs(
    output_directory,
    exist_ok=True
)


# ==========================================
# SAVE DATA
# ==========================================

output_file = (
    "data/processed/"
    "pit_strategy_training_data.csv"
)


strategy_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print(
    "\n========================================"
)

print(
    "       PIT STOP DATA SUMMARY"
)

print(
    "========================================"
)


print(
    "\nTraining records:",
    len(strategy_df)
)


if len(strategy_df) > 0:

    print(
        "\nTyre combinations:"
    )

    print(
        strategy_df[
            [
                "PreviousTyre",
                "NewTyre"
            ]
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nExample records:"
    )

    print(
        strategy_df.head(10)
        .to_string(
            index=False
        )
    )


print(
    "\nSaved to:"
)

print(
    output_file
)


print(
    "\n========================================"
)

print(
    "     PIT STOP DATASET READY!"
)

print(
    "========================================"
)