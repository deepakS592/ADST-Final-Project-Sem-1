import pandas as pd
import numpy as np


# ==========================================
# F1 PIT STOP STRATEGY MODEL
# ==========================================

print("\n========================================")
print("        F1 PIT STOP STRATEGY MODEL")
print("========================================")


# ------------------------------------------
# Load data
# ------------------------------------------

file_path = "data/raw/2024_multi_race_laps.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded!")
print("Total records:", len(df))


# ------------------------------------------
# Clean data
# ------------------------------------------

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

df["Position"] = pd.to_numeric(
    df["Position"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "Race",
        "Driver",
        "LapNumber",
        "TyreLife",
        "LapTime",
        "Compound"
    ]
)


# Remove unrealistic lap times
df = df[
    (df["LapTime"] > 50) &
    (df["LapTime"] < 150)
]


# Remove first/last tyre laps where
# data can sometimes be unusual
df = df[df["TyreLife"] > 0]


print("Records after cleaning:", len(df))


# ==========================================
# BASIC RACE INFORMATION
# ==========================================

print("\n========================================")
print("             AVAILABLE RACES")
print("========================================")

for race in df["Race"].unique():
    race_data = df[df["Race"] == race]

    print(
        race,
        "→",
        int(race_data["LapNumber"].max()),
        "laps"
    )


# ==========================================
# AVERAGE LAP TIME BY COMPOUND
# ==========================================

print("\n========================================")
print("        AVERAGE LAP TIME")
print("========================================")

compound_times = (
    df.groupby("Compound")["LapTime"]
    .mean()
    .sort_values()
)

print(compound_times)


# ==========================================
# AVERAGE LAP TIME BY TYRE AGE
# ==========================================

print("\n========================================")
print("      LAP TIME BY TYRE AGE")
print("========================================")


tyre_age_analysis = (
    df.groupby(
        ["Compound", "TyreLife"]
    )["LapTime"]
    .mean()
    .reset_index()
)


for compound in df["Compound"].unique():

    print("\n", compound)

    compound_data = tyre_age_analysis[
        tyre_age_analysis["Compound"] == compound
    ]

    print(
        compound_data.head(15).to_string(
            index=False
        )
    )


# ==========================================
# DRIVER PACE
# ==========================================

print("\n========================================")
print("          DRIVER PACE")
print("========================================")


driver_pace = (
    df.groupby("Driver")["LapTime"]
    .median()
    .sort_values()
)


print(driver_pace)


# ==========================================
# STRATEGY DATA
# ==========================================

print("\n========================================")
print("       STRATEGY DATA READY")
print("========================================")


strategy_data = (
    df.groupby(
        [
            "Race",
            "Driver",
            "Compound",
            "TyreLife"
        ]
    )
    .agg(
        AverageLapTime=("LapTime", "mean"),
        MedianLapTime=("LapTime", "median"),
        LapCount=("LapTime", "count")
    )
    .reset_index()
)


print(
    "\nStrategy records:",
    len(strategy_data)
)


# ------------------------------------------
# Save strategy dataset
# ------------------------------------------

output_file = (
    "data/processed/"
    "strategy_training_data.csv"
)

strategy_data.to_csv(
    output_file,
    index=False
)


print("\nSaved to:")
print(output_file)


# ==========================================
# COMPLETE
# ==========================================

print("\n========================================")
print("       STRATEGY MODEL READY!")
print("========================================")