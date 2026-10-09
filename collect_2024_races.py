import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache("cache")

races = [
    "Bahrain Grand Prix",
    "Emilia-Romagna Grand Prix",
    "British Grand Prix",
    "Belgian Grand Prix",
    "Italian Grand Prix"
]

all_laps = []

for race in races:

    print(f"\nLoading: {race}")

    session = fastf1.get_session(2024, race, "R")

    session.load(
        telemetry=False,
        weather=True,
        messages=True
    )

    laps = session.laps.copy()

    columns = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Stint",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "PitInTime",
        "PitOutTime",
        "Position",
        "TrackStatus"
    ]

    laps = laps[columns]

    # Add race name
    laps["Race"] = race

    all_laps.append(laps)

    print(f"Loaded {len(laps)} laps")

# Combine all races
df = pd.concat(all_laps, ignore_index=True)

# Convert lap time to seconds
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

# Remove missing lap times
df = df.dropna(subset=["LapTime"])

# Create folder
os.makedirs("data/raw", exist_ok=True)

# Save dataset
df.to_csv(
    "data/raw/2024_multi_race_laps.csv",
    index=False
)

print("\n================================")
print("Multi-race dataset created!")
print("Total records:", len(df))
print("Races:", df["Race"].nunique())
print("Drivers:", df["Driver"].nunique())
print("================================")

print("\nSaved to:")
print("data/raw/2024_multi_race_laps.csv")