import fastf1
import pandas as pd
import os

# Create cache folder
fastf1.Cache.enable_cache("cache")

# Load 2024 Italian Grand Prix
session = fastf1.get_session(2024, "Italian Grand Prix", "R")
session.load(
    telemetry=False,
    weather=True,
    messages=True
)

# Get lap data
laps = session.laps.copy()

# Select useful columns
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

# Create data folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save dataset
laps.to_csv("data/raw/italy_2024_laps.csv", index=False)

print("Race lap dataset created successfully!")
print("Number of records:", len(laps))
print("Saved to: data/raw/italy_2024_laps.csv")