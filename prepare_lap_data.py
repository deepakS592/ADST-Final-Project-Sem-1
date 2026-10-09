import pandas as pd
import os

# Load the raw lap data
df = pd.read_csv("data/raw/italy_2024_laps.csv")

# Convert lap time to seconds
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

# Remove rows where lap time is missing
df = df.dropna(subset=["LapTime"])

# Sort the data
df = df.sort_values(["Driver", "LapNumber"])

# Create processed folder
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
df.to_csv("data/processed/italy_2024_laps_clean.csv", index=False)

print("Lap data cleaned successfully!")
print("Rows after cleaning:", len(df))
print("Columns:", list(df.columns))
print("Saved to: data/processed/italy_2024_laps_clean.csv")