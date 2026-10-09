import pandas as pd

# Load cleaned lap data
df = pd.read_csv("data/processed/italy_2024_laps_clean.csv")

# Remove rows without tyre information
df = df.dropna(subset=["Compound", "TyreLife", "LapTime"])

# Calculate average lap time for each tyre compound and tyre age
degradation = (
    df.groupby(["Compound", "TyreLife"])["LapTime"]
    .mean()
    .reset_index()
)

# Sort the results
degradation = degradation.sort_values(
    ["Compound", "TyreLife"]
)

print("\n--- Tyre Degradation Analysis ---")
print(degradation.to_string(index=False))

# Save the result
degradation.to_csv(
    "data/processed/tyre_degradation.csv",
    index=False
)

print("\nTyre degradation analysis completed!")
print("Saved to: data/processed/tyre_degradation.csv")