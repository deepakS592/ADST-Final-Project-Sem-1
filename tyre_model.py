import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# ==========================================
# F1 TYRE DEGRADATION MODEL
# ==========================================

print("\n========================================")
print("       F1 TYRE DEGRADATION MODEL")
print("========================================")


# Load multi-race dataset
file_path = "data/raw/2024_multi_race_laps.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Total records:", len(df))


# ==========================================
# CLEAN DATA
# ==========================================

df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
df["LapTime"] = pd.to_numeric(df["LapTime"], errors="coerce")

df = df.dropna(
    subset=[
        "TyreLife",
        "LapNumber",
        "LapTime",
        "Driver",
        "Compound"
    ]
)

# Remove unrealistic values
df = df[df["TyreLife"] > 0]
df = df[df["LapTime"] > 50]
df = df[df["LapTime"] < 200]

print("Records after cleaning:", len(df))


# ==========================================
# FEATURES
# ==========================================

features = [
    "TyreLife",
    "LapNumber",
    "Driver",
    "Compound"
]

target = "LapTime"


X = df[features]
y = df[target]


# ==========================================
# PREPROCESSING
# ==========================================

categorical_features = [
    "Driver",
    "Compound"
]

numeric_features = [
    "TyreLife",
    "LapNumber"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ==========================================
# MODEL
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)


print("\nTraining model...")

model.fit(X, y)


# ==========================================
# MODEL SCORE
# ==========================================

r2 = model.score(X, y)

print("\n========================================")
print("             MODEL RESULTS")
print("========================================")

print("R² score:", round(r2, 3))


# ==========================================
# SIMPLE TYRE DEGRADATION ANALYSIS
# ==========================================

print("\n========================================")
print("       TYRE DEGRADATION ANALYSIS")
print("========================================")


for compound in df["Compound"].unique():

    tyre_data = df[
        df["Compound"] == compound
    ].copy()

    if len(tyre_data) < 30:
        print("\n", compound)
        print("Not enough data")
        continue

    X_compound = tyre_data[
        ["TyreLife", "LapNumber", "Driver", "Compound"]
    ]

    y_compound = tyre_data["LapTime"]

    compound_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regression", LinearRegression())
        ]
    )

    compound_model.fit(
        X_compound,
        y_compound
    )

    score = compound_model.score(
        X_compound,
        y_compound
    )

    # Get the regression coefficients
    regression = compound_model.named_steps[
        "regression"
    ]

    coefficients = regression.coef_

    # Numeric variables are at the end
    tyre_coefficient = coefficients[-2]
    lap_coefficient = coefficients[-1]

    print("\n----------------------------------------")
    print("Compound:", compound)
    print("Data points:", len(tyre_data))

    print(
        "Estimated tyre effect:",
        round(tyre_coefficient, 4),
        "seconds/lap"
    )

    print(
        "Estimated lap-number effect:",
        round(lap_coefficient, 4),
        "seconds/lap"
    )

    print(
        "R² score:",
        round(score, 3)
    )


# ==========================================
# SAVE MODEL SUMMARY
# ==========================================

print("\n========================================")
print("       MODEL COMPLETE")
print("========================================")

print("""
The model now considers:

- Tyre age
- Race lap number
- Driver
- Tyre compound

This gives a more realistic estimate of tyre
behaviour than using TyreLife alone.
""")


print("========================================")
print("       TYRE MODEL READY!")
print("========================================")