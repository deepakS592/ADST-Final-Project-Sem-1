import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# F1 PIT STOP PREDICTION MODEL
# ==========================================

print("\n========================================")
print("       F1 PIT STOP PREDICTION MODEL")
print("========================================")


# ==========================================
# LOAD DATA
# ==========================================

file_path = (
    "data/processed/"
    "pit_strategy_training_data.csv"
)

df = pd.read_csv(
    file_path
)


print("\nDataset loaded!")

print(
    "Training records:",
    len(df)
)


# ==========================================
# CLEAN DATA
# ==========================================

df = df.dropna(
    subset=[
        "Driver",
        "PitLap",
        "PreviousTyre",
        "PreviousTyreAge",
        "NewTyre"
    ]
).copy()


# ==========================================
# FEATURES
# ==========================================

features = [

    "Driver",

    "PitLap",

    "PreviousTyre",

    "PreviousTyreAge"

]


target = "NewTyre"


# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================

driver_encoder = LabelEncoder()

previous_tyre_encoder = LabelEncoder()

new_tyre_encoder = LabelEncoder()


df["DriverEncoded"] = (
    driver_encoder.fit_transform(
        df["Driver"]
    )
)


df["PreviousTyreEncoded"] = (
    previous_tyre_encoder.fit_transform(
        df["PreviousTyre"]
    )
)


df["NewTyreEncoded"] = (
    new_tyre_encoder.fit_transform(
        df["NewTyre"]
    )
)


# ==========================================
# CREATE INPUT AND OUTPUT
# ==========================================

X = df[
    [
        "DriverEncoded",
        "PitLap",
        "PreviousTyreEncoded",
        "PreviousTyreAge"
    ]
]


y = df[
    "NewTyreEncoded"
]


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )
)


print(
    "\nTraining records:",
    len(X_train)
)


print(
    "Testing records:",
    len(X_test)
)


# ==========================================
# CREATE MODEL
# ==========================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=8,

    random_state=42

)


# ==========================================
# TRAIN MODEL
# ==========================================

print(
    "\nTraining model..."
)


model.fit(
    X_train,
    y_train
)


# ==========================================
# TEST MODEL
# ==========================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\n========================================"
)

print(
    "             MODEL RESULTS"
)

print(
    "========================================"
)


print(
    "\nAccuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print(
    "\nClassification Report:"
)


print(
    classification_report(
        y_test,
        y_pred,
        target_names=
        new_tyre_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({

    "Feature": [

        "Driver",

        "Pit Lap",

        "Previous Tyre",

        "Previous Tyre Age"

    ],

    "Importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "Importance",
    ascending=False
)


print(
    "\n========================================"
)

print(
    "        FEATURE IMPORTANCE"
)

print(
    "========================================"
)


print(
    importance.to_string(
        index=False
    )
)


# ==========================================
# TEST A DRIVER
# ==========================================

print(
    "\n========================================"
)

print(
    "        PIT STOP PREDICTION"
)

print(
    "========================================"
)


driver = input(
    "\nEnter driver code (example VER): "
).upper()


previous_tyre = input(
    "Enter current tyre (SOFT/MEDIUM/HARD): "
).upper()


tyre_age = int(
    input(
        "Enter current tyre age: "
    )
)


race_lap = int(
    input(
        "Enter current race lap: "
    )
)


# ==========================================
# CHECK DRIVER
# ==========================================

if driver not in driver_encoder.classes_:

    print(
        "\nDriver not found in training data."
    )

    exit()


# ==========================================
# CHECK TYRE
# ==========================================

if (
    previous_tyre
    not in previous_tyre_encoder.classes_
):

    print(
        "\nTyre not found in training data."
    )

    exit()


# ==========================================
# ENCODE INPUT
# ==========================================

driver_value = (
    driver_encoder.transform(
        [driver]
    )[0]
)


tyre_value = (
    previous_tyre_encoder.transform(
        [previous_tyre]
    )[0]
)


# ==========================================
# CREATE PREDICTION INPUT
# ==========================================

prediction_data = pd.DataFrame({

    "DriverEncoded": [
        driver_value
    ],

    "PitLap": [
        race_lap
    ],

    "PreviousTyreEncoded": [
        tyre_value
    ],

    "PreviousTyreAge": [
        tyre_age
    ]

})


# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(
    prediction_data
)[0]


predicted_tyre = (
    new_tyre_encoder.inverse_transform(
        [prediction]
    )[0]
)


# ==========================================
# PREDICTION PROBABILITY
# ==========================================

probabilities = model.predict_proba(
    prediction_data
)[0]


probability_data = pd.DataFrame({

    "Tyre":
        new_tyre_encoder.classes_,

    "Probability":
        probabilities * 100

})


probability_data = (
    probability_data
    .sort_values(
        "Probability",
        ascending=False
    )
)


# ==========================================
# DISPLAY PREDICTION
# ==========================================

print(
    "\n========================================"
)

print(
    "          PREDICTION RESULT"
)

print(
    "========================================"
)


print(
    "\nDriver:",
    driver
)


print(
    "Current tyre:",
    previous_tyre
)


print(
    "Current tyre age:",
    tyre_age,
    "laps"
)


print(
    "Current race lap:",
    race_lap
)


print(
    "\nRecommended next tyre:",
    predicted_tyre
)


print(
    "\nTyre probabilities:"
)


for _, row in probability_data.iterrows():

    print(
        f"{row['Tyre']:<15}"
        f"{row['Probability']:.2f}%"
    )


print(
    "\n========================================"
)

print(
    "       PIT STOP MODEL READY!"
)

print(
    "========================================"
)