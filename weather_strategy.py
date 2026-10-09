# ============================================================
# F1 WEATHER STRATEGY MODULE
# ============================================================

print("\n========================================")
print("        F1 WEATHER STRATEGY MODULE")
print("========================================")


# ============================================================
# RACE INFORMATION
# ============================================================

race_information = {

    "Bahrain Grand Prix": {
        "laps": 57,
        "location": "Sakhir, Bahrain"
    },

    "Emilia-Romagna Grand Prix": {
        "laps": 63,
        "location": "Imola, Italy"
    },

    "British Grand Prix": {
        "laps": 52,
        "location": "Silverstone, United Kingdom"
    },

    "Belgian Grand Prix": {
        "laps": 44,
        "location": "Spa-Francorchamps, Belgium"
    },

    "Italian Grand Prix": {
        "laps": 53,
        "location": "Monza, Italy"
    }
}


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
            "Please enter a valid race number."
        )

    except ValueError:

        print(
            "Please enter a number."
        )


selected_race = races[
    race_choice - 1
]


race_info = race_information[
    selected_race
]


print(
    "\nSelected race:",
    selected_race
)

print(
    "Circuit:",
    race_info["location"]
)

print(
    "Race distance:",
    race_info["laps"],
    "laps"
)


# ============================================================
# WEATHER INPUT
# ============================================================

print("\n========================================")
print("          WEATHER FORECAST")
print("========================================")


print(
    "\nEnter the expected weather conditions "
    "for the race."
)


# ============================================================
# TEMPERATURE
# ============================================================

while True:

    try:

        temperature = float(
            input(
                "\nTrack temperature (°C): "
            )
        )

        break

    except ValueError:

        print(
            "Please enter a valid temperature."
        )


# ============================================================
# RAIN PROBABILITY
# ============================================================

while True:

    try:

        rain_probability = float(
            input(
                "Rain probability (%): "
            )
        )

        if 0 <= rain_probability <= 100:

            break

        print(
            "Enter a value between 0 and 100."
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )


# ============================================================
# RAIN DURATION
# ============================================================

while True:

    try:

        rain_duration = float(
            input(
                "Expected rain duration (minutes): "
            )
        )

        if rain_duration >= 0:

            break

        print(
            "Duration cannot be negative."
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )


# ============================================================
# TRACK CONDITION
# ============================================================

print("\nTrack condition:")

print(
    "1 - Dry"
)

print(
    "2 - Damp"
)

print(
    "3 - Wet"
)


while True:

    try:

        track_choice = int(
            input(
                "Select track condition: "
            )
        )

        if track_choice in [1, 2, 3]:

            break

        print(
            "Please select 1, 2 or 3."
        )

    except ValueError:

        print(
            "Please enter a number."
        )


track_conditions = {

    1: "DRY",

    2: "DAMP",

    3: "WET"
}


track_condition = track_conditions[
    track_choice
]


# ============================================================
# WEATHER CLASSIFICATION
# ============================================================
#
# IMPORTANT:
# Track condition has priority over rain probability.
#
# Example:
#
# WET + 30% rain
#
# must still be treated as WET.
# ============================================================

if track_condition == "WET":

    weather_impact = "HIGH"


elif track_condition == "DAMP":

    weather_impact = "HIGH"


elif rain_probability >= 60:

    weather_impact = "HIGH"


elif rain_probability >= 30:

    weather_impact = "MEDIUM"


else:

    weather_impact = "LOW"


# ============================================================
# WEATHER STRATEGY
# ============================================================

recommended_strategy = ""

strategy_reason = ""

primary_tyre = ""

backup_tyre = ""


# ============================================================
# WET TRACK
# ============================================================

if track_condition == "WET":

    if (
        rain_probability >= 70
        or
        rain_duration >= 45
    ):

        recommended_strategy = (
            "FULL WET / INTERMEDIATE"
        )

        primary_tyre = "FULL WET"

        backup_tyre = "INTERMEDIATE"

        strategy_reason = (
            "The track is already wet and significant "
            "rain is expected. Full Wet tyres may be "
            "required if standing water increases."
        )

    else:

        recommended_strategy = (
            "INTERMEDIATE"
        )

        primary_tyre = "INTERMEDIATE"

        backup_tyre = "FULL WET"

        strategy_reason = (
            "The track is already wet. Intermediate "
            "tyres are the preferred option, while "
            "Full Wet tyres should be prepared if "
            "rain becomes heavier."
        )


# ============================================================
# DAMP TRACK
# ============================================================

elif track_condition == "DAMP":

    if rain_probability >= 60:

        recommended_strategy = (
            "INTERMEDIATE"
        )

        primary_tyre = "INTERMEDIATE"

        backup_tyre = "FULL WET"

        strategy_reason = (
            "The track is damp and significant rain "
            "is expected. Intermediate tyres are "
            "recommended."
        )

    else:

        recommended_strategy = (
            "INTERMEDIATE / DRY TRANSITION"
        )

        primary_tyre = "INTERMEDIATE"

        backup_tyre = "MEDIUM"

        strategy_reason = (
            "The track is damp. Intermediate tyres "
            "are safer initially, but a switch to "
            "dry tyres may become possible as the "
            "track dries."
        )


# ============================================================
# DRY TRACK + HIGH RAIN PROBABILITY
# ============================================================

elif rain_probability >= 60:

    recommended_strategy = (
        "DRY → INTERMEDIATE"
    )

    primary_tyre = "MEDIUM"

    backup_tyre = "INTERMEDIATE"

    strategy_reason = (
        "The track is currently dry, but the high "
        "rain probability means the team should "
        "prepare for a transition to Intermediate tyres."
    )


# ============================================================
# DRY TRACK + MEDIUM RAIN PROBABILITY
# ============================================================

elif rain_probability >= 30:

    recommended_strategy = (
        "FLEXIBLE DRY / INTERMEDIATE STRATEGY"
    )

    primary_tyre = "MEDIUM"

    backup_tyre = "INTERMEDIATE"

    strategy_reason = (
        "The track is currently dry but rain is "
        "possible. The dry strategy can continue "
        "initially while preparing for Intermediate tyres."
    )


# ============================================================
# NORMAL DRY CONDITIONS
# ============================================================

else:

    recommended_strategy = (
        "NORMAL DRY TYRE STRATEGY"
    )

    primary_tyre = "MEDIUM"

    backup_tyre = "HARD"

    strategy_reason = (
        "The track is dry and the probability of "
        "rain is low. A normal dry-race strategy "
        "can be used."
    )


# ============================================================
# TEMPERATURE ANALYSIS
# ============================================================

if temperature >= 35:

    temperature_effect = (
        "HIGH TRACK TEMPERATURE"
    )

    temperature_reason = (
        "Tyre degradation may increase. "
        "The team should monitor tyre life carefully."
    )


elif temperature >= 25:

    temperature_effect = (
        "MODERATE TRACK TEMPERATURE"
    )

    temperature_reason = (
        "Normal tyre temperature and degradation "
        "conditions are expected."
    )


else:

    temperature_effect = (
        "LOW TRACK TEMPERATURE"
    )

    temperature_reason = (
        "Tyres may take longer to reach their "
        "optimal operating temperature."
    )


# ============================================================
# WEATHER ANALYSIS OUTPUT
# ============================================================

print("\n========================================")
print("          WEATHER ANALYSIS")
print("========================================")


print(
    "\nRace:",
    selected_race
)

print(
    "Circuit:",
    race_info["location"]
)

print(
    "Temperature:",
    temperature,
    "°C"
)

print(
    "Rain probability:",
    rain_probability,
    "%"
)

print(
    "Expected rain duration:",
    rain_duration,
    "minutes"
)

print(
    "Track condition:",
    track_condition
)


print(
    "\nWeather impact:",
    weather_impact
)


print(
    "\nRecommended weather strategy:"
)

print(
    recommended_strategy
)


print(
    "\nPrimary tyre:",
    primary_tyre
)


print(
    "Backup tyre:",
    backup_tyre
)


print(
    "\nReason:"
)

print(
    strategy_reason
)


print(
    "\nTemperature analysis:"
)

print(
    temperature_effect
)

print(
    temperature_reason
)


# ============================================================
# PIT STOP IMPACT
# ============================================================

print("\n========================================")
print("          WEATHER PIT IMPACT")
print("========================================")


if weather_impact == "LOW":

    print(
        "\nNo major weather-related pit stop change."
    )

    print(
        "Continue with the planned dry strategy."
    )


elif weather_impact == "MEDIUM":

    print(
        "\nWeather may create a strategic pit window."
    )

    print(
        "Monitor rainfall and track conditions."
    )

    print(
        "Prepare Intermediate tyres."
    )

    print(
        "An earlier pit stop may be useful if "
        "rain begins."
    )


else:

    print(
        "\nHIGH WEATHER IMPACT"
    )

    print(
        "The normal dry strategy may no longer "
        "be optimal."
    )

    print(
        "Prepare the recommended wet-weather tyre."
    )

    print(
        "Monitor rainfall intensity and standing water."
    )


# ============================================================
# FINAL WEATHER DECISION
# ============================================================

print("\n========================================")
print("        FINAL WEATHER DECISION")
print("========================================")


if track_condition == "WET":

    if primary_tyre == "FULL WET":

        final_decision = (
            "USE FULL WET / WET-WEATHER STRATEGY"
        )

    else:

        final_decision = (
            "USE INTERMEDIATE STRATEGY"
        )


elif track_condition == "DAMP":

    final_decision = (
        "USE INTERMEDIATE AND MONITOR DRYING TRACK"
    )


elif rain_probability >= 60:

    final_decision = (
        "START DRY BUT PREPARE FOR INTERMEDIATE"
    )


elif rain_probability >= 30:

    final_decision = (
        "KEEP DRY STRATEGY BUT PREPARE FOR RAIN"
    )


else:

    final_decision = (
        "USE NORMAL DRY-RACE STRATEGY"
    )


print(
    "\nDecision:",
    final_decision
)


# ============================================================
# STRATEGY CHANGE FLAG
# ============================================================

if track_condition in [
    "DAMP",
    "WET"
]:

    strategy_change_required = True

elif rain_probability >= 60:

    strategy_change_required = True

else:

    strategy_change_required = False


print(
    "\nStrategy change required:",
    "YES"
    if strategy_change_required
    else
    "NO"
)


# ============================================================
# MODULE OUTPUT SUMMARY
# ============================================================

print("\n========================================")
print("       WEATHER MODULE SUMMARY")
print("========================================")


print(
    "\nRace:",
    selected_race
)

print(
    "Weather impact:",
    weather_impact
)

print(
    "Track:",
    track_condition
)

print(
    "Primary tyre:",
    primary_tyre
)

print(
    "Backup tyre:",
    backup_tyre
)

print(
    "Strategy change:",
    "YES"
    if strategy_change_required
    else
    "NO"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("       WEATHER MODULE COMPLETE")
print("========================================")