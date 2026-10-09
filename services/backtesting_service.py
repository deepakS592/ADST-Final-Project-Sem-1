"""One-event, strict pre-race backtesting from data held in SQLite."""

from __future__ import annotations

from services.data_service import classify_event_status, drivers_for_event
from services.database import event_laps, get_event
from services.prediction_service import predict_strategy


def backtest_event(event_id: str, driver: str) -> dict:
    event = get_event(event_id)
    if not event:
        raise ValueError("The selected race is not available for backtesting.")
    if classify_event_status(event) != "COMPLETED":
        raise ValueError("Backtesting is available only for completed races with observed laps.")
    if str(driver).upper() not in {row["driver_code"] for row in drivers_for_event(event_id)}:
        raise ValueError("Choose a driver who participated in this race.")
    prediction = predict_strategy(event_id, driver, fetch_weather=False)
    actual = event_laps(event_id)
    actual = actual[actual["Driver"].astype(str).str.upper().eq(str(driver).upper())].sort_values("LapNumber")
    grouped = actual.dropna(subset=["Stint"]).groupby("Stint", sort=True)
    sequence = []
    pit_laps = []
    for _, rows in grouped:
        compounds = rows["Compound"].dropna().astype(str)
        if not compounds.empty:
            sequence.append(compounds.mode().iloc[0])
            pit_laps.append(int(rows["LapNumber"].min()) - 1)
    if pit_laps:
        pit_laps = pit_laps[1:]
    chosen = prediction["result"].get("expected_strategy", {})
    predicted_sequence = str(chosen.get("strategy", "")).split(" → ")
    compare_length = max(len(sequence), len(predicted_sequence), 1)
    return {
        "event": event,
        "driver": str(driver).upper(),
        "predicted_strategy": chosen.get("strategy"),
        "actual_strategy": " → ".join(sequence) if sequence else "Unavailable",
        "predicted_pit_laps": chosen.get("pit_laps", []),
        "actual_pit_laps": pit_laps,
        "strategy_sequence_similarity": sum(a == b for a, b in zip(predicted_sequence, sequence)) / compare_length,
        "selected_event_laps_used_for_training": prediction["result"].get("selected_event_laps_used_for_training", True),
        "training_cutoff_time_utc": prediction["result"].get("training_cutoff_time_utc"),
        "note": "Historical strategy comparison uses observed stint compounds. Evaluation-event lap data is read only after the prediction is completed.",
    }
