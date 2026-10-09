"""Manual FastF1 data refresh helper (not a pytest test).

It is intentionally guarded so ``pytest`` never makes a live network request.
The supplied historical CSV is enough for the normal project workflow.
"""


def main() -> None:
    try:
        import fastf1
    except ImportError as error:
        raise SystemExit("Install the optional 'fastf1' package to refresh this source data.") from error

    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(2024, "Italy", "R")
    session.load()
    pit_stops = session.laps[session.laps["PitInTime"].notna()].copy()
    pit_stops = pit_stops[["Driver", "LapNumber", "Compound", "TyreLife", "PitInTime"]]
    pit_stops.to_csv("data/italy_2024_pit_stops.csv", index=False)
    print(f"Pit-stop dataset created; rows: {len(pit_stops)}")
    print("Saved to: data/italy_2024_pit_stops.csv")


if __name__ == "__main__":
    main()
