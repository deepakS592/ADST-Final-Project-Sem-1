# Data sources and source labels

## Formula 1 schedule and timing

- **FastF1** supplies the event schedule, race-session timing, tyre metadata, and driver/team details used by the local data update. The project retains the event-level download ledger and raw/clean lap tables. See the [FastF1 documentation](https://docs.fastf1.dev/).
- The dashboard's normal completed-event selector requires usable cleaned laps. Upcoming schedule rows are a separate pre-race selector and never imply that results or driver entries are already known.
- Event dates in the lap snapshot have day-level precision. When the event start time is available from the schedule, it is displayed separately; training still stops at midnight UTC on the event date because the lap rows do not contain session timestamps.

## Event schedule corrections and tyre nominations

- `data/processed/event_metadata_overrides.csv` stores source-linked corrections for schedule changes not reflected in the saved FastF1 ledger. The included 2026 Bahrain-in-Malaysia entry uses Formula 1's timetable for Sepang, Kuala Lumpur, the October 4 start, and the 56-lap distance: [official F1 timetable](https://www.formula1.com/en/latest/article/formula-1-gulf-air-bahrain-grand-prix-in-malaysia-2026.1IHoy7D2dfCIugb5mH78cE).
- `data/processed/event_tyre_nominations.csv` contains source-linked event nominations. Pirelli's release explicitly maps C3/C4/C5 as Hard/Medium/Soft for 2026 Baku and Singapore. For Sepang, the release names the C2–C4 range but its text does not state each role; the project records the official range while leaving its Hard/Medium/Soft mapping unavailable until the official table is verified. Event roles must not be generalized to other races.
- `data/processed/tyre_allocations.csv` is a separate register for source-backed set counts and Sprint status. It is empty until verified event counts are available. Missing set counts use the labeled model baseline, not an official claim.
- Every event nomination or allocation should include `Source`, `SourceURL`, `VerifiedAt`, and `Status`. The loader ignores unverified values for official display.

## Weather

- Historical weather uses the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). It is a gridded reanalysis product near the event location, not circuit sensor data.
- Current conditions and the event-date outlook use the [Open-Meteo Forecast API](https://open-meteo.com/en/docs). A response is shown only when the selected race date falls inside its requested forecast horizon.
- The [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) and [Previous Model Runs API](https://open-meteo.com/en/docs/previous-runs-api) can support archived-forecast research. The current implementation does not fetch archived lead-time runs, so it does not claim calibrated forecast stability or weather-forecast backtest accuracy.

## Model outputs

Lap predictions, compound profiles, degradation rates, pit-transition probabilities, scenario weights, strategy times, and confidence are model estimates. They are always labeled separately from official schedule, nomination, and observed lap/weather records. Missing source values remain unavailable rather than being filled with fabricated measurements.
