# System architecture

## Data flow

```text
FastF1 event schedules (2019–configured end year)
        │
        ├─ skip future races and sessions without lap observations
        ├─ cache downloaded timing resources in cache/fastf1/
        └─ save resumable race CSVs in data/raw/fastf1/<season>/
                       │
                       ▼
       data_loader.py ──► combined source laps + download ledger
                       │     event_catalog.csv + data_coverage.json
                       ▼
       data_preparation.py
          ├─ normalized all_seasons_laps_clean.csv
          ├─ observed tyre transitions
          ├─ compound performance and degradation profiles
          └─ source metadata, season, event and driver identity
                       │
                       ▼
       train_models.py
          ├─ Random Forest lap-time regressor
          ├─ Random Forest next-tyre classifier
          └─ held-out metrics + versioned-by-data artifacts
                       │
                       ▼
       strategy_engine.py ◄──── weather_api.py
          ├─ season/event/driver-specific race context
          ├─ physical set and stint feasibility before scoring
          ├─ predicted lap surface and race simulation
          ├─ recommendation, confidence and factual explanation
          └─ CSV + JSON backend outputs
                       │
                       ▼
       pre_race_analysis.py
          ├─ strict event-date cutoff + selected-event exclusion
          ├─ cutoff-specific persisted Random Forest models
          ├─ current conditions + event-date forecast
          ├─ DRY/DAMP/WET policy + seeded Monte Carlo
          └─ adaptive contingency and robustness outputs
                       │
                       ▼
       dashboard.py → dynamic_dashboard_app.py
          └─ historical / pre-race mode + dynamic event/driver choices
       data_quality.py + backtesting.py
          ├─ source/cleaning coverage reports
          └─ pre-race recommendations compared with observed events
```

## Data management

The supported season interval defaults to 2019–2026 and can be changed with `F1_SEASON_START` and `F1_SEASON_END`. `data_loader.py` gets event schedules from FastF1 and only attempts races whose source schedule date has passed. It then loads race sessions with lap data enabled and telemetry, weather, and messages disabled. Empty, future, or unavailable sessions are recorded as such; they do not become empty synthetic race results.

FastF1 caches source responses under `cache/fastf1/`. Each usable event has a CSV under `data/raw/fastf1/<season>/round_<round>_race_laps.csv`. This makes downloads resumable. The loader also writes a combined source file, event catalog, download status ledger, and coverage JSON. The supplied 2024 CSV is used only as a real-data fallback for that season.

The Flask app serves its HTML, CSS, and JavaScript frontend and reads the local SQLite database for event options, participants, historical track context, and lap features. The local refresh button reimports the included CSV snapshot; network-backed FastF1 updates are run explicitly with `scripts/update_data.py`. Pre-race model artifacts are cached by cutoff and eligible-data fingerprint so repeated predictions do not retrain unchanged historical rows.

## Dynamic metadata and filtering

`Season`, `Round`, source event name, official event name, circuit/location, country, race date, observed race distance, driver code, driver full name, driver number, and team are stored with laps whenever FastF1 returns them. Dashboard choices are derived from actual event/driver lap rows. A driver cannot appear for an event without rows for that driver in that season. Same-named events are disambiguated by season before analysis.

2026 is labeled current/partial when it is the current year. The configured endpoint never causes future sessions to be displayed as completed. The available data source controls which races, drivers, teams, and sessions are selectable.

Upcoming schedule entries appear only in the separate pre-race selector. `build_event_calendar` enriches them with historical metadata only as context and labels prior same-event race distance as an estimate. `event_metadata_overrides.csv` stores source-linked schedule changes such as venue relocation. The scheduled start is shown where known; model data stops at midnight UTC on race date because historical lap rows have date-only precision. A second guard removes the selected season/event even if date metadata is malformed.

## Modeling and strategy path

Model features are built from the selected historical data. The training module evaluates by holding out the latest completed season when possible. The final persisted models are then retrained using the full available dataset. Model metrics are measured on the holdout and saved separately from the fitted full-data estimator.

Strategy generation enumerates one- and two-stop plans and four-stint three-stop plans for dry races of at least 65 laps. It checks compound inventory, weather-compatible compounds, dry compound diversity, minimum stint length, tyre-age limits, and exact total race distance. Only validated candidates enter race-time simulation, scoring, and ranking. Random Forest probabilities support the result but never bypass those checks.

The separate pre-race path filters source laps to the conservative cutoff, then fits or loads cutoff-specific models keyed by a hash of the eligible table and cutoff. All model fitting, tyre profiles, driver history, and transitions use the same as-of subset. Official nominations and allocations are read from source-attributed registers; unavailable values remain explicit.

## Weather integration

`weather_api.py` uses event date and location. If coordinates are not in local metadata, it calls Open-Meteo geocoding and caches the coordinates. Historical hourly weather responses are cached by location/date. Temperature, precipitation/rain, weather code, wind, and precipitation probability when present are summarized for the local race-day window. If an event cannot be geocoded or the request fails, measured fields remain null and the source is labeled `fallback assumption`.

Pre-race current conditions and hourly event-date forecasts come from the Forecast API, using a selectable 1–16 day horizon and 30-minute disk cache. Daytime forecast values are summarized when exact session timing is unavailable. Heuristic rain weights generate dry/damp/wet branches. Seeded Monte Carlo samples scenario weights, lap-model residuals, and measured cross-stint degradation spread where supported; its assumptions are recorded in output JSON.

## Outputs and runtime paths

The dashboard and CLI share `strategy_engine.run_analysis`. A run writes:

- `data/processed/strategy_simulation_results.csv`
- `data/processed/weather_analysis.csv`
- `data/processed/backend_result.json`

Preprocessing writes the clean lap table, pit-transition examples, tyre profile/degradation tables, and model metrics. All paths are rooted at the project folder through `config.py`.

`data_quality.py` writes JSON/CSV quality reports. `backtesting.py` writes approximate pre-race-vs-observed reports with distance, weather, and elapsed-time limitations labeled.

## Reliability boundaries

FastF1 is a community-maintained data interface; upstream session availability and API shape may change. The per-event ledger records unavailable sessions and why they were skipped or failed. Existing local data remains usable if a download fails. No code infers future lap times, drivers, completed races, or measured weather.
