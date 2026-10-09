# F1 Pit Stop Strategy System

A Flask application for exploring Formula 1 tyre and pit-stop strategies. It combines the project's real historical lap data, circuit-level tyre profiles, pit-transition history, available weather, and its leak-aware Random Forest models. SQLite is the application database. The browser interface is served by Flask templates and plain JavaScript; Streamlit is not used.

The application is strategy support, not a race-control simulator. Its recommendation is an estimate, and confidence is a data-support score rather than a win probability.

## What the project includes

- Real FastF1 race-lap observations from the supplied 2019–2026 snapshot, plus targeted historical lap times and pit stops for circuits that need older race history. Older-source rows are kept in circuit-analysis tables, separate from model training and driver history. Completed-race driver lists come from observed event laps. Upcoming schedule entries contain metadata only and never imply event entrants. The captured 2026 season is partial.
- Historical event, driver, team, circuit, lap, tyre, pit-transition, weather, data-quality and prediction-weather-snapshot tables in SQLite.
- The preserved Random Forest lap-time regressor, next-tyre classifier, tyre degradation analysis and physically constrained strategy simulator.
- A pre-race prediction flow that excludes the selected event and all rows on or after its conservative UTC event-date cutoff from model training.
- Historical track context, pit-lap patterns, compound performance, current weather and event-date forecast when Open-Meteo can provide them.
- A race-weekend dashboard that automatically selects the current or next scheduled event and lets users choose any current/upcoming event in any imported season. Track analysis loads before driver selection; the separate historical workflow remains for backtesting.
- Local circuit SVGs for supported circuits, a driver card with source-backed racing number/team details, six-race circuit coverage, compound pace/degradation, pit-window distributions and historical strategy sequences.
- Offline-built, cutoff-safe event bundles containing fitted models and circuit/tyre/strategy summaries; live predictions use only the selected driver's prior laps plus that bundle. Repeat predictions and compatible aggregates are cached.
- Data import/update, model training, weather refresh and backtest commands.

## Architecture

```text
FastF1 / supplied historical CSVs
              ↓
     cleaning and features
              ↓
        SQLite database
              ↓
 track / tyre / pit / weather history
              ↓
 cutoff-safe ML feature creation
              ↓
 Random Forest + strategy simulator
              ↓
 Flask JSON API → HTML / CSS / JavaScript
```

The included `data/processed/all_seasons_laps_clean.csv` is the preferred first-import source; `data/raw/f1_laps_2019_2026.csv` is the fallback. Targeted pre-2019 circuit history is stored in `data/raw/historical_circuit_laps.csv` and `data/raw/historical_circuit_pit_stops.csv`. The original preparation, training, download and strategy-engine modules are retained. The saved model files are kept in `models/`.

## Machine-learning method

The existing project contains two scikit-learn Random Forest pipelines:

1. **Lap-time regression.** Predicts lap time from season, event/circuit, driver/team, tyre compound, tyre life, lap number and race progress. The strategy simulator uses the result for each candidate stint lap and blends in measured tyre profiles when available.
2. **Next-tyre classification.** Predicts the next observed compound at a pit transition from the event, driver/team, previous tyre and age, current lap and race progress. It supports candidate scoring; inventory and weather feasibility checks remain hard constraints.

Categorical features are one-hot encoded with unknown-category support. Validation holds out the latest season when multiple seasons are available; otherwise it uses grouped race/stint holdouts. The included full-snapshot temporal holdout is 2025: lap-time MAE **7.10 s**, RMSE **10.50 s**, R² **0.388**; next-tyre accuracy **0.570**, versus a 0.427 most-common-tyre baseline. These metrics describe that historical holdout only. They do not guarantee future prediction accuracy. The complete metrics are in `data/processed/model_metrics.json`.

**Live pre-race prediction uses pre-trained cutoff-safe artifacts and does not train models during a user prediction request.** The endpoint loads a bundle keyed by event, conservative UTC cutoff, and SQLite import version. It then queries only the selected driver's prior laps and reads precomputed circuit/tyre/strategy summaries. A missing or stale bundle returns a clear service error with the offline build command; it never falls back to training. The selected event and all same-date laps are excluded from training. A date-only timing source means the whole event date is excluded. The automatically selected current/next race is based on the supplied calendar and configured planning timezone (`F1_APP_TIMEZONE`, default `Asia/Kolkata`); current race weekends run from Friday through Sunday.

The race distance comes from event metadata in the supplied calendar or observed race data. Most F1 race distances are set before race day; an event shortened by an incident can make the stored distance an imperfect proxy. Track-level analysis does not require a driver. Driver-specific prediction uses only a source-backed participant or an explicitly labelled historical candidate with prior history; future grid entries are never inferred.

## SQLite database

The database is `database/f1_strategy.db`. `database/schema.sql` defines its tables and indexes. `services/data_service.py` imports the supplied real data and builds dynamic event and participant rows. Prediction queries read laps from SQLite, and unknown user input is parameterized or validated before database access.

Important tables include `events`, model-input `laps`, circuit-only `circuit_history_laps` and `circuit_history_pit_stops`, `event_drivers`, `pit_transitions`, `tyre_performance`, `tyre_degradation`, `weather_observations`, `weather_snapshots`, `data_quality_issues` and `app_metadata`.

The import records observed data-quality counts; the application does not invent missing lap times, drivers, weather, races or compounds. Use the Data quality control in the page or `GET /api/quality` to inspect imported coverage.

## Weather

The supplied weather-analysis rows are imported into `weather_observations`. In this snapshot, historical hourly weather is sparse (24 rows for one loaded event); `/api/quality` reports events with no historical weather. For a prediction, the existing weather service requests current conditions and an hourly forecast from Open-Meteo, using event location geocoding when coordinates are not already available. Official, source-ledger upcoming races are available for forecast-backed strategy analysis; their driver choice is a prior-history context and is explicitly marked unconfirmed. Requests are cached and failures return an explicit unavailable/fallback result. Historical archive rows can be refreshed for an event using the weather command below.

An event-date forecast is available only inside the provider's current forecast horizon (up to 16 days in the existing service). For older completed races, current forecasts cannot be reconstructed; the strategy run records that limitation and falls back to documented assumptions. Network requests use a short timeout, and failed lookups are cached for 90 seconds to keep repeated predictions responsive; the refresh control bypasses that cache. Weather-scenario weights are heuristic, not calibrated probabilities. Historical weather is location-level reanalysis, not track-surface telemetry.

## Strategy and visualization

The simulator ranks feasible two- and three-stint strategies using estimated lap times, circuit compound offsets/degradation blended with recent global F1 history, pit loss, weather adjustment and observed pit-transition support. Circuit-local analysis selects the six most recent completed circuit races strictly before the selected event cutoff, regardless of calendar gaps. If fewer than six exist, every available completed circuit race is used and the page reports the exact count, years, laps, stints, pit transitions, tyre observations and date range. Local values are shrunk toward the recent global profile according to sample size. The selected event and future rows cannot enter its own circuit sample. It applies the source project's compound inventory, stint-age and dry/wet constraints. Inventory quantities and pit loss are documented model assumptions, not official remaining tyre sets for an individual driver. One- and two-stop plans are exhaustively evaluated; for long dry races, three-stop plans use a five-lap coarse search and refine nearby pit laps for plans close to the best score. This bounds the search while keeping the most competitive three-stop plans in the comparison.

The frontend uses a graphite/red pit-wall visual system and local circuit SVGs where available. It visualizes the selected race outline, event status/round/distance, circuit-history coverage, compound samples, tyre degradation, pit-window and strategy distributions, weather, driver number/team, the recommended compound sequence and model metrics. Analysis remains visible before a driver is selected. Scenario share is labeled as heuristic and is not presented as strategy-success probability. All local track statistics stop at the selected event cutoff. A circuit without supplied pre-cutoff observations remains explicitly empty; the prediction model shrinks toward recent all-circuit data instead of inventing local rows. Circuit geometry/length/turns are not shown when no local asset/metadata is available.

The strategy search builds its per-stint lap-time totals with NumPy arrays rather than calling Python once per lap and candidate. Historical sequence and pit-window support are small, smoothed priors; they can influence ranking without overriding estimated lap time or physical feasibility. The model runs the dominant weather scenario and adds at most one meaningful contingency scenario. Compatible models, tyre profiles, track summaries and repeat event/driver predictions are cached. The prediction API exposes database, model, weather, strategy-generation, simulation and response timings plus cache-hit flags.

### Local prediction timing check

The original 2026 Singapore check with the supplied 179,886 model-input laps took **2.63 seconds** on its first `/api/predict` call after loading event context and **0.41 seconds** on repeat. After adding the six older Sepang races, a fresh local process returned the Sepang event context, then predicted VER in **1.60 seconds** and **0.29 seconds** on repeat (prediction-cache hit; weather disabled). That first prediction measured 315 ms database work, 0.8 ms model lookup and 913 ms strategy simulation. A standalone cold `/api/predict` call without loading the event context first took 3.55 seconds. These are local checks, not hardware-independent response-time guarantees. API timings include `database_ms`, `model_ms`, `weather_ms`, `strategy_generation_ms`, `simulation_ms` and `response_ms`, alongside the original detailed timing names.

## Backtesting

Use the **Compare with actual race strategy** control on any event with suitable driver history. The predictor runs first using the strict pre-race cutoff. Only then does the backtest read the selected event to derive its observed stint-compound sequence and pit laps. The report includes the prediction, actual strategy, pit laps, sequence similarity and cutoff. Historical actual stint data remains separate from model features.

## Installation and startup

Python 3.11 or newer is required for the pinned scikit-learn version used by the included fitted artifacts. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/import_data.py
python app.py
```

Open `http://127.0.0.1:5000`. The app initializes the schema and performs the local import automatically on first start when the database is empty. `run_dashboard.bat` is a Windows launcher for the same Flask application.

On app startup, a background sync refreshes the current local calendar year and the next published calendar, then checks recently completed schedule entries for missing **Race** session laps. It imports only those race sessions, updates SQLite and the precomputed cutoff fingerprints, and rebuilds bundles only for future event cutoffs that can include the new race. It does not request practice or qualifying sessions, create future driver entries, or block Flask from starting if an upstream source is unavailable. Set `F1_STARTUP_SYNC=0` to disable this background update.

The season selector includes every calendar season present in SQLite. The dashboard keeps the in-progress season primary while a later official calendar is selectable as a **PRE-SEASON** preview. Once the final scheduled race date passes in the configured `F1_APP_TIMEZONE`, focus advances to the next published season. Calendar rows without laps are labeled as calendar data; they do not count as race observations.

## Data import and update

Rebuild SQLite from the included snapshot without network access:

```powershell
python scripts/import_data.py
```

Fetch available race laps from FastF1, run the existing cleaning/feature pipeline, and import its refreshed outputs:

```powershell
python scripts/update_data.py --seasons 2024-2026
```

Add `--force` to refresh source event sessions even when cached race files exist. FastF1 and Open-Meteo availability determines how much new data can be retrieved. The Flask **Refresh local data** action rebuilds SQLite from the currently included local CSV files; it does not download every race session.

Fill gaps in scheduled circuit history with only the completed race-lap and pit-stop records needed to approach six local races:

```powershell
python scripts/update_historical_circuit_data.py
```

This targeted offline import uses the [Jolpica F1 race calendar](https://github.com/jolpica/jolpica-f1/tree/main/docs/endpoints) to identify missing events, then requests only race laps and pit stops for those events. Lap-time history in this source begins in 1996; pit-stop records begin in 2011. The importer never requests practice or qualifying sessions. Legacy rows are stored separately from model-training laps. The bundled supplemental history has **7 races, 7,490 lap timings and 343 pit stops**, including Sepang 2012–2017 and Singapore 2018. Their compound is `UNKNOWN` and tyre age is blank when the source does not provide tyre telemetry; the app reports zero local tyre observations for Sepang and uses its existing fallback profile.

Refresh published calendar metadata without adding race observations or future driver entries:

```powershell
python scripts/update_calendar.py --seasons 2027
```

The season argument is dynamic and also accepts ranges or comma-separated years. The importer checks FastF1 and Jolpica schedules and saves published events as metadata-only rows, so `/api/seasons` and the season selector can use a newly published season. A past calendar row without downloaded race laps remains calendar-only; it does not become an observation or driver entry. If upstream has not published a calendar, no rows are fabricated; rerun the command after publication.

Refresh weather for one event (get its `event_id` from `/api/events`):

```powershell
python scripts/update_weather.py <event_id>
```

The API endpoints include `/api/next-event` (also available as `/api/focus-event`), `/api/event-context/<event_id>` (race context, track asset, six-race circuit analysis, cached weather and matching model-bundle status), `/api/seasons`, `/api/events?season=2027&status=available` (current/upcoming calendar entries for the requested imported season), `/api/drivers?event_id=...`, `/api/predict`, `/api/backtest`, `/api/weather/<event_id>`, `/api/quality` and `/api/model-metrics`. The `available` event filter includes both the current event weekend and all upcoming races; it is not limited to the next race.

`event_id` has one canonical form: `{season}-{round:02d}-{sha1(season|round|race)[:10]}`, generated by `services.data_service._event_id` and stored as the SQLite `events.event_id` primary key. Event APIs, browser selections, prediction/backtest requests and model-bundle keys use that persisted ID unchanged. `scripts/train_model.py --asof-event` resolves the exact SQLite key; it does not create aliases or events. Copy the ID returned by the API or read from SQLite rather than reconstructing it.

## Model training and notebook

Train new full-snapshot artifacts from SQLite:

```powershell
python scripts/train_model.py
```

Build the cutoff-safe model and feature bundle for a current, upcoming, or historical event (run this after importing refreshed data, or whenever the live API reports a missing/stale bundle):

```powershell
python scripts/train_model.py --asof-event <event_id>
```

Prepare all scheduled current and upcoming races in one offline pass:

```powershell
python scripts/train_model.py --asof-upcoming
```

Find event IDs with `GET /api/next-event`, `GET /api/events?season=2026`, or in `database/f1_strategy.db`. The command selects the same conservative UTC cutoff used by live prediction. You may supply `--cutoff 2026-10-03T00:00:00Z` for a deliberate as-of rebuild. This is an offline operation; it may fit models and summarize historical laps.

Explore coverage, features and held-out metrics in `notebooks/f1_pit_stop_model.ipynb` using Jupyter Notebook. Run Jupyter from the project root so the notebook can import the project modules.

## Backtest command

```powershell
python scripts/run_backtest.py <event_id> VER
```

Replace the example driver code with a participant returned by the driver selector.

## Tests

```powershell
python -m pytest -q
```

The test suite covers SQLite setup/import, dynamic event and driver selection, Flask routes and invalid input, weather fallback, visualization data, tyre feasibility, model artifact loading, and temporal cutoff behavior.

## Limitations

- Only calendar events and driver participation present in the supplied/imported data are shown; unavailable upstream sessions and future driver entries are omitted. Season logic is dynamic, but a season with no calendar rows cannot be selected until schedule metadata is imported.
- The supplied FastF1 lap snapshot begins in 2019. Targeted older circuit rows are separately sourced from Jolpica and may include lap times and pit stops without tyre-compound telemetry. No compound, tyre-age, driver-entry or race observation is inferred when the source does not provide it.
- The supplied snapshot has no 2027 calendar rows. `scripts/update_calendar.py` imports future schedule metadata once FastF1 or Jolpica publishes it; a calendar update does not create drivers or lap observations.
- Local circuit SVG coverage is bundled for Sepang and every circuit in the included 2026 current/upcoming calendar, plus several historical circuits. Other circuits show an explicit outline-unavailable placeholder while retaining their historical analysis where data exists. Track length/turn counts are omitted unless the source provides them.
- An event driver without pre-cutoff history cannot receive an ML-based recommendation. A new or refreshed SQLite dataset requires its event's cutoff-safe bundle to be built offline before predictions are enabled.
- Race distances, inventory, tyre-age bounds and pit loss may use documented assumptions or observed metadata.
- Open-Meteo forecast range, geocoding, weather coverage and network availability can limit weather inputs. The app clearly labels missing data.
- Lap-time errors and next-tyre accuracy vary by season, event and driver. Confidence is not a calibrated probability, and recommendations are not guaranteed.
- The model does not have team telemetry, traffic, fuel-load estimates, safety-car forecasts or complete per-driver weekend tyre inventories.
