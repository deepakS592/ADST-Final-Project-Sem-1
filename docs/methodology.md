# Methodology

## Source data and coverage

The downloader queries real historical Formula 1 race-session data through FastF1. The default request range is 2019–2026, configurable via `F1_SEASON_START` and `F1_SEASON_END`. The actual dataset consists only of completed sessions for which the selected source returned usable laps. Future scheduled sessions are skipped; a current-year season is described as partial. A missing event is not imputed. The original supplied 2024 lap CSV remains as a local fallback.

Source lap rows retain season, round, event name, official event name, circuit/location, country, event date, observed race distance, driver code/name/number, team, tyre compound/age, stint, lap number/time, and available timing flags. Preprocessing converts times to seconds, normalizes compound spelling, removes unusable lap rows, rejects unsupported compound labels, removes duplicate season/event/driver/lap records, and checks lap numbers against the event distance. It never manufactures race results.

FastF1 provides race laps and tyre metadata where published. Driver/team/event fields can be missing in some sessions; unknown fields remain unknown. Downloads are cached by event and can resume. Coverage is written to `data/processed/data_coverage.json`; event download status is in `download_status.csv`.

## Tyre profile and degradation

Lap time is normalized against the median for a driver and event, limiting the effect of baseline pace and track differences. Compound offsets use median normalized times. Degradation uses linear slopes of normalized lap time against tyre age within observed stints. Each stint needs at least five rows and four unique tyre ages; extreme normalized laps are trimmed by quantile before slope fitting. Historical degradation is surfaced only with sufficient laps and multiple usable stints.

Fallback rates exist only for the three slick compounds and are labeled as fallback. Unsupported Intermediate/Wet estimates stay unavailable or use the documented wet-weather heuristic; they are not described as trained data when observations are insufficient.

## Lap-time Random Forest

The regressor uses season, event name, circuit/location, driver, team, compound, lap number, tyre life, and race progress. Categorical fields are one-hot encoded with `handle_unknown="ignore"`. Race progress is built from observed event distance. In simulation, lap number and tyre age refer to the candidate lap/stint being evaluated; this is a race-time simulation input, not a claim that a team's pre-race model knows future realized state.

When at least two seasons have enough usable rows, evaluation holds out the latest completed season, excluding the current calendar year from the validation target. The validation estimator trains only on earlier seasons and reports real MAE, RMSE, and R² on the held-out season. The persisted estimator is subsequently refit on all currently available historical rows for inference. If temporal validation cannot be formed, complete race-driver-stint groups are held out so a stint cannot leak laps across train and test. Metrics and evaluation method are saved in `model_metrics.json`.

## Next-tyre Random Forest

Observed stint transitions are derived from source stint-ID changes and/or tyre-age resets on adjacent laps. The classifier predicts the observed next compound from season, event, circuit/location, driver, team, previous compound and tyre age, current lap, and race progress. It returns class probabilities when the model has enough rows and at least two target compounds. It predicts a historical choice; it is not a causal pit-call model.

Validation uses the latest completed season when possible. Otherwise, it uses complete event groups. The final classifier is refit on all available observed transitions. The engine may use its probabilities as a small support term, but physical inventory and stint validation remain authoritative and execute first.

## Feasibility constraints

A baseline dry weekend has eight Soft, three Medium, and two Hard sets. This is a model assumption; actual per-driver remaining sets depend on all practice, qualifying, sprint, and race use. Repeated compounds are allowed if set counts remain within allocation. Dry strategies require at least two different slick compounds. Wet strategies use separately modeled Intermediate/Wet inventory and are not forced to use slick diversity.

Minimum stint length is ten laps. Slick minimum tyre ages are 8/10/12 for Soft/Medium/Hard; maximum model ages are 25/35/45. Wet age limits are separate. These are configurable simulator limits, not official Pirelli life guarantees. The pit loss defaults to 22 seconds per stop. Plans must sum exactly to the selected race distance. Every impossible candidate is rejected before prediction aggregation, score calculation, or ranking.

Event set counts can be loaded from the source-attributed nomination/allocation registers. If those values are missing, the simulator uses the 8/3/2 slick and 4/4 wet model assumptions and labels them. Strategy simulation uses generic Soft/Medium/Hard roles; a source that provides only a selected C-compound range does not qualify as a verified role mapping.

Long dry races of at least 65 laps also enumerate physically feasible 3-stop plans (four stints). The extra search is limited to dry conditions to keep mixed-weather enumeration bounded; damp and wet scenario searches retain 1/2-stop plans. Four-stint plans must satisfy the same inventory, compound diversity, stint-age, and total-distance checks.

## Pre-race cutoff and leakage guard

Upcoming races are listed from the local FastF1 schedule ledger plus source-linked metadata corrections. Each pre-race run shows the scheduled start where known and trains only on lap events dated before race day. The local lap table has event dates but no exact race-session timestamp for each lap, so the model data cutoff is conservatively set to 00:00 UTC on the event date. A second guard removes the selected season/event pair even if source dates are malformed. Model fitting, tyre profiles, driver history, and pit transitions all receive only this as-of subset. Current-event entries and qualifying data are not used.

When the schedule has no official lap distance, a prior same-event median can be shown as an estimate. Source-linked metadata overrides can replace it with the official distance. An event without either value cannot be simulated.

## Race-time estimate and strategy score

The simulator sums per-lap estimates across the selected race distance for each compound/stint sequence and adds pit-stop loss. Lap-time estimates blend a trained Random Forest prediction with historical driver-normalized compound offsets and degradation when both are available. Weather adjustments use measured local weather when returned, with explicit assumptions for dry/damp/wet scenarios. A small deterministic strategy penalty reflects tyre-age/set utilization and classifier support; lower total score ranks first. The JSON output records estimated race time, pit loss, degradation contribution, weather adjustment, penalty, and total score separately.

## Weather

Event coordinates are taken from metadata when available, otherwise geocoded by location/country through Open-Meteo. Historical hourly temperature, rain/precipitation, weather code, and wind are requested and cached. Precipitation probability is displayed only when the response contains it. Condition is classified from returned precipitation during the local race-day window. Wind and temperature can affect the impact label. API/geocoding failures produce a dry fallback assumption with null measurements and a reason, so fallback values cannot be mistaken for observations.

Historical hourly weather near a city/circuit is a proxy; the project does not have continuous track-surface moisture or temperature data for every lap.

Pre-race weather uses Open-Meteo current conditions and an hourly event-date forecast. The requested horizon can be 1–16 days and successful responses are cached on disk for 30 minutes. The forecast is summarized over local 11:00–22:00 as a daytime proxy because the general schedule ledger may not contain the exact race start time. The dashboard keeps current weather separate from event-day forecast.

Weather scenario weights use hourly precipitation probabilities with an independence approximation to estimate any rain in the race window; the wet share is split using forecast hourly rain amounts. This is a heuristic, not a calibrated event-level probability. When the API does not cover the race date, the app runs a dry baseline assumption and does not claim weather robustness. Archived forecast lead-time stability is not yet queried.

The adaptive contingency table scores a separate best feasible plan for each included weather condition. Its expected, best, worst, and standard-deviation figures summarize those scenario-best times. A seeded 500-run Monte Carlo samples weather regimes, independent normal lap-model residuals, and measured cross-stint degradation spread when enough source stints exist. The percentile outputs are not calibrated prediction intervals and do not compare every candidate in every weather state.

## Backtesting

`backtesting.py` chooses completed races, reconstructs a cutoff at midnight UTC on race day, estimates distance from earlier runs at the same event, and uses only drivers with prior lap history. The completed race is consulted after recommendation generation to compare observed compound sequences and recorded lap-time sums. The time difference is approximate: the simulation includes modeled pit loss, while the observed lap sum is not a calibrated reconstruction of full elapsed race time. Archived weather forecast accuracy is reported unavailable because pre-race forecast snapshots are not included in the current run.

## Confidence

The bounded confidence score combines selected-driver lap coverage, held-out lap-model MAE, score separation between tyre sequences, weather certainty, maximum tyre-age and inventory use, and next-tyre model support. Wet-compound heuristics with insufficient training observations reduce confidence. The score is an explanatory index, not a calibrated success probability.

## Limitations

Source lap-time quality, season coverage, driver/team metadata, and compound labels vary. The output is not a live pit wall strategy and omits fuel, traffic, safety cars, full tyre allocations, changing track evolution, and team telemetry. The baseline inventory, pit loss, tyre-age bounds, and weather-to-lap-time adjustments are modeling assumptions. Missing historical data remains visible as missing rather than being silently synthesized.
