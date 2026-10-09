# Weather methodology

## Separate observation types

The application presents four distinct weather records:

- **Historical weather:** hourly location-level reanalysis from the Open-Meteo Archive API. It is not track telemetry or an official race-control record.
- **Current conditions:** the latest current block returned for the circuit location. It describes weather now, not the upcoming race.
- **Race-date forecast:** the event-date hourly outlook returned by Open-Meteo's Forecast API, when the selected date is inside the requested 1–16 day horizon.
- **Fallback:** when coordinates, network access, or forecast coverage are unavailable, weather measurements remain null. A dry baseline may be simulated but is labeled as an assumption.

The race schedule may expose only an event date. Forecast summaries therefore use local 11:00–22:00 as a daytime proxy, and the dashboard shows this limitation. If exact scheduled start time is available, it is displayed independently from this weather window.

## Forecast cache

Forecast responses are cached by coordinates, event date, and requested horizon for 30 minutes. Historical archive responses keep their existing persistent event-date cache. Re-running analysis before cache expiration does not make another weather request.

## Scenario weights

For each available forecast hour in the daytime window, hourly precipitation probability `p_i` is read as a probability between 0 and 1. A rough chance of any rain is computed as:

```text
P(any rain) = 1 - product(1 - p_i)
```

This assumes hourly events are independent, which is not generally true. The probability is split between DAMP and WET using the share of rain-weighted hours whose forecast rain amount is at least 1 mm. The remainder is DRY. The labels and weights are a scenario-generation heuristic, not a calibrated race-level probability distribution. Missing rain probabilities fall back to the forecast's dominant estimated condition; a missing forecast produces a dry assumption with no weather confidence.

Because the forecast is a current model run, not a verified ensemble distribution, the dashboard does not claim calibrated uncertainty or multi-lead forecast stability. Open-Meteo's Historical Forecast and Previous Model Runs APIs are documented as future sources for a forecast-stability study, but are not used to fabricate retrospective forecasts in this release.

## Weather-aware strategy output

Each included weather scenario is simulated independently with weather-specific feasible compound rules. DRY candidate generation excludes Intermediate and Wet. Damp/wet plans include a wet compound under the current feasibility rules; Wet plans use Wet compounds only. The dashboard reports the best plan per condition and derives a condition-aware contingency policy. The expected/best/worst/standard-deviation values summarize that adaptive policy's scenario-best times. A seeded 500-run Monte Carlo then samples weather regimes and adds independent normal lap-model residuals plus measured cross-stint degradation spread when enough source stints exist. The JSON includes assumptions and P5/P50/P95; these are not calibrated prediction intervals and do not compare every candidate in every weather state.

The current lap-time adjustment uses coarse, documented condition/temperature/rain heuristics. Wet compound predictions remain labeled as fallback if the historical sample is insufficient. These outputs should be interpreted as comparative strategy estimates, not circuit-surface simulation.

## Sources

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo Weather Forecast API](https://open-meteo.com/en/docs)
- [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
- [Open-Meteo Previous Model Runs API](https://open-meteo.com/en/docs/previous-runs-api)
