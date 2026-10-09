# Pre-race backtesting

Run a compact historical backtest with:

```powershell
python backtesting.py --seasons 2024,2025 --limit 2
```

The default limit is two events. The runner chooses completed races deterministically, then reconstructs what was available immediately before each race date. The local timing table contains an event date but not timestamps for individual laps, so the conservative data cutoff is midnight UTC at the start of race day. The full target season/event is removed from the training set even if metadata is inconsistent.

Only a driver with prior lap history is eligible. Race distance is estimated from earlier same-event history; a target event's observed distance is not fed into the recommendation. If there is no prior same-event distance, the event is skipped with a reason. Weather lookup is disabled for the default backtest because today's weather is not historical pre-race information. Archived forecast runs are not reconstructed by this runner, so forecast accuracy is reported unavailable.

After generating the recommendation, the evaluation step reads the completed event to calculate:

- recommended vs observed compound sequence similarity;
- recommended and observed stop counts;
- model-estimated race time;
- sum of available recorded lap times; and
- actual driver lap coverage; and
- an approximate time difference when the evaluation driver has recorded at least 90% of the estimated race distance.

The time difference is a rough indicator, not a calibrated realized race-time metric. It is omitted for retirements or incomplete source coverage that leaves less than 90% of estimated race distance. The model adds a nominal pit loss to its simulated lap prediction; recorded lap-time sums are also affected by in-laps and flags. The report keeps the comparison reason beside each row.

Outputs are written to `data/processed/backtesting/backtest_report.csv` and `backtest_summary.json`, with each recommendation's detailed scenario output in a per-event directory. Reports are reproducible from the included data and model seed.
