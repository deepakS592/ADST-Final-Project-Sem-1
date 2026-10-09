import pandas as pd

from data_loader import filter_laps_before_cutoff
from strategy_engine import strategy_is_feasible


def test_training_cutoff_excludes_race_day_rows():
    laps = pd.DataFrame({"Season": [2024, 2024, 2024], "Race": ["Earlier", "Target", "Later"], "EventDate": ["2024-03-01", "2024-03-02", "2024-03-03"]})
    filtered = filter_laps_before_cutoff(laps, "2024-03-02T00:00:00Z")
    assert filtered["Race"].tolist() == ["Earlier"]


def test_dry_strategies_obey_compound_rules():
    assert strategy_is_feasible(["MEDIUM", "HARD"], "DRY")
    assert not strategy_is_feasible(["MEDIUM", "MEDIUM"], "DRY")
