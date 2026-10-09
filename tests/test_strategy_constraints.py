from strategy_engine import strategy_is_feasible, validate_stint_lengths


def test_three_hard_sets_rejected():
    assert not strategy_is_feasible(["HARD", "HARD", "HARD"], "DRY")


def test_three_medium_sets_rejected_for_dry_diversity():
    assert not strategy_is_feasible(["MEDIUM", "MEDIUM", "MEDIUM"], "DRY")


def test_medium_hard_medium_is_feasible():
    assert strategy_is_feasible(["MEDIUM", "HARD", "MEDIUM"], "DRY")


def test_soft_medium_hard_is_feasible():
    assert strategy_is_feasible(["SOFT", "MEDIUM", "HARD"], "DRY")


def test_repeated_soft_is_allowed_within_allocation():
    assert strategy_is_feasible(["SOFT", "SOFT", "MEDIUM"], "DRY")


def test_three_hard_sets_are_rejected_by_physical_counter():
    assert not strategy_is_feasible(["HARD", "HARD", "HARD"], "DRY")


def test_four_medium_sets_are_rejected_by_physical_counter():
    assert not strategy_is_feasible(["MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM"], "DRY")


def test_wet_strategy_does_not_use_dry_diversity_rule():
    assert strategy_is_feasible(["INTERMEDIATE", "INTERMEDIATE"], "WET")


def test_damp_strategy_can_use_intermediate_to_dry_transition():
    assert strategy_is_feasible(["INTERMEDIATE", "MEDIUM"], "DAMP")


def test_invalid_compound_is_rejected():
    assert not strategy_is_feasible(["ULTRA", "MEDIUM"], "DRY")


def test_stint_below_minimum_is_rejected():
    valid, reasons = validate_stint_lengths(["SOFT", "MEDIUM"], [9, 35], 44, "DRY")
    assert not valid
    assert any("below" in reason for reason in reasons)


def test_excessive_tyre_age_is_rejected():
    valid, reasons = validate_stint_lengths(["SOFT", "HARD"], [26, 18], 44, "DRY")
    assert not valid
    assert any("exceeds" in reason for reason in reasons)


def test_invalid_race_distance_is_rejected():
    valid, reasons = validate_stint_lengths(["SOFT", "MEDIUM"], [20, 24], 0, "DRY")
    assert not valid
    assert any("Race distance" in reason for reason in reasons)


def test_stints_must_sum_to_distance():
    valid, reasons = validate_stint_lengths(["SOFT", "MEDIUM"], [20, 23], 44, "DRY")
    assert not valid
    assert any("add up" in reason for reason in reasons)
