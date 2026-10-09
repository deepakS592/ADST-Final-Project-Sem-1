from strategy_engine import generate_compound_sequences, simulate_race_time


def test_valid_race_simulation_returns_numeric_output():
    result = simulate_race_time(
        ["SOFT", "MEDIUM"],
        [20, 24],
        44,
        lambda compound, tyre_age, race_lap: 95.0 + 0.04 * (tyre_age - 1),
    )
    assert isinstance(result["estimated_race_time"], float)
    assert result["estimated_race_time"] > 0
    assert result["pit_laps"] == [20]
    assert result["pit_stop_cost"] == 22.0


def test_generator_returns_only_physically_feasible_dry_sequences():
    sequences = generate_compound_sequences("DRY")
    assert sequences
    assert all(sequence.count("HARD") <= 2 for sequence in sequences)
    assert all(sequence.count("MEDIUM") <= 3 for sequence in sequences)
    assert all(len(set(sequence)) >= 2 for sequence in sequences)
    assert ("SOFT", "SOFT", "MEDIUM") in sequences


def test_wet_generator_does_not_force_slick_diversity():
    sequences = generate_compound_sequences("WET")
    assert sequences
    assert all(set(sequence) <= {"INTERMEDIATE", "WET"} for sequence in sequences)


def test_non_finite_lap_prediction_is_rejected():
    try:
        simulate_race_time(
            ["SOFT", "MEDIUM"], [20, 24], 44,
            lambda compound, tyre_age, race_lap: float("nan"),
        )
    except ValueError as error:
        assert "non-positive or non-finite" in str(error)
    else:
        raise AssertionError("A non-finite lap-time estimate must be rejected.")
