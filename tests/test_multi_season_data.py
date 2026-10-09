from datetime import date
from types import SimpleNamespace

import pandas as pd

import data_loader
from data_loader import (
    available_seasons,
    build_data_catalog,
    drivers_for_event,
    event_info_from_laps,
    events_for_season,
    laps_for_event,
    parse_seasons,
)
from data_preparation import clean_lap_data
from train_models import _temporal_holdout


def _sample_laps():
    return clean_lap_data(pd.DataFrame([
        {"Season": 2024, "Round": 1, "Race": "Same Grand Prix", "Circuit": "Track A", "Country": "A",
         "Location": "City A", "EventDate": "2024-03-01", "RaceLaps": 3, "Driver": "AAA",
         "DriverName": "Alex A", "DriverNumber": "1", "Team": "Team A", "Compound": "SOFT",
         "LapNumber": 1, "TyreLife": 1, "LapTime": 90.0},
        {"Season": 2024, "Round": 1, "Race": "Same Grand Prix", "Circuit": "Track A", "Country": "A",
         "Location": "City A", "EventDate": "2024-03-01", "RaceLaps": 3, "Driver": "AAA",
         "DriverName": "Alex A", "DriverNumber": "1", "Team": "Team A", "Compound": "SOFT",
         "LapNumber": 2, "TyreLife": 2, "LapTime": 91.0},
        {"Season": 2025, "Round": 2, "Race": "Same Grand Prix", "Circuit": "Track B", "Country": "B",
         "Location": "City B", "EventDate": "2025-03-01", "RaceLaps": 3, "Driver": "BBB",
         "DriverName": "Blair B", "DriverNumber": "2", "Team": "Team B", "Compound": "MEDIUM",
         "LapNumber": 1, "TyreLife": 1, "LapTime": 100.0},
        {"Season": 2025, "Round": 2, "Race": "Same Grand Prix", "Circuit": "Track B", "Country": "B",
         "Location": "City B", "EventDate": "2025-03-01", "RaceLaps": 3, "Driver": "BBB",
         "DriverName": "Blair B", "DriverNumber": "2", "Team": "Team B", "Compound": "MEDIUM",
         "LapNumber": 2, "TyreLife": 2, "LapTime": 101.0},
    ]))


def test_season_range_is_configurable_and_never_extends_into_future():
    assert parse_seasons("2019-2026") == tuple(range(2019, 2027))
    assert parse_seasons("2019-2099")[-1] == date.today().year
    assert parse_seasons([2020, 2024, 2028]) == (2020, 2024)


def test_available_tracks_drivers_and_events_come_from_lap_rows():
    laps = _sample_laps()
    assert available_seasons(laps) == [2024, 2025]
    assert events_for_season(laps, 2024)["Circuit"].tolist() == ["Track A"]
    assert events_for_season(laps, 2025)["Circuit"].tolist() == ["Track B"]
    assert drivers_for_event(laps, 2024, "Same Grand Prix") == ["AAA"]
    assert drivers_for_event(laps, 2025, "Same Grand Prix") == ["BBB"]
    assert drivers_for_event(laps, 2024, "Unavailable Grand Prix") == []
    assert len(laps_for_event(laps, 2024, "Same Grand Prix")) == 2
    assert len(laps_for_event(laps, 2025, "Same Grand Prix")) == 2


def test_event_metadata_is_season_specific_even_when_event_name_repeats():
    laps = _sample_laps()
    first = event_info_from_laps(laps, 2024, "Same Grand Prix")
    second = event_info_from_laps(laps, 2025, "Same Grand Prix")
    assert first["circuit"] == "Track A"
    assert second["circuit"] == "Track B"
    assert first["laps"] == second["laps"] == 3


def test_catalog_coverage_counts_only_observed_source_rows():
    laps = _sample_laps()
    events, coverage = build_data_catalog(laps)
    assert len(events) == 2
    assert coverage["lap_rows"] == 4
    assert coverage["race_count"] == 2
    assert coverage["available_seasons"] == [2024, 2025]


def test_legacy_rows_reconcile_to_source_event_names_and_rounds():
    source = pd.DataFrame([{
        "Season": 2024, "Round": 16, "Race": "Emilia Romagna Grand Prix",
        "EventName": "Emilia Romagna Grand Prix", "OfficialEventName": "Formula 1 Emilia Romagna Grand Prix",
        "Circuit": "Imola", "Country": "Italy", "Location": "Imola", "EventDate": "2024-05-19",
        "RaceLaps": 63,
    }])
    legacy = pd.DataFrame([{
        "Season": 2024, "Round": 0, "Race": "Emilia-Romagna Grand Prix",
        "Circuit": "Imola", "EventName": "Emilia-Romagna Grand Prix", "OfficialEventName": "Emilia-Romagna Grand Prix",
        "Country": "Unknown", "Location": "Imola", "EventDate": "", "RaceLaps": pd.NA,
    }])
    reconciled = data_loader._reconcile_legacy_event_metadata(legacy, source)
    assert reconciled.loc[0, "Race"] == "Emilia Romagna Grand Prix"
    assert reconciled.loc[0, "Round"] == 16
    assert reconciled.loc[0, "OfficialEventName"] == "Formula 1 Emilia Romagna Grand Prix"


def test_download_skips_future_events_and_resumes_cached_rounds(tmp_path, monkeypatch):
    schedule = pd.DataFrame([
        {"RoundNumber": 1, "EventName": "Past Grand Prix", "OfficialEventName": "Past GP",
         "Location": "City", "Country": "Country", "Session5": "Race",
         "Session5Date": "2026-09-26T14:00:00+00:00"},
        {"RoundNumber": 2, "EventName": "Future Grand Prix", "OfficialEventName": "Future GP",
         "Location": "City", "Country": "Country", "Session5": "Race",
         "Session5Date": "2026-10-04T14:00:00+00:00"},
    ])
    laps = pd.DataFrame([
        {"Driver": "AAA", "DriverNumber": "1", "LapTime": pd.Timedelta(90 + index, unit="s"),
         "LapNumber": index, "TyreLife": index, "Compound": "SOFT", "Stint": 1}
        for index in range(1, 4)
    ])

    class FakeSession:
        results = pd.DataFrame([{"DriverNumber": "1", "FullName": "Alex A", "TeamName": "Team A"}])
        def __init__(self):
            self.laps = laps
        def load(self, **kwargs):
            assert kwargs["telemetry"] is False
            assert kwargs["weather"] is False

    calls = []
    fake_fastf1 = SimpleNamespace(
        get_event_schedule=lambda year, include_testing=False: schedule,
        get_session=lambda year, round_number, session_type: (calls.append(round_number) or FakeSession()),
    )
    monkeypatch.setattr(data_loader, "_load_fastf1", lambda: fake_fastf1)
    monkeypatch.setattr(data_loader, "SEASON_DATA_DIR", tmp_path / "fastf1")

    first_laps, first_status = data_loader.download_season(2026, today=date(2026, 9, 29))
    assert len(first_laps) == 3
    assert first_status[0]["Status"] == "downloaded"
    assert first_status[1]["Status"] == "future/unrun"
    assert calls == [1]

    second_laps, second_status = data_loader.download_season(2026, today=date(2026, 9, 29))
    assert len(second_laps) == 3
    assert second_status[0]["Status"] == "cached"
    assert calls == [1]


def test_temporal_model_holdout_uses_latest_completed_season():
    seasons = pd.Series([2023] * 15 + [2024] * 15 + [2025] * 15 + [date.today().year] * 15)
    features = pd.DataFrame({"value": range(len(seasons))})
    target = pd.Series([float(value) for value in range(len(seasons))])
    split = _temporal_holdout(features, target, seasons)
    assert split is not None
    train_index, test_index, held_out = split
    assert held_out == 2025
    assert set(seasons.iloc[train_index]) == {2023, 2024}
    assert set(seasons.iloc[test_index]) == {2025}
