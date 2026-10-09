from datetime import date

import pandas as pd
import pytest
import joblib


def test_pre_race_engine_never_trains_when_artifact_is_missing(monkeypatch):
    import strategy_engine

    def unexpected_training(*args, **kwargs):
        raise AssertionError("model training was called from a live pre-race path")

    monkeypatch.setattr(strategy_engine, "train_lap_time_model", unexpected_training)
    monkeypatch.setattr(strategy_engine, "train_pit_stop_model", unexpected_training)
    laps = pd.DataFrame({
        "Season": [2024], "Race": ["Earlier Grand Prix"], "EventDate": ["2024-03-01"], "Driver": ["VER"],
    })
    with pytest.raises(ValueError, match="pre-trained cutoff-safe"):
        strategy_engine.run_analysis(
            "Future Grand Prix", "VER", analysis_mode="pre-race",
            cutoff_time="2024-04-01T00:00:00Z", event_info_override={"laps": 50},
            training_laps=laps,
        )


def test_prediction_uses_artifact_and_repeat_cache_without_loading_all_laps(monkeypatch):
    import pre_race_analysis
    import services.prediction_service as service

    event = {
        "event_id": "test-event", "season": 2026, "round": 18, "race": "Test Grand Prix",
        "official_event_name": "Test Grand Prix", "circuit": "Test Circuit", "country": "Testland",
        "location": "Test City", "event_date": "2026-10-25", "race_laps": 52,
    }
    calls = {"analysis": 0, "bundle": 0}
    context_cutoffs = []
    version_cutoffs = []
    history = pd.DataFrame({
        "Season": [2025], "Race": ["Prior Grand Prix"], "EventDate": ["2025-10-01"],
        "Driver": ["VER"], "Circuit": ["Prior Circuit"], "Team": ["Team"], "DriverName": ["Driver"],
    })
    strategy = {
        "strategy": "SOFT → HARD", "pit_laps": [24], "stint_lengths": [24, 28],
        "pit_window": {"start_lap": 21, "end_lap": 27, "central_lap": 24},
        "estimated_race_time": 5000.0, "strategy_score": 5000.3, "top_alternatives": [],
    }
    fake_result = {
        "expected_strategy": strategy, "primary_strategy": strategy, "strategy_alternatives": [],
        "strategy_score_breakdown": {"total_score_seconds": 5000.3}, "pit_laps": [24],
        "confidence_score": 70, "confidence_level": "MEDIUM", "model_metrics": {}, "scenarios": [],
        "training_rows": 1000, "training_event_count": 10,
    }

    monkeypatch.setattr(service, "get_event", lambda event_id: event)
    monkeypatch.setattr(service, "classify_event_status", lambda *args: "CURRENT")
    monkeypatch.setattr(service, "event_driver", lambda *args: None)
    monkeypatch.setattr(service, "driver_options_for_event", lambda *args: {"drivers": [{"driver_code": "VER"}]})
    monkeypatch.setattr(service, "app_today", lambda: date(2026, 10, 3))
    monkeypatch.setattr(service, "_data_version", lambda cutoff: (version_cutoffs.append(cutoff) or "test-data-v1"))
    monkeypatch.setattr(service, "driver_laps_frame", lambda *args, **kwargs: history.copy())
    monkeypatch.setattr(service, "_all_laps", lambda: (_ for _ in ()).throw(AssertionError("full laps table loaded")))
    monkeypatch.setattr(service, "_event_tyre_register_version", lambda: "tyre-v1")
    monkeypatch.setattr(service, "_cached_history_summary", lambda circuit, before, *args: (context_cutoffs.append(before) or {}))
    monkeypatch.setattr(service, "_cached_tyre_profile", lambda *args: ())
    monkeypatch.setattr(service, "_cached_degradation_curve", lambda *args: ())
    monkeypatch.setattr(service, "historical_weather", lambda *args: {"available": False})
    monkeypatch.setattr(service, "historical_weather_for_circuit", lambda *args, **kwargs: {"available": False})
    monkeypatch.setattr(pre_race_analysis, "load_prediction_bundle", lambda *args: (calls.__setitem__("bundle", calls["bundle"] + 1) or {"historical_context": {}}, False))
    monkeypatch.setattr(pre_race_analysis, "train_lap_time_model", lambda *args: (_ for _ in ()).throw(AssertionError("live lap model training")))
    monkeypatch.setattr(pre_race_analysis, "train_pit_stop_model", lambda *args: (_ for _ in ()).throw(AssertionError("live pit model training")))

    def fake_analysis(*args, **kwargs):
        calls["analysis"] += 1
        return {"result": fake_result}

    monkeypatch.setattr(pre_race_analysis, "run_pre_race_analysis", fake_analysis)
    with service._CACHE_LOCK:
        service._PREDICTION_CACHE.clear()
    try:
        first = service.predict_strategy("test-event", "VER", fetch_weather=False)
        second = service.predict_strategy("test-event", "VER", fetch_weather=False)
        assert first["strategy"] == "SOFT → HARD"
        assert first["pit_window"]["central_lap"] == 24
        assert first["prediction_timings"]["response_ms"] < 5000
        assert first["cache_hits"]["prediction"] is False
        assert second["cache_hits"]["prediction"] is True
        assert calls == {"analysis": 1, "bundle": 1}
        assert {"database_ms", "model_ms", "weather_ms", "strategy_generation_ms", "simulation_ms", "response_ms"}.issubset(second["prediction_timings"])
        assert {"database_load_ms", "model_load_ms", "strategy_simulation_ms"}.issubset(second["prediction_timings"])
        assert context_cutoffs == ["2026-10-03"]
        assert version_cutoffs == ["2026-10-03T00:00:00Z", "2026-10-03T00:00:00Z"]
    finally:
        with service._CACHE_LOCK:
            service._PREDICTION_CACHE.clear()


def test_cutoff_bundle_loader_validates_event_cutoff_and_data_version(monkeypatch, tmp_path):
    import pre_race_analysis

    cutoff = "2026-10-25T00:00:00Z"
    bundle = {
        "bundle_version": pre_race_analysis.ASOF_BUNDLE_VERSION,
        "event_id": "test-event", "cutoff_utc": cutoff, "data_version": "data-v1",
        "lap_artifact": {"model": "loaded"}, "pit_artifact": {"model": "loaded"},
    }
    artifact_path = tmp_path / "offline-bundle.joblib"
    joblib.dump(bundle, artifact_path)
    monkeypatch.setattr(pre_race_analysis, "_prediction_bundle_path", lambda *args: artifact_path)
    pre_race_analysis._load_prediction_bundle.cache_clear()
    loaded, cache_hit = pre_race_analysis.load_prediction_bundle("test-event", cutoff, "data-v1")
    assert loaded["lap_artifact"]["model"] == "loaded"
    assert cache_hit is False
    loaded_again, cache_hit = pre_race_analysis.load_prediction_bundle("test-event", cutoff, "data-v1")
    assert loaded_again["event_id"] == "test-event"
    assert cache_hit is True
    wrong_data, _ = pre_race_analysis.load_prediction_bundle("test-event", cutoff, "new-data-v2")
    assert wrong_data is None
