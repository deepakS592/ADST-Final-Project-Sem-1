from weather_api import fallback_weather, weather_scenario_probabilities
from services.prediction_service import _strategy_support, _timeline
import weather_api


def test_weather_fallback_is_explicit_and_does_not_crash():
    result = fallback_weather("Example Grand Prix", {"date": "2024-01-01"}, "offline")
    assert result["source"] == "fallback assumption"
    assert result["weather_certainty"] == "LOW"
    assert "offline" in result["reason"]


def test_scenario_weights_and_visualization_are_well_formed():
    package = {"forecast": {"available": False}, "current": {"available": False}}
    weights = weather_scenario_probabilities(package)["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert _timeline({"strategy": "MEDIUM → HARD", "pit_laps": [24], "stint_lengths": [24, 29]}, 53)[1]["start_lap"] == 25
    support = _strategy_support([{"strategy": "MEDIUM → HARD", "weight": 0.7}, {"strategy": "MEDIUM → HARD", "weight": 0.3}])
    assert support == [{"strategy": "MEDIUM → HARD", "scenario_weight": 1.0}]


def test_failed_weather_lookup_is_cached_but_manual_refresh_retries(monkeypatch):
    calls = []
    monkeypatch.setattr(weather_api, "_WEATHER_FAILURE_CACHE", {})

    def fail(*args, **kwargs):
        calls.append(args[0])
        raise OSError("offline")

    info = {"location": "Track A", "country": "Example", "circuit": "Track A", "date": "2026-10-04"}
    first = weather_api.get_current_and_forecast_weather("Track A GP", info, request_get=fail)
    cached = weather_api.get_current_and_forecast_weather("Track A GP", info, request_get=fail)
    refreshed = weather_api.get_current_and_forecast_weather("Track A GP", info, request_get=fail, force_refresh=True)
    assert len(calls) == 2
    assert first["cache_status"] == "failure"
    assert cached["cache_status"] == "recent failure cached"
    assert refreshed["cache_status"] == "failure"
