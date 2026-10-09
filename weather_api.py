"""Historical Open-Meteo lookup with explicit estimated fallback behavior."""

from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from datetime import date as date_type, datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import pandas as pd
import numpy as np
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
RACE_START_HOUR = 11
RACE_END_HOUR = 22
WEATHER_FAILURE_TTL_SECONDS = 90
_WEATHER_FAILURE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def interpret_weather_code(code: Any) -> str:
    if code is None or pd.isna(code):
        return "Unknown"
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "Unknown"
    if value == 0:
        return "Clear"
    if value in (1, 2, 3):
        return "Cloudy"
    if value in (45, 48):
        return "Fog"
    if value in (51, 53, 55, 56, 57):
        return "Drizzle"
    if value in (61, 63, 65, 66, 67, 80, 81, 82, 85, 86):
        return "Rain"
    if value in (71, 73, 75, 77):
        return "Snow"
    if value in (95, 96, 99):
        return "Thunderstorm"
    return "Unknown"


def fallback_weather(race_name: str, race_info: dict[str, Any], reason: str) -> dict[str, Any]:
    """Return an assumption, not invented weather observations."""

    return {
        "season": race_info.get("season"),
        "race": race_name,
        "circuit": race_info.get("circuit"),
        "country": race_info.get("country"),
        "date": race_info.get("date"),
        "condition": "DRY",
        "impact": "LOW",
        "temperature": None,
        "precipitation": None,
        "rain": None,
        "precipitation_probability": None,
        "wind_speed": None,
        "wind_direction": None,
        "dominant_weather": "Unavailable",
        "source": "fallback assumption",
        "weather_certainty": "LOW",
        "reason": f"Historical weather could not be verified; dry conditions are used as a documented fallback. {reason}",
        "hourly": [],
    }


def _read_hourly(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
        raise ValueError("Open-Meteo response did not contain hourly observations.")
    hourly = payload["hourly"]
    time_values = hourly.get("time")
    if not isinstance(time_values, list) or not time_values:
        raise ValueError("Open-Meteo response contained no hourly timestamps.")
    if "temperature_2m" not in hourly:
        raise ValueError("Open-Meteo response contained no temperature observations.")
    if "rain" not in hourly and "precipitation" not in hourly:
        raise ValueError("Open-Meteo response contained no rain or precipitation observations.")

    frame = pd.DataFrame({"time": time_values})
    for source, target in (
        ("temperature_2m", "temperature_2m"),
        ("precipitation", "precipitation"),
        ("rain", "rain"),
        ("weather_code", "weather_code"),
        ("weathercode", "weather_code"),
        ("precipitation_probability", "precipitation_probability"),
        ("wind_speed_10m", "wind_speed_10m"),
        ("wind_direction_10m", "wind_direction_10m"),
    ):
        if source in hourly and target not in frame:
            values = hourly[source]
            if not isinstance(values, list) or len(values) != len(time_values):
                raise ValueError(f"Open-Meteo field {source} had an unexpected shape.")
            frame[target] = values
    for required in ("temperature_2m", "precipitation", "rain", "precipitation_probability", "wind_speed_10m", "wind_direction_10m"):
        if required not in frame:
            frame[required] = pd.NA
    if "weather_code" not in frame:
        frame["weather_code"] = pd.NA
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"])
    for column in ("temperature_2m", "precipitation", "rain", "weather_code", "precipitation_probability", "wind_speed_10m", "wind_direction_10m"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty:
        raise ValueError("Open-Meteo response contained no valid hourly records.")
    return frame


def analyze_weather_payload(
    race_name: str,
    race_info: dict[str, Any],
    payload: Any,
) -> dict[str, Any]:
    hourly = _read_hourly(payload)
    race_window = hourly[
        hourly["time"].dt.hour.between(RACE_START_HOUR, RACE_END_HOUR)
    ].copy()
    if race_window.empty:
        race_window = hourly.copy()

    temperature = race_window["temperature_2m"].dropna()
    wind_speed = race_window["wind_speed_10m"].dropna()
    wind_direction = race_window["wind_direction_10m"].dropna()
    rain_probability = race_window["precipitation_probability"].dropna()
    rain_values = race_window["rain"].fillna(0.0)
    precipitation_values = race_window["precipitation"].fillna(0.0)
    total_rain = float(rain_values.sum())
    total_precipitation = float(precipitation_values.sum())
    codes = race_window["weather_code"].dropna().astype(int).tolist()
    names = [interpret_weather_code(code) for code in codes]
    dominant = pd.Series(names).mode().iloc[0] if names else "Unknown"

    if total_rain >= 1.0 or total_precipitation >= 1.0:
        condition = "WET"
    elif total_rain > 0.0 or total_precipitation > 0.0:
        condition = "DAMP"
    else:
        condition = "DRY"
    if condition == "WET":
        impact = "HIGH"
    elif condition == "DAMP" or (not temperature.empty and float(temperature.mean()) >= 32) or (not wind_speed.empty and float(wind_speed.mean()) >= 40):
        impact = "MEDIUM"
    else:
        impact = "LOW"

    hourly_records = []
    for row in hourly.itertuples(index=False):
        hourly_records.append(
            {
                "time": row.time.isoformat(timespec="minutes"),
                "temperature_2m": None if pd.isna(row.temperature_2m) else float(row.temperature_2m),
                "precipitation": None if pd.isna(row.precipitation) else float(row.precipitation),
                "rain": None if pd.isna(row.rain) else float(row.rain),
                "weather_code": None if pd.isna(row.weather_code) else int(row.weather_code),
                "weather_name": interpret_weather_code(row.weather_code),
                "precipitation_probability": None if pd.isna(row.precipitation_probability) else float(row.precipitation_probability),
                "wind_speed_10m": None if pd.isna(row.wind_speed_10m) else float(row.wind_speed_10m),
                "wind_direction_10m": None if pd.isna(row.wind_direction_10m) else float(row.wind_direction_10m),
            }
        )

    return {
        "season": race_info.get("season"),
        "race": race_name,
        "circuit": race_info.get("circuit"),
        "country": race_info.get("country"),
        "date": race_info.get("date"),
        "condition": condition,
        "impact": impact,
        "temperature": float(temperature.mean()) if not temperature.empty else None,
        "precipitation": total_precipitation,
        "rain": total_rain,
        "precipitation_probability": float(rain_probability.mean()) if not rain_probability.empty else None,
        "wind_speed": float(wind_speed.mean()) if not wind_speed.empty else None,
        "wind_direction": float(wind_direction.mean()) if not wind_direction.empty else None,
        "dominant_weather": dominant,
        "source": "Open-Meteo historical archive",
        "weather_certainty": "HIGH",
        "reason": "Condition is classified from observed historical precipitation during the local race-day window; wind contributes to the impact label when available.",
        "hourly": hourly_records,
    }


def get_historical_weather(
    race_name: str,
    race_info: dict[str, Any],
    *,
    request_get: Callable[..., Any] | None = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Fetch the event's historical weather, falling back without crashing."""

    get = request_get or requests.get
    try:
        latitude = race_info.get("latitude")
        longitude = race_info.get("longitude")
        coordinate: dict[str, float] | None = None
        if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
            from config import COORDINATE_CACHE_PATH

            cache_path = Path(COORDINATE_CACHE_PATH)
            try:
                location_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                location_cache = {}
            location = str(race_info.get("location") or race_info.get("circuit") or "").strip()
            country = str(race_info.get("country") or "").strip()
            key = f"{location}|{country}".casefold()
            coordinate = location_cache.get(key)
            if not coordinate:
                if not location or location == "Unknown":
                    raise ValueError("The selected event has no geocodable location metadata.")
                geo_response = get(
                    GEOCODING_URL,
                    params={"name": location, "count": 10, "language": "en", "format": "json"},
                    timeout=timeout,
                )
                geo_response.raise_for_status()
                candidates = geo_response.json().get("results", [])
                country_key = country.casefold()
                match = next((item for item in candidates if str(item.get("country", "")).casefold() == country_key), None)
                match = match or (candidates[0] if candidates else None)
                if not match:
                    raise ValueError(f"Open-Meteo could not geocode {location}, {country}.")
                coordinate = {"latitude": float(match["latitude"]), "longitude": float(match["longitude"])}
                location_cache[key] = coordinate
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(location_cache, indent=2), encoding="utf-8")
            latitude, longitude = coordinate["latitude"], coordinate["longitude"]

        from config import WEATHER_CACHE_DIR

        cache_key = hashlib.sha256(
            f"{float(latitude):.5f}|{float(longitude):.5f}|{race_info['date']}".encode("utf-8")
        ).hexdigest()
        cache_path = Path(WEATHER_CACHE_DIR) / f"{cache_key}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return analyze_weather_payload(race_name, race_info, payload)
            except Exception:
                pass
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": race_info["date"],
            "end_date": race_info["date"],
            "hourly": "temperature_2m,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m",
            "timezone": "auto",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        }
        response = get(ARCHIVE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return analyze_weather_payload(race_name, race_info, payload)
    except Exception as error:
        return fallback_weather(race_name, race_info, str(error))


def _resolve_coordinates(
    race_info: dict[str, Any],
    *,
    request_get: Callable[..., Any],
    timeout: float,
) -> tuple[float, float]:
    """Use event coordinates or the same cached location geocoder as history."""

    latitude, longitude = race_info.get("latitude"), race_info.get("longitude")
    if latitude is not None and longitude is not None and pd.notna(latitude) and pd.notna(longitude):
        return float(latitude), float(longitude)
    from config import COORDINATE_CACHE_PATH

    cache_path = Path(COORDINATE_CACHE_PATH)
    try:
        location_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        location_cache = {}
    location = str(race_info.get("location") or race_info.get("circuit") or "").strip()
    country = str(race_info.get("country") or "").strip()
    key = f"{location}|{country}".casefold()
    coordinate = location_cache.get(key)
    if not coordinate:
        if not location or location == "Unknown":
            raise ValueError("The selected event has no geocodable location metadata.")
        response = request_get(
            GEOCODING_URL,
            params={"name": location, "count": 10, "language": "en", "format": "json"},
            timeout=timeout,
        )
        response.raise_for_status()
        candidates = response.json().get("results", [])
        country_key = country.casefold()
        match = next((item for item in candidates if str(item.get("country", "")).casefold() == country_key), None)
        match = match or (candidates[0] if candidates else None)
        if not match:
            raise ValueError(f"Open-Meteo could not geocode {location}, {country}.")
        coordinate = {"latitude": float(match["latitude"]), "longitude": float(match["longitude"])}
        location_cache[key] = coordinate
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(location_cache, indent=2), encoding="utf-8")
    return float(coordinate["latitude"]), float(coordinate["longitude"])


def get_current_and_forecast_weather(
    race_name: str,
    race_info: dict[str, Any],
    *,
    request_get: Callable[..., Any] | None = None,
    timeout: float = 3.0,
    forecast_days: int = 16,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch current conditions and the forecast for the event date separately.

    Open-Meteo's live endpoint supplies current conditions and hourly forecasts.
    The forecast object is unavailable when the race date falls outside its returned
    hourly horizon. Race-hour summaries use local 10:00–22:00 as a clearly labeled
    daytime proxy when an exact race-session start time is not present.
    """

    get = request_get or requests.get
    failure_key = hashlib.sha256(
        f"{race_info.get('location') or race_info.get('circuit')}|{race_info.get('country')}|{race_info.get('date')}".casefold().encode("utf-8")
    ).hexdigest()
    failed = _WEATHER_FAILURE_CACHE.get(failure_key)
    if not force_refresh and failed and monotonic() - failed[0] < WEATHER_FAILURE_TTL_SECONDS:
        result = deepcopy(failed[1])
        result["cache_status"] = "recent failure cached"
        return result
    try:
        latitude, longitude = _resolve_coordinates(race_info, request_get=get, timeout=timeout)
        forecast_days = max(1, min(16, int(forecast_days)))
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "hourly": "temperature_2m,apparent_temperature,precipitation_probability,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover,relative_humidity_2m",
            "forecast_days": forecast_days,
            "timezone": "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }
        from config import FORECAST_CACHE_DIR

        cache_key = hashlib.sha256(
            f"{latitude:.5f}|{longitude:.5f}|{race_info.get('date')}|{forecast_days}".encode("utf-8")
        ).hexdigest()
        cache_path = Path(FORECAST_CACHE_DIR) / f"{cache_key}.json"
        cache_status = "miss"
        payload = None
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                fetched_at = pd.to_datetime(cached.get("fetched_at_utc"), utc=True, errors="coerce")
                age_seconds = (pd.Timestamp.now(tz="UTC") - fetched_at).total_seconds() if pd.notna(fetched_at) else float("inf")
                if not force_refresh and age_seconds < 1800 and isinstance(cached.get("payload"), dict):
                    payload = cached["payload"]
                    fetched_at_utc = cached.get("fetched_at_utc")
                    cache_status = "hit"
            except Exception:
                payload = None
        if payload is None:
            response = get(FORECAST_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            fetched_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"fetched_at_utc": fetched_at_utc, "payload": payload}), encoding="utf-8")
        now_values = payload.get("current") or {}
        hourly = payload.get("hourly") or {}
        times = pd.to_datetime(hourly.get("time", []), errors="coerce")
        target_date = pd.to_datetime(race_info.get("date"), errors="coerce")
        if pd.isna(target_date):
            raise ValueError("Event date is unavailable; forecast cannot be aligned.")
        if getattr(times, "tz", None) is not None:
            target_mask = times.date == target_date.date()
        else:
            target_mask = times.date == target_date.date()
        frame = pd.DataFrame({"time": times})
        for name, values in hourly.items():
            if name == "time" or not isinstance(values, list) or len(values) != len(times):
                continue
            frame[name] = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
        day = frame.loc[target_mask].copy()
        if not day.empty:
            day_hours = pd.to_datetime(day["time"], errors="coerce").dt.hour
            race_window = day[day_hours.between(RACE_START_HOUR, RACE_END_HOUR)].copy()
            if race_window.empty:
                race_window = day
            hourly_records = []
            for row in race_window.to_dict("records"):
                timestamp = pd.to_datetime(row.get("time"), errors="coerce")
                code = row.get("weather_code")
                hourly_records.append({
                    "time": timestamp.isoformat(timespec="minutes") if pd.notna(timestamp) else None,
                    "temperature_2m": None if pd.isna(row.get("temperature_2m")) else float(row["temperature_2m"]),
                    "apparent_temperature": None if pd.isna(row.get("apparent_temperature")) else float(row["apparent_temperature"]),
                    "precipitation": None if pd.isna(row.get("precipitation")) else float(row["precipitation"]),
                    "rain": None if pd.isna(row.get("rain")) else float(row["rain"]),
                    "precipitation_probability": None if pd.isna(row.get("precipitation_probability")) else float(row["precipitation_probability"]),
                    "wind_speed_10m": None if pd.isna(row.get("wind_speed_10m")) else float(row["wind_speed_10m"]),
                    "wind_direction_10m": None if pd.isna(row.get("wind_direction_10m")) else float(row["wind_direction_10m"]),
                    "wind_gusts_10m": None if pd.isna(row.get("wind_gusts_10m")) else float(row["wind_gusts_10m"]),
                    "cloud_cover": None if pd.isna(row.get("cloud_cover")) else float(row["cloud_cover"]),
                    "relative_humidity_2m": None if pd.isna(row.get("relative_humidity_2m")) else float(row["relative_humidity_2m"]),
                    "weather_code": None if pd.isna(code) else int(code),
                    "weather_name": interpret_weather_code(code),
                })
            def avg(field: str) -> float | None:
                if field not in race_window:
                    return None
                values = pd.to_numeric(race_window[field], errors="coerce").dropna()
                return float(values.mean()) if not values.empty else None
            precipitation = pd.to_numeric(race_window.get("precipitation", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            rain = pd.to_numeric(race_window.get("rain", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            code_values = pd.to_numeric(race_window.get("weather_code", pd.Series(dtype=float)), errors="coerce").dropna()
            names = [interpret_weather_code(code) for code in code_values]
            dominant = pd.Series(names).mode().iloc[0] if names else "Unknown"
            current_race_condition = "WET" if float(rain.sum()) >= 1.0 else "DAMP" if float(rain.sum()) > 0 else "DRY"
            forecast = {
                "available": True,
                "condition": current_race_condition,
                "temperature": avg("temperature_2m"),
                "apparent_temperature": avg("apparent_temperature"),
                "precipitation": float(precipitation.sum()),
                "rain": float(rain.sum()),
                "precipitation_probability": avg("precipitation_probability"),
                "wind_speed": avg("wind_speed_10m"),
                "wind_direction": avg("wind_direction_10m"),
                "wind_gusts": avg("wind_gusts_10m"),
                "cloud_cover": avg("cloud_cover"),
                "relative_humidity": avg("relative_humidity_2m"),
                "dominant_weather": dominant,
                "hourly": hourly_records,
                "window": "Local 11:00–22:00 proxy; exact race start unavailable",
            }
        else:
            forecast = {"available": False, "reason": "The event date is outside the current forecast horizon.", "hourly": []}

        current_code = now_values.get("weather_code")
        current = {
            "available": bool(now_values),
            "time": payload.get("current", {}).get("time"),
            "temperature": now_values.get("temperature_2m"),
            "apparent_temperature": now_values.get("apparent_temperature"),
            "precipitation": now_values.get("precipitation"),
            "rain": now_values.get("rain"),
            "wind_speed": now_values.get("wind_speed_10m"),
            "wind_direction": now_values.get("wind_direction_10m"),
            "wind_gusts": now_values.get("wind_gusts_10m"),
            "cloud_cover": now_values.get("cloud_cover"),
            "relative_humidity": now_values.get("relative_humidity_2m"),
            "weather_code": current_code,
            "weather_name": interpret_weather_code(current_code),
            "source": "Open-Meteo current conditions",
        }
        lead_days = (target_date.date() - datetime.now(timezone.utc).date()).days
        forecast.update({
            "source": "Open-Meteo forecast",
            "lead_days": int(lead_days),
            "uncertainty_note": "Forecast is a single current run. Scenario weights are heuristic and are not calibrated probabilities; no archived forecast stability was available in this response.",
        })
        return {
            "race": race_name,
            "date": str(race_info.get("date")),
            "circuit": race_info.get("circuit"),
            "country": race_info.get("country"),
            "latitude": latitude,
            "longitude": longitude,
            "current": current,
            "forecast": forecast,
            "source": "Open-Meteo Forecast API",
            "timezone": payload.get("timezone"),
            "cache_status": cache_status,
            "retrieved_at_utc": fetched_at_utc,
            "forecast_horizon_days": forecast_days,
        }
    except Exception as error:
        result = {
            "race": race_name,
            "date": race_info.get("date"),
            "current": {"available": False, "source": "Open-Meteo current conditions", "reason": str(error)},
            "forecast": {"available": False, "source": "Open-Meteo forecast", "reason": str(error), "hourly": []},
            "source": "unavailable",
            "cache_status": "failure",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _WEATHER_FAILURE_CACHE[failure_key] = (monotonic(), deepcopy(result))
        return result


def weather_scenario_probabilities(weather_package: dict[str, Any]) -> dict[str, Any]:
    """Convert hourly precipitation forecast into documented scenario weights.

    Hourly precipitation probabilities are treated as independent solely to obtain
    a rough probability of any rain during the race window. The wet share is split
    using expected hourly rain amounts. This is a transparent scenario heuristic,
    not a calibrated event-level weather probability.
    """

    forecast = weather_package.get("forecast", {})
    if not forecast.get("available"):
        return {"available": False, "weights": {"DRY": 1.0}, "source": "forecast unavailable; dry is a simulation assumption"}
    hourly = forecast.get("hourly", [])
    probs = [min(1.0, max(0.0, float(row["precipitation_probability"]) / 100.0)) for row in hourly
             if row.get("precipitation_probability") is not None]
    rain = [max(0.0, float(row.get("rain") or row.get("precipitation") or 0.0)) for row in hourly]
    if probs:
        any_rain = 1.0 - float(np.prod([1.0 - p for p in probs]))
        weighted_rain = sum(p * (rain[i] if i < len(rain) else 0.0) for i, p in enumerate(probs))
        weighted_wet = sum(p for i, p in enumerate(probs) if i < len(rain) and rain[i] >= 1.0)
        wet_fraction = min(1.0, weighted_wet / sum(probs)) if sum(probs) else 0.0
        # When amounts are missing, leave precipitation risk in the damp bucket.
        if weighted_rain <= 0.0:
            wet_fraction = 0.0
        wet = any_rain * wet_fraction
        damp = any_rain - wet
        dry = 1.0 - any_rain
        total = dry + damp + wet
        weights = {"DRY": dry / total, "DAMP": damp / total, "WET": wet / total}
    else:
        condition = str(forecast.get("condition", "DRY")).upper()
        weights = {name: float(name == condition) for name in ("DRY", "DAMP", "WET")}
    return {
        "available": True,
        "weights": {key: round(value, 5) for key, value in weights.items()},
        "source": "hourly precipitation probability and forecast rain; independence heuristic, not calibrated",
        "uncertainty_note": forecast.get("uncertainty_note"),
    }


def weather_rows(weather: dict[str, Any]) -> list[dict[str, Any]]:
    """Create rows for the generated weather CSV, labeling fallback clearly."""

    hourly = weather.get("hourly") or []
    if not hourly:
        hourly = [
            {
                "time": None,
                "temperature_2m": weather.get("temperature"),
                "precipitation": weather.get("precipitation"),
                "rain": weather.get("rain"),
                "weather_code": None,
                "weather_name": weather.get("dominant_weather"),
                "precipitation_probability": weather.get("precipitation_probability"),
                "wind_speed_10m": weather.get("wind_speed"),
                "wind_direction_10m": weather.get("wind_direction"),
            }
        ]
    return [
        {
            "Season": weather.get("season"),
            "Race": weather.get("race"),
            "Circuit": weather.get("circuit"),
            "Country": weather.get("country"),
            "Date": weather.get("date"),
            "Time": row.get("time"),
            "Temperature": row.get("temperature_2m"),
            "Precipitation": row.get("precipitation"),
            "Rain": row.get("rain"),
            "PrecipitationProbability": row.get("precipitation_probability"),
            "WindSpeed": row.get("wind_speed_10m"),
            "WindDirection": row.get("wind_direction_10m"),
            "WeatherCode": row.get("weather_code"),
            "Weather": row.get("weather_name"),
            "Condition": weather.get("condition"),
            "Impact": weather.get("impact"),
            "Source": weather.get("source"),
        }
        for row in hourly
    ]
