"""Offline-only access to historical race laps and calendar rows from Jolpica F1."""

from __future__ import annotations

import json
import time
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_ROOT = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "F1PitStopStrategy/1.1 (offline historical data import)"
PAGE_SIZE = 100


class JolpicaError(RuntimeError):
    """Raised when the external race-history service cannot return usable data."""


class JolpicaF1Client:
    """Small paginated client for the public Ergast-compatible Jolpica API."""

    def __init__(self, *, timeout: float = 30.0):
        self.timeout = float(timeout)
        self._last_request_at = 0.0

    def _get_json(self, route: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = {"limit": PAGE_SIZE, **(params or {})}
        url = f"{API_ROOT}/{route.lstrip('/')}?{urlencode(query)}"
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        for attempt in range(4):
            wait = 0.30 - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self._last_request_at = time.monotonic()
                break
            except HTTPError as error:
                self._last_request_at = time.monotonic()
                if error.code == 429 and attempt < 3:
                    retry_after = error.headers.get("Retry-After") if error.headers else None
                    try:
                        delay = max(1.0, float(retry_after))
                    except (TypeError, ValueError):
                        delay = min(30.0, 2.0 ** (attempt + 1))
                    time.sleep(delay)
                    continue
                raise JolpicaError(f"Jolpica request failed for {route}: {error}") from error
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                raise JolpicaError(f"Jolpica request failed for {route}: {error}") from error
        else:
            raise JolpicaError(f"Jolpica request was throttled for {route} after retries.")
        if not isinstance(payload, dict) or not isinstance(payload.get("MRData"), dict):
            raise JolpicaError(f"Jolpica returned an invalid response for {route}.")
        return payload["MRData"]

    def _race_pages(self, route: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            data = self._get_json(route, {"offset": offset})
            table = data.get("RaceTable") or {}
            page = table.get("Races") or []
            rows.extend(row for row in page if isinstance(row, dict))
            total = int(data.get("total") or len(rows))
            limit = max(1, int(data.get("limit") or PAGE_SIZE))
            if not page or offset + limit >= total:
                return rows
            offset += limit

    def fetch_schedule(self, season: int | None = None) -> list[dict[str, Any]]:
        """Fetch calendar metadata only; no session observations are requested."""
        route = f"{int(season)}/races.json" if season is not None else "races.json"
        return self._race_pages(route)

    def fetch_laps(self, season: int, round_number: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Fetch completed race lap times, paging through driver-lap records."""
        route = f"{int(season)}/{int(round_number)}/laps.json"
        offset = 0
        race_meta: dict[str, Any] | None = None
        laps: dict[tuple[int, str], dict[str, Any]] = {}
        while True:
            data = self._get_json(route, {"offset": offset})
            table = data.get("RaceTable") or {}
            races = table.get("Races") or []
            if races and race_meta is None:
                race_meta = dict(races[0])
            if races:
                for lap in races[0].get("Laps") or []:
                    lap_number = int(lap.get("number") or 0)
                    for timing in lap.get("Timings") or []:
                        driver = str(timing.get("driverId") or "").strip().casefold()
                        lap_seconds = _parse_lap_seconds(timing.get("time"))
                        if lap_number <= 0 or not driver or lap_seconds is None:
                            continue
                        laps[(lap_number, driver)] = {
                            "driver": driver,
                            "lap_number": lap_number,
                            "lap_time": lap_seconds,
                            "position": _number(timing.get("position")),
                        }
            total = int(data.get("total") or 0)
            limit = max(1, int(data.get("limit") or PAGE_SIZE))
            if not races or offset + limit >= total:
                break
            offset += limit
        return race_meta, list(laps.values())

    def fetch_pit_stops(self, season: int, round_number: int) -> list[dict[str, Any]]:
        """Fetch observed race pit stops (available from 2011 in this source)."""
        route = f"{int(season)}/{int(round_number)}/pitstops.json"
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            data = self._get_json(route, {"offset": offset})
            races = (data.get("RaceTable") or {}).get("Races") or []
            page: list[dict[str, Any]] = []
            if races:
                for pit in races[0].get("PitStops") or []:
                    driver = str(pit.get("driverId") or "").strip().casefold()
                    lap = _number(pit.get("lap"))
                    stop = _number(pit.get("stop"))
                    if driver and lap is not None:
                        page.append({
                            "driver": driver,
                            "pit_lap": int(lap),
                            "stop_number": int(stop) if stop is not None else None,
                            "duration_seconds": _parse_lap_seconds(pit.get("duration")),
                        })
            rows.extend(page)
            total = int(data.get("total") or len(rows))
            limit = max(1, int(data.get("limit") or PAGE_SIZE))
            if not races or offset + limit >= total:
                return rows
            offset += limit


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _parse_lap_seconds(value: Any) -> float | None:
    """Parse lap/pit durations such as ``1:32.800`` or ``20.655``."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds if seconds > 0 else None


def race_date(row: dict[str, Any]) -> date | None:
    try:
        return date.fromisoformat(str(row.get("date", ""))[:10])
    except ValueError:
        return None


def circuit_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    circuit = row.get("Circuit") or {}
    location = circuit.get("Location") or {}
    return (
        str(circuit.get("circuitName") or "").strip(),
        str(location.get("country") or "").strip(),
        str(location.get("locality") or "").strip(),
    )
