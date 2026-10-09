from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.database import get_event
from services.weather_service import historical_weather, live_weather, refresh_historical_weather, store_weather


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh weather for one locally stored F1 event.")
    parser.add_argument("event_id", help="SQLite event id shown by /api/events.")
    args = parser.parse_args()
    event = get_event(args.event_id)
    if not event:
        parser.error("No SQLite event matches that id.")
    package = live_weather(event)
    store_weather(args.event_id, package)
    historical = refresh_historical_weather(event)
    print(json.dumps({"live": package, "historical_lookup": historical, "stored_history": historical_weather(args.event_id)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
