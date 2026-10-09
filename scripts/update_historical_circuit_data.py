from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from historical_import import update_historical_circuit_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import only missing historical race laps needed by scheduled circuits; no practice or qualifying sessions are fetched."
    )
    parser.add_argument("--event-id", action="append", default=None, help="Limit backfill to a current/upcoming event ID. May be supplied more than once.")
    args = parser.parse_args()
    result = update_historical_circuit_data(event_ids=set(args.event_id) if args.event_id else None)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
