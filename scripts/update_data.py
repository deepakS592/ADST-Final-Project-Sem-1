from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import update_historical_data
from data_preparation import prepare_data
from services.data_service import import_project_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh FastF1 race laps and load the results into SQLite.")
    parser.add_argument("--seasons", default=None, help="Season year or range, for example 2024-2026.")
    parser.add_argument("--force", action="store_true", help="Refresh completed source events even when a cached CSV exists.")
    parser.add_argument("--local-only", action="store_true", help="Rebuild SQLite from the included local CSV snapshot without network calls.")
    args = parser.parse_args()
    if not args.local_only:
        update_historical_data(args.seasons, force=args.force)
        prepare_data()
    report = import_project_data(force=True)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
