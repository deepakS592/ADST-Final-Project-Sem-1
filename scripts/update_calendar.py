from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calendar_import import parse_calendar_seasons, update_calendar


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import published F1 race-calendar metadata without creating driver entries or lap observations."
    )
    parser.add_argument("--seasons", default=None, help="Year, comma-separated years, or range (for example 2027 or 2026-2028). Default: next calendar year.")
    args = parser.parse_args()
    print(json.dumps(update_calendar(parse_calendar_seasons(args.seasons)), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
