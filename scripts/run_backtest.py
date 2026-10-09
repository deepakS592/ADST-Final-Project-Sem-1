from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.backtesting_service import backtest_event


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a pre-race prediction with one completed event.")
    parser.add_argument("event_id")
    parser.add_argument("driver")
    args = parser.parse_args()
    print(json.dumps(backtest_event(args.event_id, args.driver), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
