from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.data_service import import_project_data


if __name__ == "__main__":
    print(json.dumps(import_project_data(force=True), indent=2, default=str))
