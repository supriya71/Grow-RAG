"""CLI: python code/loading/run.py  (from repo root)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from loading.loader import load_corpus  # noqa: E402


def main() -> int:
    manifest = load_corpus()
    print(json.dumps(manifest, indent=2))
    if manifest["succeeded"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
