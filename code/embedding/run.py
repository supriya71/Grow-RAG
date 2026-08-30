"""CLI: python code/embedding/run.py  (from repo root)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from embedding.embedder import embed_chunks  # noqa: E402


def main() -> int:
    manifest = embed_chunks()
    print(json.dumps(manifest, indent=2))
    if manifest["vector_count"] != manifest["chunk_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
