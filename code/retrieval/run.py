"""CLI: python code/retrieval/run.py "<question>" [--k 5]  (from repo root)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.retriever import DEFAULT_K, retrieve  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Retrieve top-k chunks for a question")
    parser.add_argument("question", help="The user question to retrieve for")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to return")
    args = parser.parse_args()

    result = retrieve(args.question, k=args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())