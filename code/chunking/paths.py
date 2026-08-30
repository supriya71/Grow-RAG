from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
CHUNKS_PATH = CHUNKS_DIR / "chunks.json"
MANIFEST_PATH = CHUNKS_DIR / "manifest.json"


def ensure_chunks_dir() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
