from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"
RECORDS_PATH = EMBEDDINGS_DIR / "records.json"
MANIFEST_PATH = EMBEDDINGS_DIR / "manifest.json"


def ensure_embeddings_dir() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
