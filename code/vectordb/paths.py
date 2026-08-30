from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VECTORDB_DIR = REPO_ROOT / "data" / "vectordb"
CHROMA_DIR = VECTORDB_DIR / "chroma"
MANIFEST_PATH = VECTORDB_DIR / "manifest.json"

COLLECTION_NAME = "groww_faq_chunks"


def ensure_vectordb_dir() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)