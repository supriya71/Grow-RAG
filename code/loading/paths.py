from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
HTML_DIR = RAW_DIR / "html"
DOCUMENTS_PATH = RAW_DIR / "documents.json"
FAILURES_PATH = RAW_DIR / "failures.json"
MANIFEST_PATH = RAW_DIR / "manifest.json"


def ensure_raw_dirs() -> None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
