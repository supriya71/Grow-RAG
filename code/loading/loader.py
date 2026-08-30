"""Phase 1 data loading: GET the frozen allowlist only; persist successful pages."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import requests

from config.corpus import ALLOWLIST_SIZE, ALLOWLIST_URLS, CORPUS, CorpusEntry, validate_corpus
from loading.parse import html_to_text
from loading.paths import (
    DOCUMENTS_PATH,
    FAILURES_PATH,
    HTML_DIR,
    MANIFEST_PATH,
    ensure_raw_dirs,
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
MAX_ATTEMPTS = 3
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
REQUEST_TIMEOUT_S = 30


class AllowlistError(ValueError):
    """Raised if a fetch is attempted outside the frozen corpus."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fetch_html(url: str) -> str:
    if url not in ALLOWLIST_URLS:
        raise AllowlistError(f"Refusing to fetch URL outside allowlist: {url}")

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                timeout=REQUEST_TIMEOUT_S,
                allow_redirects=True,
            )
            if response.status_code in RETRY_STATUSES:
                last_error = RuntimeError(f"HTTP {response.status_code}")
            elif response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}")
            else:
                final_url = response.url.split("#")[0].rstrip("/")
                allowed = url.rstrip("/")
                if final_url != allowed and not final_url.startswith(allowed):
                    # Stay on the same allowlisted resource; do not accept a different page.
                    if final_url not in {u.rstrip("/") for u in ALLOWLIST_URLS}:
                        raise RuntimeError(f"Redirect left allowlist: {response.url}")
                return response.text
        except AllowlistError:
            raise
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if isinstance(exc, RuntimeError) and "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
                raise
        if attempt < MAX_ATTEMPTS:
            time.sleep(attempt)
    raise RuntimeError(f"Failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _document_from_entry(entry: CorpusEntry, html: str, fetched_at: str) -> dict[str, str]:
    text = html_to_text(html)
    if not text:
        raise RuntimeError("Parsed page text was empty")
    return {
        "fund_name": entry["fund_name"],
        "url": entry["url"],
        "text": text,
        "fetched_at": fetched_at,
    }


def load_corpus() -> dict[str, Any]:
    validate_corpus()
    ensure_raw_dirs()

    documents: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for entry in CORPUS:
        fetched_at = _utc_now()
        try:
            html = _fetch_html(entry["url"])
            document = _document_from_entry(entry, html, fetched_at)
            (HTML_DIR / f"{entry['slug']}.html").write_text(html, encoding="utf-8")
            documents.append(document)
        except Exception as exc:
            failures.append(
                {
                    "fund_name": entry["fund_name"],
                    "url": entry["url"],
                    "fetched_at": fetched_at,
                    "error": str(exc),
                }
            )

    if len(documents) > ALLOWLIST_SIZE:
        raise RuntimeError("Invariant violated: more documents than allowlisted URLs")

    DOCUMENTS_PATH.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    FAILURES_PATH.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "allowlist_size": ALLOWLIST_SIZE,
        "attempted": len(CORPUS),
        "succeeded": len(documents),
        "failed": len(failures),
        "fetched_at_by_url": {doc["url"]: doc["fetched_at"] for doc in documents},
        "documents_path": str(DOCUMENTS_PATH),
        "failures_path": str(FAILURES_PATH),
        "html_dir": str(HTML_DIR),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
