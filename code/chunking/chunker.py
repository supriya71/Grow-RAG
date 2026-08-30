"""Phase 2 chunking: one allowlisted page → chunks with fund identity on every row."""

from __future__ import annotations

import json
from typing import Any

from config.corpus import ALLOWLIST_SIZE, ALLOWLIST_URLS, CORPUS
from chunking.paths import CHUNKS_PATH, MANIFEST_PATH, ensure_chunks_dir
from chunking.split import chunk_page_text
from loading.paths import DOCUMENTS_PATH

_SLUG_BY_URL = {entry["url"]: entry["slug"] for entry in CORPUS}


def _slug_for(url: str) -> str:
    if url in _SLUG_BY_URL:
        return _SLUG_BY_URL[url]
    return url.rstrip("/").split("/")[-1]


def chunk_document(document: dict[str, str]) -> list[dict[str, str]]:
    url = document["url"]
    if url not in ALLOWLIST_URLS:
        raise ValueError(f"Refusing to chunk URL outside allowlist: {url}")

    fund_name = document["fund_name"]
    fetched_at = document["fetched_at"]
    slug = _slug_for(url)
    bodies = chunk_page_text(document["text"])
    chunks: list[dict[str, str]] = []
    for index, body in enumerate(bodies):
        chunks.append(
            {
                "chunk_id": f"{slug}__{index:04d}",
                "text": f"{fund_name}\n{body}",
                "fund_name": fund_name,
                "url": url,
                "fetched_at": fetched_at,
            }
        )
    return chunks


def chunk_corpus() -> dict[str, Any]:
    if not DOCUMENTS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {DOCUMENTS_PATH}. Run Phase 1 first: py -3 code/loading/run.py"
        )

    documents = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    ensure_chunks_dir()

    chunks: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for document in documents:
        url = document.get("url", "")
        if url in seen_urls:
            skipped.append({"url": url, "reason": "duplicate document url"})
            continue
        seen_urls.add(url)
        try:
            page_chunks = chunk_document(document)
        except ValueError as exc:
            skipped.append({"url": url, "reason": str(exc)})
            continue
        chunks.extend(page_chunks)

    urls_in_chunks = {c["url"] for c in chunks}
    if not urls_in_chunks.issubset(ALLOWLIST_URLS):
        raise RuntimeError("Invariant violated: chunk URL outside allowlist")
    if len(urls_in_chunks) > ALLOWLIST_SIZE:
        raise RuntimeError("Invariant violated: more source URLs than allowlist")

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk["url"]] = counts.get(chunk["url"], 0) + 1

    manifest = {
        "source_documents": len(documents),
        "chunk_count": len(chunks),
        "urls": sorted(urls_in_chunks),
        "chunks_per_url": counts,
        "skipped": skipped,
        "chunks_path": str(CHUNKS_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
