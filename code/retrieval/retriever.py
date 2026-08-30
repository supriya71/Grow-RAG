"""Phase 5 retrieval: question → embed (all-MiniLM-L6-v2) → Chroma top-k chunks.

Only facts from the five-page corpus are returned. This module never calls
Mistral, never follows new URLs, and never computes/compares returns.
"""

from __future__ import annotations

import re
from typing import Any

import chromadb

from config.corpus import ALLOWLIST_URLS
from embedding.model import embed_texts
from vectordb.paths import CHROMA_DIR, COLLECTION_NAME

DEFAULT_K = 5
QUERY_POOL = 12
MAX_COSINE_DISTANCE = 0.75

# Alias → (canonical fund_name, allowlisted URL). "Flexi Cap" intentionally maps
# to the corpus fund_name whose page slug is hdfc-equity-fund-direct-growth.
_ALIASES: tuple[tuple[str, str, str], ...] = (
    (
        "large cap",
        "HDFC Large Cap Fund Direct Growth",
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    ),
    (
        "flexi cap",
        "HDFC Flexi Cap Fund Direct Growth",
        "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    ),
    (
        "elss",
        "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    ),
    (
        "tax saver",
        "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    ),
    (
        "small cap",
        "HDFC Small Cap Fund Direct Growth",
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    ),
    (
        "balanced advantage",
        "HDFC Balanced Advantage Fund Direct Growth",
        "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    ),
)


def detect_funds(question: str) -> list[str]:
    """Which of the five funds the question *confidently* names (corpus alias order).

    Only counts when the user actually mentions "HDFC" — otherwise an off-corpus
    fund that shares a category word (e.g. "Parag Parikh Flexi Cap") must not be
    bent toward one of our funds (wrong-fund risk, PRD #8 / #12).
    """
    normalized = re.sub(r"[^a-z0-9 ]", " ", question.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    matched: list[str] = []
    if "hdfc" not in normalized:
        return matched
    for _alias, _fund_name, url in _ALIASES:
        if url in matched:
            continue
        if _alias in normalized:
            matched.append(url)
    return matched


def retrieve(question: str, k: int = DEFAULT_K) -> dict[str, Any]:
    """Return top-k chunks (with metadata + cosine distance) a generator may cite.

    - Named fund + fact: chunks of that fund are surfaced first.
    - No/very-low similarity: empty result (honest miss) — nothing is invented.
    - Question about a fund outside the five: still only the five can appear.
    """
    question = (question or "").strip()
    matched = detect_funds(question)
    if not question:
        return _result(question, [], matched)

    query_vectors = embed_texts([question])

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except ValueError as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' missing. "
            f"Run Phase 4 first: py -3 code/vectordb/run.py"
        ) from exc

    pool_size = max(QUERY_POOL, k)
    response = collection.query(
        query_embeddings=query_vectors.tolist(),
        n_results=pool_size,
        include=["metadatas", "documents", "distances"],
    )

    rows: list[dict[str, Any]] = []
    for row_id, meta, doc, dist in zip(
        response["ids"][0],
        response["metadatas"][0],
        response["documents"][0],
        response["distances"][0],
    ):
        url = meta.get("url", "")
        if url not in ALLOWLIST_URLS:
            # Invariant guard: never surface a non-allowlisted source.
            continue
        if float(dist) > MAX_COSINE_DISTANCE:
            # Too dissimilar to be a defensible match.
            continue
        rows.append(
            {
                "chunk_id": row_id,
                "text": doc,
                "fund_name": meta.get("fund_name", ""),
                "url": url,
                "fetched_at": meta.get("fetched_at", ""),
                "distance": round(float(dist), 4),
            }
        )

    if len(matched) == 1:
        # Unambiguous fund: prefer its chunks, keep original similarity order within groups.
        rows = _prefer_fund(rows, matched[0])

    return _result(question, rows[:k], matched)


def _prefer_fund(rows: list[dict[str, Any]], url: str) -> list[dict[str, Any]]:
    own = [row for row in rows if row["url"] == url]
    other = [row for row in rows if row["url"] != url]
    return own + other


def _result(question: str, rows: list[dict[str, Any]], matched: list[str]) -> dict[str, Any]:
    fetched = [row["fetched_at"] for row in rows]
    return {
        "query": question,
        "count": len(rows),
        "chunks": rows,
        "matched_funds": matched,
        "citation_urls": sorted({row["url"] for row in rows}),
        "fetched_at_min": min(fetched) if fetched else None,
        "fetched_at_max": max(fetched) if fetched else None,
        "empty": not rows,
    }