"""Phase 3: embed chunk text with all-MiniLM-L6-v2 only."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from chunking.paths import CHUNKS_PATH
from config.corpus import ALLOWLIST_URLS
from embedding.model import EXPECTED_DIM, MODEL_ID, embed_texts
from embedding.paths import (
    MANIFEST_PATH,
    RECORDS_PATH,
    VECTORS_PATH,
    ensure_embeddings_dir,
)

REQUIRED_CHUNK_FIELDS = ("chunk_id", "text", "fund_name", "url", "fetched_at")


def embed_chunks() -> dict[str, Any]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CHUNKS_PATH}. Run Phase 2 first: py -3 code/chunking/run.py"
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise RuntimeError("No chunks to embed")

    for chunk in chunks:
        missing = [key for key in REQUIRED_CHUNK_FIELDS if key not in chunk]
        if missing:
            raise RuntimeError(f"Chunk {chunk.get('chunk_id')} missing {missing}")
        if chunk["url"] not in ALLOWLIST_URLS:
            raise RuntimeError(f"Refusing to embed URL outside allowlist: {chunk['url']}")

    # Embed chunk text only. Identity already lives in text (fund_name prefix)
    # and is stored again as metadata — not used as a substitute source.
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)

    if vectors.shape != (len(chunks), EXPECTED_DIM):
        raise RuntimeError(
            f"Expected one {EXPECTED_DIM}-d vector per chunk, got {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embeddings contain non-finite values")

    records = [
        {
            "chunk_id": chunk["chunk_id"],
            "fund_name": chunk["fund_name"],
            "url": chunk["url"],
            "fetched_at": chunk["fetched_at"],
            "text": chunk["text"],
            "vector_index": index,
        }
        for index, chunk in enumerate(chunks)
    ]

    ensure_embeddings_dir()
    np.save(VECTORS_PATH, vectors)
    RECORDS_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "model_id": MODEL_ID,
        "dimension": EXPECTED_DIM,
        "chunk_count": len(chunks),
        "vector_count": int(vectors.shape[0]),
        "embedding_input": "chunk.text",
        "vectors_path": str(VECTORS_PATH),
        "records_path": str(RECORDS_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
