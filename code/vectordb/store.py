"""Phase 4 vector store: persist Phase 2/3 chunks + vectors into local ChromaDB.

Rebuild-from-allowlist only: the collection is wiped and rebuilt from the last
Phase 3 output (records.json + vectors.npy). No arbitrary URL is ever upserted.
"""

from __future__ import annotations

import json
from typing import Any

import chromadb

from config.corpus import ALLOWLIST_SIZE, ALLOWLIST_URLS
from embedding.model import EXPECTED_DIM, MODEL_ID
from embedding.paths import RECORDS_PATH, VECTORS_PATH
from vectordb.paths import (
    CHROMA_DIR,
    COLLECTION_NAME,
    MANIFEST_PATH,
    ensure_vectordb_dir,
)

REQUIRED_RECORD_FIELDS = ("chunk_id", "fund_name", "url", "fetched_at", "text", "vector_index")
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def build_vector_store() -> dict[str, Any]:
    if not VECTORS_PATH.is_file() or not RECORDS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing embeddings. Run Phase 3 first: py -3 code/embedding/run.py"
        )

    vectors = _load_vectors()
    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))

    if vectors.shape != (len(records), EXPECTED_DIM):
        raise RuntimeError(
            f"Expected one {EXPECTED_DIM}-d vector per record, got {vectors.shape}"
        )

    ids, documents, metadatas = _rows_from_records(records)

    ensure_vectordb_dir()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if COLLECTION_NAME in {collection.name for collection in client.list_collections()}:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=dict(COLLECTION_METADATA),
    )
    collection.add(
        ids=ids,
        embeddings=vectors.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    _verify_collection(collection, ids, metadatas)

    fetched_at_values = sorted(record["fetched_at"] for record in records)
    urls_in_store = sorted({meta["url"] for meta in metadatas})

    manifest = {
        "collection_name": COLLECTION_NAME,
        "model_id": MODEL_ID,
        "dimension": EXPECTED_DIM,
        "chunk_count": len(records),
        "urls": urls_in_store,
        "urls_within_allowlist": True,
        "fetched_at_min": fetched_at_values[0],
        "fetched_at_max": fetched_at_values[-1],
        "chroma_dir": str(CHROMA_DIR),
        "manifest_path": str(MANIFEST_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_vectors():
    import numpy as np

    vectors = np.load(VECTORS_PATH, allow_pickle=False)
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embeddings contain non-finite values")
    return vectors


def _rows_from_records(records: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, str]]]:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    seen: set[str] = set()

    for index, record in enumerate(records):
        missing = [key for key in REQUIRED_RECORD_FIELDS if key not in record]
        if missing:
            raise RuntimeError(f"Record missing fields: {missing}")
        if record["vector_index"] != index:
            raise RuntimeError(
                f"Record {index} vector_index mismatch: {record['vector_index']}"
            )
        url = record["url"]
        if url not in ALLOWLIST_URLS:
            raise RuntimeError(f"Refusing to persist URL outside allowlist: {url}")
        chunk_id = record["chunk_id"]
        if chunk_id in seen:
            raise RuntimeError(f"Duplicate chunk_id: {chunk_id}")
        seen.add(chunk_id)
        ids.append(chunk_id)
        documents.append(record["text"])
        metadatas.append(
            {
                "fund_name": record["fund_name"],
                "url": url,
                "fetched_at": record["fetched_at"],
            }
        )
    return ids, documents, metadatas


def _verify_collection(collection, ids: list[str], metadatas: list[dict[str, str]]) -> None:
    if collection.count() != len(ids):
        raise RuntimeError(
            f"Chroma count mismatch: collection has {collection.count()}, expected {len(ids)}"
        )
    readback = collection.get(ids=ids, include=["metadatas"])
    if len(readback["ids"]) != len(ids):
        raise RuntimeError("Chroma read-back count mismatch")
    stored_urls = {meta["url"] for meta in readback["metadatas"]}
    if not stored_urls.issubset(ALLOWLIST_URLS):
        raise RuntimeError("Invariant violated: stored URL outside allowlist")
    if len(stored_urls) > ALLOWLIST_SIZE:
        raise RuntimeError("Invariant violated: more stored URLs than allowlist")
    source_urls = {meta["url"] for meta in metadatas}
    if stored_urls != source_urls:
        raise RuntimeError("Invariant violated: Chroma URLs differ from sources")