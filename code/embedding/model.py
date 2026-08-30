"""Single allowed embedding model (architecture Phase 3 / PRD)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIM = 384

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_ID)
        loaded = _model.get_sentence_embedding_dimension()
        if loaded != EXPECTED_DIM:
            raise RuntimeError(
                f"Unexpected embedding size {loaded}; {MODEL_ID} must be {EXPECTED_DIM}-d"
            )
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed with all-MiniLM-L6-v2. Same function is used later for questions."""
    if not texts:
        return np.zeros((0, EXPECTED_DIM), dtype=np.float32)
    vectors = get_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise RuntimeError("Embedding count must match input texts")
    if array.shape[1] != EXPECTED_DIM:
        raise RuntimeError(f"All vectors must have dimension {EXPECTED_DIM}")
    return array
