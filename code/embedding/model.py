"""Single allowed embedding model (architecture Phase 3 / PRD).

Two backends, both loading the same all-MiniLM-L6-v2 weights -> 384-d
L2-normalized vectors:

- "sentence_transformers" (default): PyTorch via sentence-transformers, for dev.
- "fastembed": ONNX via fastembed — no torch, for memory-constrained hosts.

Pick with the EMBED_BACKEND environment variable. Importing this module never
pulls in torch/onnx — backends are imported lazily on first use.
"""

from __future__ import annotations

import os

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIM = 384

_model = None


class _TorchBackend:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MODEL_ID)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


class _FastEmbedBackend:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=MODEL_ID)

    def encode(self, texts: list[str]) -> np.ndarray:
        array = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)


def _default_backend() -> str:
    return os.getenv("EMBED_BACKEND", "sentence_transformers").strip().lower()


def get_model():
    """Return the singleton encoder backend (lazy, created once)."""
    global _model
    if _model is None:
        backend = _default_backend()
        if backend == "fastembed":
            _model = _FastEmbedBackend()
        elif backend in {"sentence_transformers", "sentence-transformers", "torch"}:
            _model = _TorchBackend()
        else:
            raise RuntimeError(f"Unknown EMBED_BACKEND: {backend!r}")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed with all-MiniLM-L6-v2 (384-d, L2-normalized). Same function for questions."""
    if not texts:
        return np.zeros((0, EXPECTED_DIM), dtype=np.float32)
    array = get_model().encode(list(texts))
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise RuntimeError("Embedding count must match input texts")
    if array.shape[1] != EXPECTED_DIM:
        raise RuntimeError(f"All vectors must have dimension {EXPECTED_DIM}")
    if not np.isfinite(array).all():
        raise RuntimeError("Embedding produced non-finite values")
    return array