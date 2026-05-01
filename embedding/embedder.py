"""
embedding/embedder.py
──────────────────────
Wraps SentenceTransformers to produce dense vector embeddings.

• Singleton model — loaded once, reused across requests.
• Batch processing for throughput.
• Returns numpy float32 arrays (FAISS-compatible).
"""

import logging
from typing import List

import numpy as np

import config

log = logging.getLogger(__name__)

_model = None  # singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        log.info("Embedding model loaded (dim=%d).", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns
    -------
    np.ndarray of shape (len(texts), embedding_dim), dtype float32
    """
    if not texts:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,   # cosine-similarity friendly
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns
    -------
    np.ndarray of shape (1, embedding_dim), dtype float32
    """
    return embed_texts([query])


def get_embedding_dim() -> int:
    """Return the output dimension of the current model."""
    return _get_model().get_sentence_embedding_dimension()
