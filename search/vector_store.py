"""
search/vector_store.py
───────────────────────
FAISS-backed vector store with JSON metadata sidecar.

Layout on disk
──────────────
  data/faiss_index/index.faiss   ← FAISS flat-IP index
  data/faiss_index/metadata.json ← list of chunk metadata dicts

Thread-safety
─────────────
A threading.Lock guards all write operations so the FastAPI worker
threads don't race on index updates.
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

import config
from embedding.embedder import get_embedding_dim
from processing.chunker import Chunk

log = logging.getLogger(__name__)

_lock = threading.Lock()


class VectorStore:
    """Singleton FAISS flat-IP index with metadata sidecar."""

    _instance: Optional["VectorStore"] = None

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    # ── Init ──────────────────────────────────────────────────────────────────

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._dim = get_embedding_dim()
        self._load_or_create()
        self._initialized = True

    def _load_or_create(self) -> None:
        import os
        if os.path.exists(config.FAISS_INDEX_PATH) and os.path.exists(config.FAISS_META_PATH):
            log.info("Loading existing FAISS index from %s", config.FAISS_INDEX_PATH)
            self._index = faiss.read_index(config.FAISS_INDEX_PATH)
            with open(config.FAISS_META_PATH, "r") as f:
                self._metadata: List[Dict[str, Any]] = json.load(f)
        else:
            log.info("Creating new FAISS index (dim=%d).", self._dim)
            # Inner-product with L2-normalised vectors == cosine similarity
            self._index = faiss.IndexFlatIP(self._dim)
            self._metadata = []

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        faiss.write_index(self._index, config.FAISS_INDEX_PATH)
        with open(config.FAISS_META_PATH, "w") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> int:
        """
        Add chunks + their embeddings to the index.
        Skips chunks that are already stored (deduplication by chunk_id).
        Returns the number of newly added vectors.
        """
        self._ensure_init()

        existing_ids = {m["chunk_id"] for m in self._metadata}
        new_chunks, new_embs = [], []

        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_embs.append(emb)

        if not new_chunks:
            log.info("No new chunks to add (all duplicates).")
            return 0

        emb_matrix = np.array(new_embs, dtype=np.float32)

        with _lock:
            self._index.add(emb_matrix)
            for chunk in new_chunks:
                self._metadata.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "file_name": chunk.file_name,
                        "source": chunk.source,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        **chunk.metadata,
                    }
                )
            self._save()

        log.info("Added %d new chunks. Total vectors: %d", len(new_chunks), self._index.ntotal)
        return len(new_chunks)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks belonging to a given document (not supported by FlatIP natively)."""
        self._ensure_init()
        with _lock:
            # Rebuild index without the target doc
            keep_idx = [i for i, m in enumerate(self._metadata) if m["doc_id"] != doc_id]
            removed = len(self._metadata) - len(keep_idx)
            if removed == 0:
                return 0

            kept_meta = [self._metadata[i] for i in keep_idx]
            # Collect kept vectors
            all_vecs = faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self._dim
            ).reshape(self._index.ntotal, self._dim)
            kept_vecs = all_vecs[keep_idx].astype(np.float32)

            new_index = faiss.IndexFlatIP(self._dim)
            if len(kept_vecs):
                new_index.add(kept_vecs)

            self._index = new_index
            self._metadata = kept_meta
            self._save()

        log.info("Removed %d chunks for doc_id='%s'.", removed, doc_id)
        return removed

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = config.TOP_K,
        filter_source: Optional[str] = None,
        filter_file_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks.

        Parameters
        ----------
        query_emb       : (1, dim) float32 array
        top_k           : number of results
        filter_source   : optional metadata filter on 'source'
        filter_file_name: optional metadata filter on 'file_name'
        """
        self._ensure_init()

        if self._index.ntotal == 0:
            return []

        # Over-fetch to allow post-filter
        fetch_k = min(top_k * 10, self._index.ntotal)
        scores, indices = self._index.search(query_emb, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]

            # Metadata filters
            if filter_source and meta.get("source") != filter_source:
                continue
            if filter_file_name and meta.get("file_name") != filter_file_name:
                continue

            results.append({**meta, "score": float(score)})
            if len(results) >= top_k:
                break

        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        self._ensure_init()
        doc_ids = {m["doc_id"] for m in self._metadata}
        return {
            "total_vectors": self._index.ntotal,
            "total_chunks": len(self._metadata),
            "total_documents": len(doc_ids),
            "embedding_dim": self._dim,
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
vector_store = VectorStore()
