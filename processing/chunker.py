"""
processing/chunker.py
──────────────────────
Splits extracted page-texts into overlapping word-level chunks with
rich metadata for retrieval.

Strategy
────────
1. Sentence-aware splitting  — prefer to cut on sentence boundaries.
2. Fixed max-word window      — CHUNK_SIZE words per chunk.
3. Sliding overlap            — CHUNK_OVERLAP words carried over.
4. Metadata attached per chunk: doc_id, file_name, source, page, chunk_index.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import config

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str            # "<doc_id>_p<page>_c<idx>"
    doc_id: str
    file_name: str
    source: str
    page: int
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


# ── Sentence splitter ──────────────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Naïve but fast sentence splitter."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ── Core chunker ───────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split *text* into overlapping word windows.
    Returns a list of chunk strings.
    """
    sentences = _split_sentences(text)
    words: List[str] = []
    for sent in sentences:
        words.extend(sent.split())

    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += chunk_size - overlap   # slide with overlap

    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: List[Tuple[int, str]],
    doc_id: str,
    file_name: str,
    source: str = "gdrive",
    extra_metadata: dict | None = None,
) -> List[Chunk]:
    """
    Convert (page_num, text) pairs into a flat list of Chunk objects.

    Parameters
    ----------
    pages          : output of parser.parse_document()
    doc_id         : unique document identifier (e.g. Drive file ID)
    file_name      : human-readable file name
    source         : provenance tag (default 'gdrive')
    extra_metadata : any extra key-value pairs to attach to every chunk
    """
    extra_metadata = extra_metadata or {}
    chunks: List[Chunk] = []

    for page_num, page_text in pages:
        raw_chunks = _chunk_text(
            page_text,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_p{page_num}_c{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    file_name=file_name,
                    source=source,
                    page=page_num,
                    chunk_index=idx,
                    text=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "source": source,
                        "page": page_num,
                        "chunk_index": idx,
                        **extra_metadata,
                    },
                )
            )

    log.info(
        "Chunked '%s': %d pages → %d chunks",
        file_name, len(pages), len(chunks),
    )
    return chunks
