"""
api/routes.py
──────────────
FastAPI route definitions.

Endpoints
─────────
POST /sync-drive          → pull files from Google Drive and index them
POST /sync-local          → index files already in data/uploads/ (for testing)
POST /ask                 → RAG question-answering
GET  /status              → vector store statistics
GET  /documents           → list indexed documents
DELETE /documents/{doc_id}→ remove a document from the index
GET  /health              → liveness probe
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from connectors.gdrive import sync_drive, DriveFile
from embedding.embedder import embed_texts, embed_query
from processing.parser import parse_document
from processing.chunker import chunk_pages, Chunk
from search.vector_store import vector_store
from api.llm import generate_answer

log = logging.getLogger(__name__)
router = APIRouter()


# ── Pydantic models ────────────────────────────────────────────────────────────

class SyncDriveRequest(BaseModel):
    folder_id: Optional[str] = Field(None, description="Google Drive folder ID to sync (optional)")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask")
    top_k: int = Field(config.TOP_K, ge=1, le=20)
    filter_source: Optional[str] = Field(None, description="Filter by source (e.g. 'gdrive')")
    filter_file_name: Optional[str] = Field(None, description="Filter by file name")


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_used: int
    query: str


class SyncResponse(BaseModel):
    status: str
    files_synced: int
    chunks_added: int
    files: List[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ingest_file(
    file_path: str,
    doc_id: str,
    file_name: str,
    source: str = "local",
    extra_metadata: Optional[Dict] = None,
) -> int:
    """Parse → chunk → embed → store. Returns number of chunks added."""
    pages = parse_document(file_path)
    if not pages:
        log.warning("No text extracted from %s", file_name)
        return 0

    chunks = chunk_pages(
        pages,
        doc_id=doc_id,
        file_name=file_name,
        source=source,
        extra_metadata=extra_metadata,
    )

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    added = vector_store.add_chunks(chunks, embeddings)
    return added


def _file_doc_id(file_path: str) -> str:
    """Stable doc_id derived from file content hash."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/")
async def root():
    return FileResponse("static/index.html")


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/status")
async def status():
    """Return vector store statistics."""
    return vector_store.stats()


@router.get("/documents")
async def list_documents():
    """Return unique documents currently indexed."""
    store = vector_store
    store._ensure_init()
    seen = {}
    for meta in store._metadata:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "file_name": meta["file_name"],
                "source": meta["source"],
                "chunk_count": 0,
            }
        seen[doc_id]["chunk_count"] += 1
    return list(seen.values())


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document and all its chunks from the index."""
    removed = vector_store.delete_by_doc_id(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found.")
    return {"status": "deleted", "chunks_removed": removed, "doc_id": doc_id}


@router.post("/sync-drive", response_model=SyncResponse)
async def sync_drive_endpoint(body: SyncDriveRequest = SyncDriveRequest()):
    """
    Sync documents from Google Drive.
    Downloads files, parses, chunks, embeds, and stores them.
    """
    try:
        drive_files: List[DriveFile] = sync_drive(folder_id=body.folder_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drive sync failed: {e}")

    total_added = 0
    synced_names: List[str] = []

    for df in drive_files:
        added = _ingest_file(
            file_path=df.local_path,
            doc_id=df.file_id,
            file_name=df.file_name,
            source="gdrive",
            extra_metadata=df.metadata,
        )
        total_added += added
        synced_names.append(df.file_name)

    return SyncResponse(
        status="ok",
        files_synced=len(drive_files),
        chunks_added=total_added,
        files=synced_names,
    )


@router.post("/sync-local", response_model=SyncResponse)
async def sync_local_endpoint():
    """
    Index all files already present in data/uploads/.
    Useful for testing without Google Drive credentials.
    """
    upload_dir = config.UPLOAD_DIR
    supported_exts = {".pdf", ".docx", ".txt"}
    files = [p for p in upload_dir.iterdir() if p.suffix.lower() in supported_exts]

    if not files:
        return SyncResponse(status="ok", files_synced=0, chunks_added=0, files=[])

    total_added = 0
    synced_names = []

    for file_path in files:
        doc_id = _file_doc_id(str(file_path))
        added = _ingest_file(
            file_path=str(file_path),
            doc_id=doc_id,
            file_name=file_path.name,
            source="local",
        )
        total_added += added
        synced_names.append(file_path.name)

    return SyncResponse(
        status="ok",
        files_synced=len(files),
        chunks_added=total_added,
        files=synced_names,
    )


@router.post("/upload-and-index")
async def upload_and_index(file: UploadFile = File(...)):
    """
    Upload a file directly via multipart/form-data and index it immediately.
    Useful for one-off document ingestion without Drive.
    """
    allowed_exts = {".pdf", ".docx", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    dest = config.UPLOAD_DIR / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    doc_id = hashlib.sha256(content).hexdigest()[:16]
    added = _ingest_file(
        file_path=str(dest),
        doc_id=doc_id,
        file_name=file.filename,
        source="upload",
    )
    return {"status": "indexed", "file_name": file.filename, "chunks_added": added, "doc_id": doc_id}


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """
    RAG Q&A endpoint.
    Embeds the query, retrieves top-k chunks, passes to LLM, returns answer.
    """
    # 1. Embed query
    query_emb = embed_query(body.query)

    # 2. Retrieve
    chunks = vector_store.search(
        query_emb,
        top_k=body.top_k,
        filter_source=body.filter_source,
        filter_file_name=body.filter_file_name,
    )

    if not chunks:
        return AskResponse(
            answer="No relevant documents found. Please sync documents first.",
            sources=[],
            chunks_used=0,
            query=body.query,
        )

    # 3. Generate
    answer = generate_answer(body.query, chunks)

    # 4. Deduplicate sources
    sources = list(dict.fromkeys(c["file_name"] for c in chunks))

    return AskResponse(
        answer=answer,
        sources=sources,
        chunks_used=len(chunks),
        query=body.query,
    )
