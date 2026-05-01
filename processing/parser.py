"""
processing/parser.py
─────────────────────
Extracts plain text from PDF, DOCX, and TXT files.

Each parser returns a list of (page_number, text) tuples so that
chunk metadata can reference a source page.
"""

import logging
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)

PageText = Tuple[int, str]  # (page_number, text)


# ── PDF ────────────────────────────────────────────────────────────────────────

def parse_pdf(file_path: str) -> List[PageText]:
    """Extract text page-by-page using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    pages: List[PageText] = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_num, text))
        doc.close()
    except Exception as exc:
        log.error("PDF parse error (%s): %s", file_path, exc)
    return pages


# ── DOCX ───────────────────────────────────────────────────────────────────────

def parse_docx(file_path: str) -> List[PageText]:
    """Extract text from a .docx file (Google Docs exported format)."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    pages: List[PageText] = []
    try:
        doc = Document(file_path)
        # Group paragraphs into virtual "pages" of ~50 paragraphs
        chunk_size = 50
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for i in range(0, max(len(paragraphs), 1), chunk_size):
            group = paragraphs[i : i + chunk_size]
            if group:
                pages.append((i // chunk_size + 1, "\n".join(group)))
    except Exception as exc:
        log.error("DOCX parse error (%s): %s", file_path, exc)
    return pages


# ── TXT ────────────────────────────────────────────────────────────────────────

def parse_txt(file_path: str) -> List[PageText]:
    """Read a plain-text file and treat every 100 lines as a virtual page."""
    pages: List[PageText] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]

        chunk_size = 100
        for i in range(0, max(len(lines), 1), chunk_size):
            group = lines[i : i + chunk_size]
            if group:
                pages.append((i // chunk_size + 1, "\n".join(group)))
    except Exception as exc:
        log.error("TXT parse error (%s): %s", file_path, exc)
    return pages


# ── Router ─────────────────────────────────────────────────────────────────────

def parse_document(file_path: str) -> List[PageText]:
    """
    Dispatch to the correct parser based on file extension.
    Returns [(page_num, text), ...].
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return parse_docx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        log.warning("Unsupported file type: %s — skipping.", ext)
        return []
