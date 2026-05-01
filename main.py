"""
main.py
────────
FastAPI application factory and entry point.

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or:
    python main.py
"""

import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG over Google Drive",
    description=(
        "A Retrieval-Augmented Generation system that connects to Google Drive, "
        "processes documents, and answers questions grounded in your documents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")


@app.on_event("startup")
async def startup_event():
    log.info("─" * 60)
    log.info("RAG over Google Drive  |  v1.0.0")
    log.info("LLM provider : %s", config.LLM_PROVIDER)
    log.info("Embedding    : %s", config.EMBEDDING_MODEL)
    log.info("Docs at      : http://%s:%s/docs", config.API_HOST, config.API_PORT)
    log.info("─" * 60)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level="info",
    )
