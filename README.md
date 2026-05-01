# RAG over Google Drive

A production-ready Retrieval-Augmented Generation (RAG) system that connects to Google Drive, processes your documents, and answers questions grounded in your own content.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI  (:8000)                     │
│                                                             │
│  POST /sync-drive   POST /ask   GET /status   GET /docs     │
└────────────┬────────────────┬────────────────────────────────┘
             │                │
     ┌───────▼──────┐  ┌──────▼──────────────────────────────┐
     │  connectors/ │  │              Query Pipeline          │
     │  gdrive.py   │  │                                      │
     │              │  │  1. embed_query()  →  (1×384)       │
     │  OAuth / SA  │  │  2. vector_store.search()           │
     └───────┬──────┘  │  3. generate_answer()  →  LLM       │
             │         └──────────────────────────────────────┘
     ┌───────▼────────────────────┐
     │       Ingestion Pipeline   │
     │                            │
     │  processing/parser.py      │  ← PDF / DOCX / TXT
     │  processing/chunker.py     │  ← sliding-window chunks
     │  embedding/embedder.py     │  ← SentenceTransformers
     │  search/vector_store.py    │  ← FAISS + JSON metadata
     └────────────────────────────┘
```

### Directory Layout

```
rag-gdrive/
├── connectors/
│   └── gdrive.py          # Google Drive OAuth + service-account auth
├── processing/
│   ├── parser.py          # PDF / DOCX / TXT text extraction
│   └── chunker.py         # Sliding-window sentence-aware chunker
├── embedding/
│   └── embedder.py        # SentenceTransformers (batch, normalised)
├── search/
│   └── vector_store.py    # FAISS flat-IP index + JSON metadata sidecar
├── api/
│   ├── routes.py          # All FastAPI endpoints
│   └── llm.py             # OpenAI / Anthropic answer generation
├── data/
│   ├── uploads/           # Downloaded / uploaded files
│   └── faiss_index/       # Persisted FAISS index + metadata
├── main.py                # App factory + startup
├── config.py              # Centralised settings (env-driven)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── test_queries.py        # Sample queries script
└── .env.example
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd rag-gdrive

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GDRIVE_CREDENTIALS_FILE` | Path to Google credentials JSON |
| `GDRIVE_FOLDER_ID` | (Optional) Specific Drive folder to sync |

### 3. Set up Google Drive credentials

**Option A — OAuth2 (recommended for personal use)**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **Google Drive API**
3. Create **OAuth 2.0 Client ID** (Desktop application)
4. Download as `credentials.json` → place in project root
5. First `/sync-drive` call opens a browser for consent

**Option B — Service Account (recommended for servers)**

1. Create a **Service Account** in Cloud Console
2. Download the JSON key as `credentials.json`
3. Share the target Drive folder with the service account email

### 4. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: **http://localhost:8000/docs**

---

## API Reference

### `POST /sync-drive`

Pull documents from Google Drive and index them.

```bash
curl -X POST http://localhost:8000/sync-drive \
  -H "Content-Type: application/json" \
  -d '{"folder_id": "1abc..."}'
```

```json
{
  "status": "ok",
  "files_synced": 3,
  "chunks_added": 142,
  "files": ["policy.pdf", "handbook.docx", "notes.txt"]
}
```

---

### `POST /ask`

Ask a question — returns an LLM answer grounded in your documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our refund policy?"}'
```

```json
{
  "answer": "According to policy.pdf, customers are entitled to a full refund within 30 days of purchase provided the item is unused and in original packaging. Refund requests must be submitted through the support portal.",
  "sources": ["policy.pdf"],
  "chunks_used": 4,
  "query": "What is our refund policy?"
}
```

Optional parameters:

| Field | Type | Default | Description |
|---|---|---|---|
| `top_k` | int | 5 | Number of chunks to retrieve |
| `filter_source` | string | — | Filter by source (`gdrive`, `local`, `upload`) |
| `filter_file_name` | string | — | Filter by exact file name |

---

### `POST /sync-local`

Index files already placed in `data/uploads/` (for testing without Drive).

```bash
curl -X POST http://localhost:8000/sync-local
```

---

### `POST /upload-and-index`

Upload and index a file directly.

```bash
curl -X POST http://localhost:8000/upload-and-index \
  -F "file=@/path/to/document.pdf"
```

---

### `GET /status`

```json
{
  "total_vectors": 874,
  "total_chunks": 874,
  "total_documents": 6,
  "embedding_dim": 384
}
```

---

### `GET /documents`

List all indexed documents with chunk counts.

---

### `DELETE /documents/{doc_id}`

Remove a document and all its chunks from the index.

---

## Sample Test Queries

```bash
python test_queries.py
```

Example questions to test with company policy documents:

```
"What is our refund policy?"
"What are the compliance requirements for data handling?"
"How many days of annual leave do employees receive?"
"What is the process for reporting a security incident?"
"What are the approved vendors for cloud services?"
```

---

## Docker Deployment

```bash
# Build and start
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` | Fast, 384-dim, no API key needed |
| **Vector store** | FAISS flat-IP | Zero-dependency, disk-persistent, cosine similarity via L2-normalised vectors |
| **Chunking** | Sliding-window (500 words, 50 overlap) | Preserves sentence context across chunk boundaries |
| **LLM** | OpenAI / Anthropic (pluggable) | Switch provider via single env var |
| **Auth** | OAuth2 + service account | Auto-detected from credential file type |

---
