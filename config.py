import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss_index"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

# ── Google Drive ───────────────────────────────────────────────────────────────
GDRIVE_CREDENTIALS_FILE = os.getenv("GDRIVE_CREDENTIALS_FILE", "credentials.json")
GDRIVE_TOKEN_FILE = os.getenv("GDRIVE_TOKEN_FILE", "token.json")
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID", None)   # None = fetch all accessible files

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 output dim; auto-overridden at runtime

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))          # tokens / words
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── FAISS ──────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = str(FAISS_DIR / "index.faiss")
FAISS_META_PATH  = str(FAISS_DIR / "metadata.json")
TOP_K = int(os.getenv("TOP_K", "5"))

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")         # "openai" | "anthropic"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
