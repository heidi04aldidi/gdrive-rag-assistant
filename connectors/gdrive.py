"""
connectors/gdrive.py
────────────────────
Google Drive connector — supports both OAuth (desktop) and service-account auth.

Fetches:
  • application/pdf
  • application/vnd.google-apps.document  (exported as docx)
  • text/plain

Saves each file to DATA_DIR/uploads/ and returns a list of DriveFile objects.
"""

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle

import config

log = logging.getLogger(__name__)

# MIME types we handle
SUPPORTED_MIME = {
    "application/pdf": ".pdf",
    "application/vnd.google-apps.document": ".docx",   # exported
    "text/plain": ".txt",
}

EXPORT_MIME = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
}


@dataclass
class DriveFile:
    file_id: str
    file_name: str
    mime_type: str
    local_path: str
    metadata: dict = field(default_factory=dict)


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _oauth_credentials() -> Credentials:
    """Interactive OAuth2 flow (stores token in token.json)."""
    creds: Optional[Credentials] = None
    token_path = config.GDRIVE_TOKEN_FILE

    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                config.GDRIVE_CREDENTIALS_FILE, config.GDRIVE_SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return creds


def _service_account_credentials() -> service_account.Credentials:
    """Service-account credentials (non-interactive, for servers)."""
    return service_account.Credentials.from_service_account_file(
        config.GDRIVE_CREDENTIALS_FILE,
        scopes=config.GDRIVE_SCOPES,
    )


def _build_service():
    """Return an authenticated Drive v3 service."""
    creds_file = config.GDRIVE_CREDENTIALS_FILE
    if not os.path.exists(creds_file):
        raise FileNotFoundError(
            f"Google credentials file not found: {creds_file}\n"
            "Set GDRIVE_CREDENTIALS_FILE in .env and download it from "
            "https://console.cloud.google.com/apis/credentials"
        )

    # Detect credential type by peeking at the JSON
    import json
    with open(creds_file) as f:
        info = json.load(f)

    if info.get("type") == "service_account":
        creds = _service_account_credentials()
    else:
        creds = _oauth_credentials()

    return build("drive", "v3", credentials=creds)


# ── File listing ───────────────────────────────────────────────────────────────

def _list_files(service, folder_id: Optional[str] = None) -> List[dict]:
    """Return all supported files from Drive (or a specific folder)."""
    mime_filter = " or ".join(
        f"mimeType='{m}'" for m in SUPPORTED_MIME
    )
    query = f"({mime_filter}) and trashed=false"
    if folder_id:
        query = f"'{folder_id}' in parents and {query}"

    files: List[dict] = []
    page_token = None

    while True:
        resp = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                pageToken=page_token,
                pageSize=100,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    log.info("Found %d supported files in Drive.", len(files))
    return files


# ── Download helpers ───────────────────────────────────────────────────────────

def _download_file(service, file_id: str, dest_path: str, mime_type: str) -> None:
    """Download (or export) a file to dest_path."""
    if mime_type in EXPORT_MIME:
        export_mime = EXPORT_MIME[mime_type]
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=file_id)

    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


# ── Public API ─────────────────────────────────────────────────────────────────

def sync_drive(folder_id: Optional[str] = None) -> List[DriveFile]:
    """
    Main entry-point called by POST /sync-drive.
    Returns a list of DriveFile objects for every downloaded file.
    """
    service = _build_service()
    raw_files = _list_files(service, folder_id or config.GDRIVE_FOLDER_ID)

    drive_files: List[DriveFile] = []

    for f in raw_files:
        file_id   = f["id"]
        mime_type = f["mimeType"]
        ext       = SUPPORTED_MIME.get(mime_type, ".bin")
        # Use original name but ensure correct extension
        safe_name = "".join(
            c if (c.isalnum() or c in " ._-") else "_" for c in f["name"]
        )
        if not safe_name.endswith(ext):
            safe_name = Path(safe_name).stem + ext

        dest = str(config.UPLOAD_DIR / safe_name)

        try:
            log.info("Downloading %s (%s) ...", f["name"], mime_type)
            _download_file(service, file_id, dest, mime_type)

            drive_files.append(
                DriveFile(
                    file_id=file_id,
                    file_name=f["name"],
                    mime_type=mime_type,
                    local_path=dest,
                    metadata={
                        "source": "gdrive",
                        "drive_file_id": file_id,
                        "modified_time": f.get("modifiedTime", ""),
                    },
                )
            )
        except Exception as exc:
            log.error("Failed to download %s: %s", f["name"], exc)

    log.info("Synced %d / %d files successfully.", len(drive_files), len(raw_files))
    return drive_files
