#!/usr/bin/env python3
"""
test_queries.py
───────────────
Run sample queries against a locally running RAG API.

Usage:
    # Make sure the API is running first:
    #   uvicorn main:app --reload
    python test_queries.py
"""

import json
import sys
import requests

BASE_URL = "http://localhost:8000"


def pretty(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def check_health():
    print("\n── Health Check ───────────────────────────────────────────────")
    r = requests.get(f"{BASE_URL}/health")
    pretty(r.json())


def check_status():
    print("\n── Vector Store Status ────────────────────────────────────────")
    r = requests.get(f"{BASE_URL}/status")
    pretty(r.json())


def list_documents():
    print("\n── Indexed Documents ──────────────────────────────────────────")
    r = requests.get(f"{BASE_URL}/documents")
    docs = r.json()
    if not docs:
        print("  (none indexed yet — run /sync-drive or /sync-local first)")
    else:
        for doc in docs:
            print(f"  • {doc['file_name']}  [{doc['chunk_count']} chunks, source={doc['source']}]")


def sync_local():
    print("\n── Syncing local uploads/ ─────────────────────────────────────")
    r = requests.post(f"{BASE_URL}/sync-local")
    pretty(r.json())


def ask(query: str, top_k: int = 5):
    print(f"\n── Query: {query!r} ─────────────────────────────────────────")
    payload = {"query": query, "top_k": top_k}
    r = requests.post(f"{BASE_URL}/ask", json=payload)
    data = r.json()
    print(f"Answer    : {data['answer']}")
    print(f"Sources   : {data['sources']}")
    print(f"Chunks    : {data['chunks_used']}")


if __name__ == "__main__":
    try:
        check_health()
        check_status()
        list_documents()

        # Sync local test documents first
        sync_local()
        check_status()
        list_documents()

        # ── Sample Queries ────────────────────────────────────────────────────
        sample_questions = [
            "What is our refund policy?",
            "What are the company's compliance requirements?",
            "How many days of annual leave do employees get?",
            "What is the process for reporting a security incident?",
            "Summarise the key points of the document.",
        ]

        for q in sample_questions:
            ask(q)

    except requests.exceptions.ConnectionError:
        print(
            "\nERROR: Could not connect to the API.\n"
            "Make sure the server is running:\n"
            "  uvicorn main:app --reload"
        )
        sys.exit(1)
