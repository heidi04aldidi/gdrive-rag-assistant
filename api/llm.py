"""
api/llm.py
───────────
LLM answer generation layer.

Supports two providers (set LLM_PROVIDER in .env):
  • "openai"    — uses openai>=1.0 SDK
  • "anthropic" — uses anthropic SDK

The prompt instructs the model to answer strictly from the provided
context chunks and to cite sources.
"""

import logging
from typing import List, Dict, Any

import config

log = logging.getLogger(__name__)

# ── Prompt builder ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided document context.

Rules:
1. Answer based strictly on the context given — do not add external knowledge.
2. If the answer is not in the context, say "I could not find this information in the available documents."
3. Be concise and accurate.
4. Always mention which document(s) the information comes from.
"""


def _build_user_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['file_name']}, Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return f"""Context from documents:

{context}

---

Question: {query}

Please answer the question based on the context above."""


# ── Provider implementations ───────────────────────────────────────────────────

def _answer_openai(query: str, chunks: List[Dict[str, Any]]) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    user_prompt = _build_user_prompt(query, chunks)

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _answer_anthropic(query: str, chunks: List[Dict[str, Any]]) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")

    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment.")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    user_prompt = _build_user_prompt(query, chunks)

    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generate an LLM answer grounded in retrieved chunks.

    Parameters
    ----------
    query  : the user's question
    chunks : list of chunk dicts returned by vector_store.search()
    """
    if not chunks:
        return "No relevant documents were found to answer your question."

    provider = config.LLM_PROVIDER.lower()
    log.info("Generating answer via provider='%s' for query: %s", provider, query[:80])

    if provider == "openai":
        return _answer_openai(query, chunks)
    elif provider == "anthropic":
        return _answer_anthropic(query, chunks)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Choose 'openai' or 'anthropic'.")
