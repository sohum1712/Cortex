"""
backend/services.py — Application-level singletons.

Design
------
- ConversationManager is cheap to construct and has NO dependency on Groq
  or Ollama. It is initialised eagerly on first request to ANY route.
- LLMRAGHandler is initialised lazily — only when a route that actually
  calls the LLM requests it (chat, upload, rebuild, url). On creation it
  restores conversation history and auto-indexes any existing PDFs.
- Routes that only manage conversation state (reset, clear) use get_conv()
  which never touches the LLM or Ollama.
"""

import logging

from core.conversation import ConversationManager
from core.llm_rag import LLMRAGHandler
from config import load_env, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_llm:         LLMRAGHandler      | None = None
_conv:        ConversationManager | None = None
_llm_api_key: str                        = ""   # key the current singleton was built with


# ---------------------------------------------------------------------------
# Conversation singleton — no LLM dependency
# ---------------------------------------------------------------------------

def get_conv() -> ConversationManager:
    """
    Return the shared ConversationManager.

    Safe to call even when Groq/Ollama are not reachable — creates a new
    instance if one does not already exist.
    """
    global _conv
    if _conv is None:
        _conv = ConversationManager(state_file=str(settings.CONV_STATE_FILE))
        logger.info("ConversationManager initialised.")
    return _conv


# ---------------------------------------------------------------------------
# LLM singleton — requires Groq API key + Ollama for embeddings
# ---------------------------------------------------------------------------

def get_llm() -> LLMRAGHandler:
    """
    Return the shared LLMRAGHandler, creating it on first call.

    On every call, reloads .env so that a key change made while the server
    is running is picked up immediately. If the GROQ_API_KEY has changed
    since the singleton was last built, the old instance is discarded and
    a fresh one is created (history is preserved across the rebuild).

    On creation:
      1. Restores saved conversation history from disk (if any).
      2. Auto-indexes any PDFs already present in UPLOAD_DIR.

    Only call this from routes that actually use the LLM
    (chat, upload, rebuild, url).
    """
    global _llm, _llm_api_key

    # Always re-read .env so an in-place edit is picked up without restart.
    load_env(override=True)
    current_key = settings.GROQ_API_KEY

    # If the key changed, tear down the cached chain so it rebuilds with the
    # new key. We keep the LLMRAGHandler instance alive (preserving history
    # and the vector store) — only the internal chain is invalidated.
    if _llm is not None and current_key != _llm_api_key:
        logger.info("GROQ_API_KEY changed — invalidating cached chain.")
        _llm.reset()          # clears _chain / _active_key inside LLMRAGHandler
        _llm_api_key = current_key

    if _llm is None:
        logger.info(
            "Initialising LLMRAGHandler (groq_model=%s, embed_model=%s)…",
            settings.GROQ_MODEL,
            settings.EMBED_MODEL,
        )

        # Ensure conv singleton exists so we can restore history into the LLM.
        conv = get_conv()

        _llm = LLMRAGHandler(
            embedding_model=settings.EMBED_MODEL,
            index_path=settings.INDEX_DIR,
            retrieval_k=settings.TOP_K,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
        )
        _llm_api_key = current_key

        # Restore previous conversation
        saved = conv.load()
        if saved:
            _llm.history = saved
            logger.info("Restored %d conversation messages.", len(saved))

        # Auto-index PDFs that were uploaded in a previous session
        existing = sorted(settings.UPLOAD_DIR.glob("*.pdf"))
        if existing:
            logger.info("Auto-indexing %d existing PDF(s)…", len(existing))
            for pdf in existing:
                _llm.add_pdf_to_context(pdf)

    return _llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset_singletons() -> None:
    """
    Tear down the LLM singleton so it is rebuilt on the next LLM request.
    The ConversationManager singleton is kept alive (cheap and stateless
    apart from the file path it writes to).
    """
    global _llm, _llm_api_key
    _llm         = None
    _llm_api_key = ""
    logger.info("LLM singleton reset — will re-initialise on next request.")


def llm_is_alive() -> bool:
    """Return True if the LLM singleton has already been initialised."""
    return _llm is not None


def active_files() -> list[str]:
    """Return a sorted list of uploaded PDF filenames."""
    return sorted(p.name for p in settings.UPLOAD_DIR.glob("*.pdf"))
