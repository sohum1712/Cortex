"""
backend/services.py — Application-level singletons.

Design
------
- ConversationManager is cheap to construct and has NO dependency on Ollama.
  It is initialised eagerly on first request to ANY route.
- LLMRAGHandler is expensive (loads ChatOllama + VectorStore) and IS
  dependent on Ollama being reachable. It is only initialised when a route
  that actually calls the LLM requests it (chat, upload, rebuild, url).
- Routes that only manage conversation state (reset, clear) use get_conv()
  which never touches Ollama.
"""

import logging

from core.conversation import ConversationManager
from core.llm_rag import LLMRAGHandler
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_llm:  LLMRAGHandler    | None = None
_conv: ConversationManager | None = None


# ---------------------------------------------------------------------------
# Conversation singleton — no Ollama dependency
# ---------------------------------------------------------------------------

def get_conv() -> ConversationManager:
    """
    Return the shared ConversationManager.

    Safe to call even when Ollama is not running — creates a new instance
    if one does not already exist.
    """
    global _conv
    if _conv is None:
        _conv = ConversationManager(state_file=str(settings.CONV_STATE_FILE))
        logger.info("ConversationManager initialised.")
    return _conv


# ---------------------------------------------------------------------------
# LLM singleton — requires Ollama
# ---------------------------------------------------------------------------

def get_llm() -> LLMRAGHandler:
    """
    Return the shared LLMRAGHandler, creating it on first call.

    On creation:
      1. Restores saved conversation history from disk (if any).
      2. Auto-indexes any PDFs already present in UPLOAD_DIR.

    Only call this from routes that actually use the LLM
    (chat, upload, rebuild, url).
    """
    global _llm

    if _llm is None:
        logger.info("Initialising LLMRAGHandler (model=%s)…", settings.OLLAMA_MODEL)

        # Ensure conv singleton exists so we can restore history into the LLM.
        conv = get_conv()

        _llm = LLMRAGHandler(
            model=settings.OLLAMA_MODEL,
            index_path=settings.INDEX_DIR,
        )

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
    The ConversationManager singleton is kept alive (it is cheap and stateless
    apart from the file path it writes to).
    """
    global _llm
    _llm = None
    logger.info("LLM singleton reset — will re-initialise on next request.")


def llm_is_alive() -> bool:
    """Return True if the LLM singleton has already been initialised."""
    return _llm is not None


def active_files() -> list[str]:
    """Return a sorted list of uploaded PDF filenames."""
    return sorted(p.name for p in settings.UPLOAD_DIR.glob("*.pdf"))
