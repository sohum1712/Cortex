"""
config.py — Centralised application configuration.

All paths and tuneable constants live here. Environment variables
override defaults when set (e.g. in a .env file or injected by Docker).

Usage
-----
    from config import settings
    print(settings.UPLOAD_DIR)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root  (this file lives at the repo root)
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()


class Settings:
    """Application-wide settings resolved at import time."""

    # ── Paths ────────────────────────────────────────────────────────────────

    FRONTEND_DIR    : Path = ROOT_DIR / "frontend"
    TEMPLATE_DIR    : Path = FRONTEND_DIR / "templates"
    STATIC_DIR      : Path = FRONTEND_DIR / "static"

    DATA_DIR        : Path = ROOT_DIR / "data"
    UPLOAD_DIR      : Path = DATA_DIR / "uploads"
    INDEX_DIR       : Path = DATA_DIR / "index"
    CONV_STATE_FILE : Path = DATA_DIR / "conversation.json"

    # ── Ollama / LLM ─────────────────────────────────────────────────────────

    OLLAMA_MODEL       : str = os.getenv("OLLAMA_MODEL",       "llama3.2:3b")
    OLLAMA_EMBED_MODEL : str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # ── Flask ────────────────────────────────────────────────────────────────

    SECRET_KEY  : str  = os.getenv("SECRET_KEY",  "change-me-in-production")
    FLASK_HOST  : str  = os.getenv("FLASK_HOST",  "0.0.0.0")
    FLASK_PORT  : int  = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG : bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # ── Upload limits ────────────────────────────────────────────────────────

    MAX_UPLOAD_MB    : int = int(os.getenv("MAX_UPLOAD_MB", "64"))
    MAX_UPLOAD_BYTES : int = MAX_UPLOAD_MB * 1024 * 1024

    # ── HTTP client identity ─────────────────────────────────────────────────
    # Suppresses LangChain's USER_AGENT warning and identifies outbound
    # requests when using the URL-indexing feature.

    USER_AGENT : str = os.getenv("USER_AGENT", "Cortex-RAG/1.0")


settings = Settings()

# Apply immediately so LangChain / requests picks it up before any
# community loader imports run.
os.environ.setdefault("USER_AGENT", settings.USER_AGENT)
