"""
config.py — Centralised application configuration.

Settings are read from environment variables on every attribute access
via properties, so the values are always current regardless of when
load_env() was called relative to this module being imported.

Paths are computed once from ROOT_DIR and never change.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.resolve()
ENV_FILE = ROOT_DIR / ".env"


def _sanitize_env_value(value: str) -> str:
    """Strip whitespace, quotes, and BOM from env values."""
    if not value:
        return ""
    value = value.strip().strip("\ufeff")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def load_env(*, override: bool = True) -> bool:
    """
    Load environment variables from the project-root .env file.

    Always uses ROOT_DIR / '.env' so the key is found even when the
    process was started from a different working directory (IDE, flask run, etc.).

    Returns True when the .env file exists and was loaded.
    """
    if not ENV_FILE.is_file():
        logger.warning(
            "No .env file at %s — copy .env.example to .env and set GROQ_API_KEY.",
            ENV_FILE,
        )
        return False

    load_dotenv(ENV_FILE, override=override)
    return True


class Settings:
    """
    All settings exposed as plain attributes.
    Paths are fixed at import time (they don't come from env vars).
    All other settings are read fresh from os.environ on each access
    via __getattr__ so load_dotenv() timing never matters.
    """

    # ── Fixed paths (not env-var-driven) ────────────────────────────────────
    FRONTEND_DIR    : Path = ROOT_DIR / "frontend"
    TEMPLATE_DIR    : Path = ROOT_DIR / "frontend" / "templates"
    STATIC_DIR      : Path = ROOT_DIR / "frontend" / "static"
    DATA_DIR        : Path = ROOT_DIR / "data"
    UPLOAD_DIR      : Path = ROOT_DIR / "data" / "uploads"
    INDEX_DIR       : Path = ROOT_DIR / "data" / "index"
    CONV_STATE_FILE : Path = ROOT_DIR / "data" / "conversation.json"

    # ── Env-var backed settings — read fresh every time via properties ───────

    @property
    def GROQ_API_KEY(self) -> str:
        return _sanitize_env_value(os.environ.get("GROQ_API_KEY", ""))

    @property
    def GROQ_MODEL(self) -> str:
        return os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip()

    @property
    def EMBED_MODEL(self) -> str:
        return os.environ.get("EMBED_MODEL", "mxbai-embed-large").strip()

    @property
    def CHUNK_SIZE(self) -> int:
        return int(os.environ.get("CHUNK_SIZE", "800"))

    @property
    def CHUNK_OVERLAP(self) -> int:
        return int(os.environ.get("CHUNK_OVERLAP", "100"))

    @property
    def TOP_K(self) -> int:
        return int(os.environ.get("TOP_K", "5"))

    @property
    def SIMILARITY_THRESHOLD(self) -> float:
        return float(os.environ.get("SIMILARITY_THRESHOLD", "0.0"))

    @property
    def CHROMA_COLLECTION_NAME(self) -> str:
        return os.environ.get("CHROMA_COLLECTION_NAME", "cortex").strip()

    @property
    def CHROMA_DISTANCE_METRIC(self) -> str:
        """Chroma hnsw:space value — l2, cosine, or ip. Default l2 (matches old FAISS)."""
        return os.environ.get("CHROMA_DISTANCE_METRIC", "l2").strip()

    @property
    def CHROMA_PERSIST_DIR(self) -> Path | None:
        """
        Override Chroma persist directory. When unset, vector_store uses
        INDEX_DIR / 'chroma' (derived from the index_path passed to VectorStore).
        """
        raw = os.environ.get("CHROMA_PERSIST_DIR", "").strip()
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def SECRET_KEY(self) -> str:
        return os.environ.get("SECRET_KEY", "change-me-in-production")

    @property
    def FLASK_HOST(self) -> str:
        return os.environ.get("FLASK_HOST", "0.0.0.0")

    @property
    def FLASK_PORT(self) -> int:
        return int(os.environ.get("FLASK_PORT", "5000"))

    @property
    def FLASK_DEBUG(self) -> bool:
        return os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    @property
    def MAX_UPLOAD_MB(self) -> int:
        return int(os.environ.get("MAX_UPLOAD_MB", "64"))

    @property
    def MAX_UPLOAD_BYTES(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def USER_AGENT(self) -> str:
        return os.environ.get("USER_AGENT", "Cortex-RAG/1.0")


settings = Settings()

# Load project-root .env on first import so every entry point (app.py, flask run,
# tests, IDE run configs) picks up GROQ_API_KEY even when cwd is not the repo root.
load_env(override=True)
