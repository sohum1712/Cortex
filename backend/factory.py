"""
backend/factory.py — Flask application factory.

Keeps app creation isolated from the entry point so the app
can be imported in tests without side effects.
"""

import logging
from pathlib import Path

from flask import Flask

from config import settings


def create_app() -> Flask:
    """Create, configure, and return the Flask application."""

    app = Flask(
        __name__,
        template_folder=str(settings.TEMPLATE_DIR),
        static_folder=str(settings.STATIC_DIR),
    )

    # ── Core config ─────────────────────────────────────────────────────────
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_UPLOAD_BYTES
    app.config["SECRET_KEY"]         = settings.SECRET_KEY

    # ── Logging ──────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Ensure runtime directories exist ────────────────────────────────────
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── Register blueprints ──────────────────────────────────────────────────
    from backend.routes import bp
    app.register_blueprint(bp)

    return app
