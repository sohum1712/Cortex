"""
wsgi.py — Gunicorn / WSGI entry point.

Usage:
    gunicorn wsgi:application
    gunicorn -c gunicorn.conf.py wsgi:application

The module-level code in app.py (load_env, validate_groq_key) runs once
at worker startup, which is exactly what we want for a production server.
"""

from app import app as application  # noqa: F401  (gunicorn looks for 'application')

__all__ = ["application"]
