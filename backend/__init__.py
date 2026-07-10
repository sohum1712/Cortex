"""
backend — Flask application factory and route registration.

Usage
-----
    from backend import create_app
    app = create_app()
"""

from backend.factory import create_app

__all__ = ["create_app"]
