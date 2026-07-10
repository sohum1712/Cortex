"""
app.py — Application entry point.

Keeps the entry point as thin as possible. All wiring lives in
backend/factory.py and backend/routes.py.

Run
---
    python app.py
    # or via gunicorn in production:
    gunicorn "app:app" --bind 0.0.0.0:5000 --workers 1 --threads 4
"""

from backend.factory import create_app
from config import settings

app = create_app()

if __name__ == "__main__":
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG,
        threaded=True,
    )
