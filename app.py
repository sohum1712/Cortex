"""
app.py — Application entry point.

Boot order (MUST stay in this order):
  1. load_dotenv()        — populate os.environ from .env
  2. import config        — Settings reads os.environ via @property (always fresh)
  3. validate_groq_key()  — test the key NOW, fail loudly if bad
  4. create_app()         — wire Flask
  5. app.run()            — serve
"""

# ── 1. Load .env before anything else ────────────────────────────────────────
import logging

from config import ENV_FILE, load_env, settings

load_env(override=True)   # always reads ROOT_DIR/.env — not the process cwd

# ── 2. Project imports ────────────────────────────────────────────────────────
from core.llm_rag import validate_groq_key
from backend.factory import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 3. Validate Groq key — warn if bad, but keep the server running ───────────
logger.info("Validating Groq API key...")
try:
    validate_groq_key()
except RuntimeError as exc:
    # Warn loudly but do NOT exit — the user can fix the key in .env and
    # the server will pick it up on the next chat request without restarting.
    print("\n" + "=" * 60)
    print("  WARNING — Groq API key problem (server will still start)")
    print("=" * 60)
    print(str(exc))
    print(f"  .env path: {ENV_FILE}")
    print("  Update GROQ_API_KEY in that file — no restart needed.")
    print("=" * 60 + "\n")

# ── 4. Create Flask app ───────────────────────────────────────────────────────
app = create_app()

# ── 5. Serve ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Cortex on http://%s:%s", settings.FLASK_HOST, settings.FLASK_PORT)
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG,
        threaded=True,
        use_reloader=False,
    )
