"""
backend/routes.py — All Flask route handlers.

Ollama dependency map
---------------------
  Requires Ollama  : /api/chat, /api/upload, /api/rebuild, /api/url
  Ollama-free      : /, /api/files, /api/reset, /api/clear

Endpoints
---------
GET  /             Serve the Cortex UI
GET  /api/files    List uploaded PDFs
POST /api/chat     Stream a RAG response (SSE)
POST /api/upload   Upload + index one or more PDFs
POST /api/rebuild  Re-index all PDFs from scratch
POST /api/clear    Delete index, uploads, and conversation
POST /api/reset    Reset conversation history only  (no Ollama needed)
POST /api/url      Index website URLs
"""

import json
import logging
import shutil

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context

from backend.services import active_files, get_conv, get_llm, llm_is_alive, reset_singletons
from config import ENV_FILE, load_env, settings

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@bp.route("/")
def index():
    return render_template("index.html", active_files=active_files())


# ---------------------------------------------------------------------------
# Knowledge-base info  (no Ollama)
# ---------------------------------------------------------------------------

@bp.route("/api/files")
def api_files():
    return jsonify({"files": active_files()})


# ---------------------------------------------------------------------------
# Health check — tests Groq key without touching the knowledge base
# ---------------------------------------------------------------------------

@bp.route("/api/health")
def api_health():
    """
    Quick liveness + Groq key check.
    Returns 200 if the key works, 503 with an error message if not.
    Open http://localhost:5000/api/health in your browser to test.
    """
    load_env(override=True)
    key = settings.GROQ_API_KEY
    if not key:
        return jsonify({
            "status": "error",
            "groq":   "GROQ_API_KEY not set",
            "env_file": str(ENV_FILE),
            "fix":    f"Copy .env.example to {ENV_FILE} and set GROQ_API_KEY.",
        }), 503

    try:
        from langchain_groq import ChatGroq
        from langchain_core.output_parsers import StrOutputParser
        llm    = ChatGroq(model=settings.GROQ_MODEL, api_key=key, temperature=0, max_tokens=1)
        result = (llm | StrOutputParser()).invoke("hi")
        return jsonify({
            "status":     "ok",
            "groq":       "connected",
            "model":      settings.GROQ_MODEL,
            "env_file":   str(ENV_FILE),
            "key_prefix": key[:8] + "...",
            "response":   result,
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "groq":   str(exc) or repr(exc),
            "env_file": str(ENV_FILE),
            "fix":    f"Check GROQ_API_KEY in {ENV_FILE} or create a new key at https://console.groq.com",
        }), 503


# ---------------------------------------------------------------------------
# Chat — SSE streaming  (requires Ollama)
# ---------------------------------------------------------------------------

@bp.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Stream a RAG response using Server-Sent Events.

    Request body  : { "message": "<question>" }
    SSE payloads  :
        {"token": "..."}              — partial token
        {"done": true, "full": "..."} — stream complete
        {"error": "..."}              — exception

    The LLM singleton (and any first-time PDF auto-indexing) is initialised
    BEFORE we enter the generator so that a slow first-time Ollama embedding
    run does not block the SSE stream from opening on the client side.
    """
    body     = request.get_json(silent=True) or {}
    user_msg = (body.get("message") or "").strip()

    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    # Initialise (and auto-index) outside the generator so any slow startup
    # work (e.g. first-time Ollama embedding of existing PDFs) happens before
    # the SSE stream is opened.  Errors here return a proper JSON 500.
    try:
        llm  = get_llm()
        conv = get_conv()
    except Exception as exc:
        logger.exception("LLM init failed")
        return jsonify({"error": str(exc)}), 500

    def _generate():
        try:
            full = ""
            for token in llm.generate_response_stream(user_msg):
                full += token
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            conv.save(llm.get_history())

            # Serialize source documents for the citations panel
            sources = []
            for doc in getattr(llm, "_last_sources", []):
                meta  = doc.metadata or {}
                score = meta.get("_score", 0)
                # Convert FAISS L2 distance to an approximate 0-1 similarity
                # Lower L2 distance = more similar; clamp to [0, 1]
                similarity = max(0.0, 1.0 - min(float(score), 2.0) / 2.0)
                sources.append({
                    "source":  meta.get("source", "Unknown"),
                    "content": doc.page_content.strip()[:300],
                    "score":   round(similarity, 3),
                })

            yield f"data: {json.dumps({'done': True, 'full': full, 'sources': sources})}\n\n"
        except Exception as exc:
            logger.exception("Error during chat stream")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


# ---------------------------------------------------------------------------
# PDF upload  (requires Ollama — triggers embedding)
# ---------------------------------------------------------------------------

@bp.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload and index one or more PDF files."""
    if "files" not in request.files:
        return jsonify({"error": "No files part in request"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    llm     = get_llm()
    results = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"name": f.filename, "status": "skipped — not a PDF"})
            continue

        save_path = settings.UPLOAD_DIR / f.filename
        f.save(str(save_path))

        try:
            llm.add_pdf_to_context(save_path)
            results.append({"name": f.filename, "status": "indexed"})
            logger.info("Indexed: %s", f.filename)
        except Exception as exc:
            logger.exception("Failed to index %s", f.filename)
            results.append({"name": f.filename, "status": f"error: {exc}"})

    return jsonify({"results": results, "files": active_files()})


# ---------------------------------------------------------------------------
# Knowledge-base management
# ---------------------------------------------------------------------------

@bp.route("/api/rebuild", methods=["POST"])
def api_rebuild():
    """Tear down the LLM singleton and re-index all PDFs from scratch."""
    reset_singletons()
    get_llm()   # re-creates and auto-indexes
    return jsonify({"status": "rebuilt", "files": active_files()})


@bp.route("/api/clear", methods=["POST"])
def api_clear():
    """
    Delete all uploaded PDFs, the FAISS index, and conversation history.
    Does NOT require Ollama.
    """
    # Remove uploaded PDFs
    for pdf in settings.UPLOAD_DIR.glob("*.pdf"):
        pdf.unlink(missing_ok=True)

    # Remove FAISS index directory
    if settings.INDEX_DIR.exists():
        shutil.rmtree(settings.INDEX_DIR)
        settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Clear conversation state and drop the LLM singleton
    get_conv().clear()
    reset_singletons()

    return jsonify({"status": "cleared"})


@bp.route("/api/reset", methods=["POST"])
def api_reset():
    """
    Reset conversation history only — knowledge base remains intact.
    Does NOT require Ollama.
    """
    get_conv().clear()

    # If the LLM singleton is already alive reset its in-memory history so
    # the next question starts a clean slate.  We deliberately do NOT call
    # get_llm() here — that would spin up Ollama unnecessarily.
    if llm_is_alive():
        get_llm().reset()

    return jsonify({"status": "reset"})


# ---------------------------------------------------------------------------
# URL indexing  (requires Ollama — triggers embedding)
# ---------------------------------------------------------------------------

@bp.route("/api/url", methods=["POST"])
def api_url():
    """Index one or more website URLs into the knowledge base."""
    body = request.get_json(silent=True) or {}
    urls = [u.strip() for u in (body.get("urls") or []) if u.strip()]

    if not urls:
        return jsonify({"error": "urls list is required"}), 400

    llm = get_llm()
    try:
        llm.vector_store.index_websites(urls)
        logger.info("Indexed %d URL(s).", len(urls))
        return jsonify({"status": "indexed", "count": len(urls)})
    except Exception as exc:
        logger.exception("URL indexing failed")
        return jsonify({"error": str(exc)}), 500
