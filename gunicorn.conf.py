"""
gunicorn.conf.py — Production Gunicorn configuration for Cortex.

Apply with:  gunicorn -c gunicorn.conf.py wsgi:application
"""

import multiprocessing
import os

# ── Workers ───────────────────────────────────────────────────────────────────
# Cortex is I/O-bound (Groq API + Ollama + file I/O).
# gevent workers give better concurrency than sync workers for SSE streams.
# Fall back to sync if gevent is not installed.
try:
    import gevent  # noqa: F401
    worker_class = "gevent"
    worker_connections = 100
except ImportError:
    worker_class = "sync"

# 2-4 workers per CPU is the standard starting point.
# For SSE-heavy workloads keep it lower to avoid memory pressure.
workers = int(os.environ.get("GUNICORN_WORKERS", max(2, multiprocessing.cpu_count())))
threads = int(os.environ.get("GUNICORN_THREADS", 2))

# ── Binding ───────────────────────────────────────────────────────────────────
# Render (and most PaaS) injects $PORT at runtime — honour it first.
host = os.environ.get("FLASK_HOST", "0.0.0.0")
port = os.environ.get("PORT") or os.environ.get("FLASK_PORT", "5000")
bind = f"{host}:{port}"

# ── Timeouts ──────────────────────────────────────────────────────────────────
# SSE streams can be long-lived; keep-alive and graceful timeouts must be generous.
timeout          = int(os.environ.get("GUNICORN_TIMEOUT", 120))      # worker silent timeout
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", 30))
keepalive        = int(os.environ.get("GUNICORN_KEEPALIVE", 5))

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog  = "-"   # stdout
errorlog   = "-"   # stderr
loglevel   = os.environ.get("GUNICORN_LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sµs'

# ── Process management ────────────────────────────────────────────────────────
preload_app   = True   # load app once before forking — saves memory, faster restarts
max_requests  = int(os.environ.get("GUNICORN_MAX_REQUESTS", 1000))
max_requests_jitter = 50   # prevent thundering-herd restarts

# ── Security ─────────────────────────────────────────────────────────────────
limit_request_line   = 4094
limit_request_fields = 100
