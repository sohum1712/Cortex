"""
core/llm_rag.py — LLM + RAG handler backed by Groq.
"""

import logging
import time
from pathlib import Path
from typing import Iterator, List

from langchain.schema import AIMessage, BaseMessage, Document, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from config import ENV_FILE, load_env, settings
from core.vector_store import DEFAULT_INDEX_PATH, VectorStore

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 4

_NO_CONTEXT_REPLY = "I could not find this in the uploaded documents."

_SYSTEM_PROMPT = (
    "You are a helpful Q&A assistant. "
    "When document context is provided, answer strictly from that context. "
    "For greetings or small talk, respond naturally and briefly. "
    "If asked about a topic and the context does not contain the answer, "
    "say: 'I could not find this in the uploaded documents.' "
    "Never fabricate facts. Reply in 3 sentences or fewer."
)

_RAG_PROMPT = PromptTemplate.from_template(
    "Use the context below to answer the question.\n"
    "If the context is relevant, base your answer on it.\n"
    "If the context does not relate to the question, say so politely.\n\n"
    "Context:\n{context}\n\n"
    "Recent conversation:\n{chat_history}\n\n"
    "Question: {input}\n"
    "Answer:"
)


def validate_groq_key() -> None:
    """
    Test the Groq API key with a minimal call.
    Raises RuntimeError with a clear message if the key is missing or rejected.
    Call this once at startup so the problem is visible immediately.
    """
    load_env(override=True)
    key = settings.GROQ_API_KEY
    if not key:
        raise RuntimeError(
            "\n\n  GROQ_API_KEY is not set.\n"
            f"  Expected .env at: {ENV_FILE}\n"
            "  1. Get a free key at https://console.groq.com\n"
            "  2. Copy .env.example to .env and add:  GROQ_API_KEY=gsk_...\n"
            "  3. Restart the server (or just send a chat message after saving).\n"
        )

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=settings.GROQ_MODEL, api_key=key, temperature=0, max_tokens=1)
        llm.invoke("hi")
        logger.info("Groq API key validated OK (model: %s)", settings.GROQ_MODEL)
    except Exception as exc:
        msg = str(exc)
        raise RuntimeError(
            f"\n\n  Groq API key validation FAILED.\n"
            f"  .env path: {ENV_FILE}\n"
            f"  Key prefix: {key[:8]}... (length {len(key)})\n"
            f"  Error: {msg}\n\n"
            f"  Fix: Go to https://console.groq.com, create a new API key,\n"
            f"  update GROQ_API_KEY in {ENV_FILE}, and try again.\n"
        ) from exc


def _is_auth_error(exc: Exception, msg: str) -> bool:
    """Return True only for genuine Groq authentication failures."""
    lower = msg.lower()
    exc_type = type(exc).__name__.lower()

    if "401" in msg or "403" in msg:
        return True
    if "invalid_api_key" in lower or "incorrect api key" in lower:
        return True
    if "authentication" in lower and "api" in lower:
        return True
    if "unauthorized" in lower or "unauthorised" in lower:
        return True
    if exc_type in {"authenticationerror", "permissiondeniederror"}:
        return True

    # Avoid false positives from generic messages like "api_key is required".
    if "api_key" in lower and any(
        phrase in lower
        for phrase in ("invalid", "incorrect", "rejected", "revoked", "unauthorized", "unauthorised")
    ):
        return True

    return False


def _is_groq_error(exc: Exception) -> str:
    """
    Always returns a non-empty, human-readable error string.
    Includes the raw exception type so the user can diagnose it.
    """
    raw = str(exc)
    msg = raw.lower()
    exc_type = type(exc).__name__

    if not raw.strip():
        raw = repr(exc)
        msg = raw.lower()

    if not settings.GROQ_API_KEY:
        return (
            f"GROQ_API_KEY is not set. Add it to {ENV_FILE} "
            "(copy from .env.example if needed), then try again."
        )

    if _is_auth_error(exc, raw):
        if "403" in msg:
            return (
                "Groq API access denied (403). "
                "Your key may have been revoked. "
                f"Create a new key at https://console.groq.com and update {ENV_FILE}."
            )
        return (
            "Groq API key is invalid or not authorised. "
            f"Check the GROQ_API_KEY value in {ENV_FILE}. "
            "Create a new key at https://console.groq.com if needed."
        )
    if "rate_limit" in msg or "429" in msg:
        return "Groq rate limit hit. Wait a few seconds and try again."
    if "connection" in msg or "timeout" in msg or "network" in msg or "unreachable" in msg:
        return "Cannot reach Groq API. Check your internet connection."
    if "assert" in msg or not raw.strip():
        return f"Internal error ({exc_type}). Check server logs."

    return f"Groq error ({exc_type}): {raw}"


class LLMRAGHandler:

    def __init__(
        self,
        embedding_model: str = None,
        index_path: Path = DEFAULT_INDEX_PATH,
        retrieval_k: int = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        similarity_threshold: float = None,
    ) -> None:
        # Read from settings at construction time (settings uses @property so always fresh)
        self.retrieval_k          = retrieval_k          if retrieval_k          is not None else settings.TOP_K
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        _embedding_model          = embedding_model      if embedding_model      is not None else settings.EMBED_MODEL
        _chunk_size               = chunk_size           if chunk_size           is not None else settings.CHUNK_SIZE
        _chunk_overlap            = chunk_overlap        if chunk_overlap        is not None else settings.CHUNK_OVERLAP

        self.vector_store = VectorStore(
            embedding_model=_embedding_model,
            index_path=index_path,
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
            retrieval_k=self.retrieval_k,
        )

        self.history: List[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]
        self._llm        = None
        self._chain      = None
        self._active_key = None   # track which key the current chain was built with
        self._last_sources: List[Document] = []  # populated after each stream call

    # ------------------------------------------------------------------
    # Lazy LLM — built fresh on first use, reads key at that moment
    # ------------------------------------------------------------------

    def _get_chain(self):
        """
        Build (or rebuild) and return the LangChain chain.

        Re-reads .env on every call so a key change takes effect immediately
        without restarting the server. If the key has changed since the chain
        was last built, the old chain is discarded and a new one is created.
        """
        from config import load_env
        load_env(override=True)   # reload .env — picks up any edits made since startup

        api_key = settings.GROQ_API_KEY   # @property → reads os.environ (just refreshed)
        model   = settings.GROQ_MODEL

        if not api_key:
            raise RuntimeError(
                f"GROQ_API_KEY is empty. Add it to {ENV_FILE} — "
                "no restart needed, just send your message again."
            )

        # Rebuild the chain if it hasn't been built yet, or if the key changed.
        if self._chain is None or api_key != self._active_key:
            if self._chain is not None:
                logger.info("GROQ_API_KEY changed — rebuilding chain with new key.")
            logger.info("Building Groq chain (model=%s)", model)
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model=model,
                api_key=api_key,
                temperature=0,
                streaming=True,
            )
            self._chain      = _RAG_PROMPT | self._llm | StrOutputParser()
            self._active_key = api_key
            logger.info("Groq chain ready.")

        return self._chain

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_history(self) -> str:
        turns  = [m for m in self.history if not isinstance(m, SystemMessage)]
        recent = turns[-(MAX_HISTORY_TURNS * 2):]
        lines  = []
        for m in recent:
            prefix = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    @staticmethod
    def _deduplicate_chunks(docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        sorted_docs = sorted(docs, key=lambda d: len(d.page_content), reverse=True)
        unique: List[Document] = []
        for candidate in sorted_docs:
            c_text = candidate.page_content.strip()
            is_dup = False
            for kept in unique:
                k_text = kept.page_content.strip()
                if c_text in k_text or k_text in c_text:
                    is_dup = True
                    break
                shorter, longer = sorted([c_text, k_text], key=len)
                overlap = sum(1 for ch in shorter if ch in longer)
                if longer and (overlap / len(longer)) > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(candidate)
        if len(unique) < len(docs):
            logger.debug("Deduplicated chunks: %d → %d", len(docs), len(unique))
        return unique

    def _format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant context found."
        return "\n\n".join(
            f"[{i + 1}] {doc.page_content.strip()}" for i, doc in enumerate(docs)
        )

    def _retrieve(self, question: str) -> List[Document]:
        t0 = time.perf_counter()
        docs_with_scores = self.vector_store.similarity_search_with_scores(
            question, k=self.retrieval_k
        )
        logger.debug("Retrieval %.2fs — %d chunks", time.perf_counter() - t0, len(docs_with_scores))

        if not docs_with_scores:
            return []

        if self.similarity_threshold > 0.0:
            best_score = docs_with_scores[0][1]
            if best_score > self.similarity_threshold:
                return []

        docs = [doc for doc, _ in docs_with_scores]
        # Store scores alongside docs as metadata for the caller
        for doc, score in docs_with_scores:
            doc.metadata["_score"] = round(float(score), 4)
        return self._deduplicate_chunks(docs)

    def _build_inputs(self, human_message: str) -> dict:
        docs = self._retrieve(human_message)
        return {
            "input":        human_message,
            "context":      self._format_context(docs),
            "chat_history": self._format_history(),
            "_docs":        docs,
        }

    def _append_history(self, human_message: str, ai_response: str) -> None:
        self.history.append(HumanMessage(content=human_message))
        self.history.append(AIMessage(content=ai_response))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(self, human_message: str) -> str:
        inputs = self._build_inputs(human_message)
        if not inputs["_docs"] and self.similarity_threshold > 0.0:
            self._append_history(human_message, _NO_CONTEXT_REPLY)
            return _NO_CONTEXT_REPLY

        chain_inputs = {k: v for k, v in inputs.items() if k != "_docs"}
        try:
            response = self._get_chain().invoke(chain_inputs)
        except Exception as exc:
            response = _is_groq_error(exc)
            logger.exception("Groq generation error: %s", exc)

        self._append_history(human_message, response)
        return response

    def generate_response_stream(self, human_message: str) -> Iterator[str]:
        inputs = self._build_inputs(human_message)

        if not inputs["_docs"] and self.similarity_threshold > 0.0:
            self._last_sources = []  # no docs found — clear so routes.py sees empty list
            self._append_history(human_message, _NO_CONTEXT_REPLY)
            yield _NO_CONTEXT_REPLY
            return

        # Store retrieved docs so the caller can access them after streaming
        self._last_sources = inputs["_docs"]

        chain_inputs  = {k: v for k, v in inputs.items() if k != "_docs"}
        full_response = ""
        t0            = time.perf_counter()
        first_token   = True

        try:
            for token in self._get_chain().stream(chain_inputs):
                if first_token:
                    logger.debug("Time to first token: %.2fs", time.perf_counter() - t0)
                    first_token = False
                full_response += token
                yield token
        except Exception as exc:
            error_msg = _is_groq_error(exc)
            logger.exception("Groq streaming error: %s", exc)
            full_response = error_msg
            yield error_msg
            self._append_history(human_message, full_response)
            return

        logger.debug("Stream done %.2fs (%d chars)", time.perf_counter() - t0, len(full_response))
        self._append_history(human_message, full_response)

    def add_pdf_to_context(self, file_path: Path) -> None:
        self.vector_store.add_document(Path(file_path))

    def reset(self) -> None:
        self.history = [SystemMessage(content=_SYSTEM_PROMPT)]
        self._last_sources = []
        # Force chain rebuild so fresh key is used on next call
        self._llm        = None
        self._chain      = None
        self._active_key = None

    def get_history(self) -> List[BaseMessage]:
        return self.history
