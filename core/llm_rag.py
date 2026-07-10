"""
core/llm_rag.py — LLM + RAG handler with lazy Ollama initialisation.

Design decisions
----------------
- ChatOllama is constructed lazily on the first generate call, NOT in
  __init__. This means importing / instantiating LLMRAGHandler never
  requires Ollama to be running (important for cold-start, reset, clear).
- VectorStore is also constructed in __init__ but is itself lazy — it only
  calls Ollama when the first document is indexed.
- Streaming via generate_response_stream() yields tokens as they arrive.
- Only the last MAX_HISTORY_TURNS human/AI pairs are sent in the prompt,
  keeping token usage bounded as the conversation grows.
- Context is formatted as plain numbered text, not raw Document repr.
"""

import logging
import time
from pathlib import Path
from typing import Iterator, List, Optional

from langchain.schema import AIMessage, BaseMessage, Document, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core.vector_store import DEFAULT_INDEX_PATH, EMBEDDING_MODEL, VectorStore

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 4   # human/AI pairs kept in the prompt

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


class LLMRAGHandler:
    """
    Retrieval-Augmented Generation handler backed by a local Ollama model.

    Ollama is contacted only on the first generate call — not on construction.

    Public API
    ----------
    generate_response(message)          str            blocking
    generate_response_stream(message)   Iterator[str]  streaming tokens
    add_pdf_to_context(path)            index a PDF into the vector store
    reset()                             clear in-memory conversation history
    get_history()                       return current BaseMessage list
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        embedding_model: str = EMBEDDING_MODEL,
        index_path: Path = DEFAULT_INDEX_PATH,
        num_predict: int = 256,
        temperature: float = 0.1,
        top_k: int = 20,
        retrieval_k: int = 4,
    ) -> None:
        self._model_name    = model
        self._num_predict   = num_predict
        self._temperature   = temperature
        self._top_k         = top_k
        self.retrieval_k    = retrieval_k

        # VectorStore is safe to construct without Ollama (lazy internally).
        self.vector_store = VectorStore(
            llm_model=model,
            embedding_model=embedding_model,
            index_path=index_path,
        )

        self.history: List[BaseMessage] = [
            SystemMessage(content=_SYSTEM_PROMPT)
        ]

        # _llm and _chain are built on first use.
        self._llm   = None
        self._chain = None

    # ------------------------------------------------------------------
    # Lazy LLM construction
    # ------------------------------------------------------------------

    def _get_chain(self):
        """Build (once) and return the LangChain runnable chain."""
        if self._chain is None:
            from langchain_ollama.chat_models import ChatOllama

            logger.info(
                "Connecting to Ollama model '%s' for the first time…",
                self._model_name,
            )
            self._llm = ChatOllama(
                model=self._model_name,
                num_predict=self._num_predict,
                temperature=self._temperature,
                top_k=self._top_k,
                num_thread=12,
            )
            self._chain = _RAG_PROMPT | self._llm | StrOutputParser()

        return self._chain

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_history(self) -> str:
        turns  = [m for m in self.history if not isinstance(m, SystemMessage)]
        recent = turns[-(MAX_HISTORY_TURNS * 2):]
        lines  = []
        for m in recent:
            prefix = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def _format_context(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant context found."
        return "\n\n".join(
            f"[{i + 1}] {doc.page_content.strip()}" for i, doc in enumerate(docs)
        )

    def _retrieve(self, question: str) -> List[Document]:
        t0   = time.perf_counter()
        docs = self.vector_store.similarity_search(question, k=self.retrieval_k)
        logger.debug("Retrieval %.2fs — %d chunks", time.perf_counter() - t0, len(docs))
        return docs

    def _build_inputs(self, human_message: str) -> dict:
        docs = self._retrieve(human_message)
        return {
            "input":        human_message,
            "context":      self._format_context(docs),
            "chat_history": self._format_history(),
        }

    def _append_history(self, human_message: str, ai_response: str) -> None:
        self.history.append(HumanMessage(content=human_message))
        self.history.append(AIMessage(content=ai_response))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(self, human_message: str) -> str:
        """Blocking generation — returns the full response string."""
        inputs   = self._build_inputs(human_message)
        response = self._get_chain().invoke(inputs)
        self._append_history(human_message, response)
        return response

    def generate_response_stream(self, human_message: str) -> Iterator[str]:
        """
        Streaming generation — yields tokens as they arrive.
        Conversation history is updated after the stream completes.
        """
        inputs        = self._build_inputs(human_message)
        full_response = ""
        t0            = time.perf_counter()
        first         = True

        for token in self._get_chain().stream(inputs):
            if first:
                logger.debug("Time to first token: %.2fs", time.perf_counter() - t0)
                first = False
            full_response += token
            yield token

        logger.debug(
            "Generation complete %.2fs (%d chars)",
            time.perf_counter() - t0,
            len(full_response),
        )
        self._append_history(human_message, full_response)

    def add_pdf_to_context(self, file_path: Path) -> None:
        """Index a PDF file into the vector store."""
        self.vector_store.add_document(Path(file_path))

    def reset(self) -> None:
        """Reset in-memory conversation history, keeping the system prompt."""
        self.history = [SystemMessage(content=_SYSTEM_PROMPT)]

    def get_history(self) -> List[BaseMessage]:
        return self.history
