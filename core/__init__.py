"""
core — RAG engine internals.

Exports the three main components so callers can do:
    from core import LLMRAGHandler, VectorStore, ConversationManager
"""

from core.llm_rag import LLMRAGHandler
from core.vector_store import VectorStore, EMBEDDING_MODEL
from core.conversation import ConversationManager

__all__ = [
    "LLMRAGHandler",
    "VectorStore",
    "EMBEDDING_MODEL",
    "ConversationManager",
]
