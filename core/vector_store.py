"""
core/vector_store.py — FAISS-backed vector store with incremental indexing.

Key design decisions
--------------------
- Lazy initialisation: the FAISS store is only created when the first
  document is added, so the app starts without requiring Ollama to be up.
- Document hash cache (SHA-256) prevents re-embedding unchanged files.
- TextSplitter instantiated once in __init__, not per call.
- Embedding dimension stored as a class constant — no live Ollama call
  needed just to create an empty index.
- No hard score threshold — the LLM handles relevance judgement in the prompt.
- Chunk size 800 / overlap 100 — fewer chunks, faster search, richer context.
- File-hash registry persisted alongside the FAISS index so the cache
  survives restarts.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Set

import bs4
import faiss
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CHAT_MODEL      = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# Known embedding dimension for nomic-embed-text — avoids a live Ollama call
# on startup just to create an empty index.
NOMIC_EMBED_DIM = 768

DEFAULT_INDEX_PATH = Path("data/index")


class VectorStore:
    """
    Lazy-initialised FAISS vector store with incremental PDF and URL indexing.

    The underlying FAISS index is not created until the first document is
    added, so constructing VectorStore never requires Ollama to be running.

    Parameters
    ----------
    index_path      : Directory where the FAISS index is persisted.
    llm_model       : Chat model name (stored for logging; not used directly).
    embedding_model : Ollama embedding model name.
    chunk_size      : Characters per text chunk.
    chunk_overlap   : Character overlap between consecutive chunks.
    persist         : If True, save index to disk after every change.
    retrieval_k     : Default number of chunks returned per query.
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        llm_model: str = CHAT_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        persist: bool = True,
        retrieval_k: int = 3,
    ) -> None:
        self.index_path  = Path(index_path)
        self.llm_model   = llm_model
        self.persist     = persist
        self.retrieval_k = retrieval_k

        # Embeddings object is cheap to create — no network call here.
        self.embeddings_model = OllamaEmbeddings(model=embedding_model)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._registry_path: Path = self.index_path / "indexed_files.json"
        self._indexed_hashes: Set[str] = self._load_registry()

        # None until _ensure_store() is called for the first time.
        self._store: Optional[FAISS] = None
        self._try_load_existing()

    # ------------------------------------------------------------------
    # Internal: store lifecycle
    # ------------------------------------------------------------------

    def _try_load_existing(self) -> None:
        """Load a persisted FAISS index from disk if one exists."""
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            logger.info(
                "No FAISS index on disk — will create on first document upload."
            )
            return
        t0 = time.perf_counter()
        logger.info("Loading FAISS index from %s", self.index_path)
        self._store = FAISS.load_local(
            str(self.index_path),
            embeddings=self.embeddings_model,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded in %.2fs", time.perf_counter() - t0)

    def _ensure_store(self) -> FAISS:
        """
        Return the FAISS store, creating an empty one if it does not exist yet.

        This is the only place that constructs a brand-new empty index.
        Uses the known embedding dimension constant so no Ollama call is needed.
        """
        if self._store is not None:
            return self._store

        logger.info(
            "Creating new empty FAISS index (dim=%d) at %s",
            NOMIC_EMBED_DIM, self.index_path,
        )
        raw_index = faiss.IndexFlatL2(NOMIC_EMBED_DIM)
        self._store = FAISS(
            embedding_function=self.embeddings_model,
            index=raw_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return self._store

    def _save(self) -> None:
        """Persist the FAISS index and file-hash registry to disk."""
        if self.persist and self._store is not None:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.index_path))
            self._save_registry()

    # Expose the underlying store as a property for callers that need it.
    @property
    def vector_store(self) -> Optional[FAISS]:
        return self._store

    # ------------------------------------------------------------------
    # File-hash registry
    # ------------------------------------------------------------------

    def _file_hash(self, file_path: Path) -> str:
        """Return the SHA-256 hex digest of a file's contents."""
        h = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_registry(self) -> Set[str]:
        if self._registry_path.exists():
            try:
                with self._registry_path.open(encoding="utf-8") as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, ValueError):
                pass
        return set()

    def _save_registry(self) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._registry_path.open("w", encoding="utf-8") as f:
            json.dump(sorted(self._indexed_hashes), f, indent=2)

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def load_document(self, pdf_path: Path) -> List[Document]:
        """Load a single PDF and return its pages as Document objects."""
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        return docs if isinstance(docs, list) else [docs]

    def load_documents(self, directory: Path) -> List[Document]:
        """Load all PDFs from a directory."""
        result: List[Document] = []
        for pdf in Path(directory).glob("*.pdf"):
            result.extend(self.load_document(pdf))
        return result

    # ------------------------------------------------------------------
    # Chunking & indexing
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    def add_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk, embed, and index a list of documents, then persist."""
        t0     = time.perf_counter()
        chunks = self.chunk_documents(documents)
        logger.debug("Produced %d chunks in %.2fs", len(chunks), time.perf_counter() - t0)

        store = self._ensure_store()
        t1    = time.perf_counter()
        store.add_documents(chunks)
        logger.debug(
            "Embedded + indexed %d chunks in %.2fs",
            len(chunks), time.perf_counter() - t1,
        )
        self._save()
        return chunks

    def add_document(self, file_path: Path) -> Optional[List[Document]]:
        """
        Index a single PDF file.

        Skips silently if the file's SHA-256 hash is already in the registry
        (prevents redundant re-embedding on duplicate uploads).

        Returns the list of indexed chunks, or None if the file was skipped.
        """
        file_hash = self._file_hash(file_path)
        if file_hash in self._indexed_hashes:
            logger.info("Skipping %s — already indexed.", file_path.name)
            return None

        logger.info("Indexing %s ...", file_path.name)
        docs   = self.load_document(file_path)
        chunks = self.add_documents(docs)

        self._indexed_hashes.add(file_hash)
        self._save_registry()
        return chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def similarity_search(self, question: str, k: int = 3) -> List[Document]:
        """
        Return the top-k most similar chunks for the given question.

        Returns an empty list when no index has been built yet (no documents
        uploaded), allowing the LLM to respond with a "no context" answer.
        """
        if self._store is None:
            logger.warning("similarity_search called but no index built yet.")
            return []

        t0      = time.perf_counter()
        results = self._store.similarity_search_with_score(question, k=k)
        elapsed = time.perf_counter() - t0

        if not results:
            return []

        scores = ", ".join(f"{s:.3f}" for _, s in results)
        logger.debug(
            "Search %.2fs | %d chunks | L2 scores: [%s]", elapsed, len(results), scores
        )
        return [doc for doc, _ in results]

    def as_retriever(self) -> VectorStoreRetriever:
        return self._ensure_store().as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )

    # ------------------------------------------------------------------
    # Web / URL indexing
    # ------------------------------------------------------------------

    def index_websites(self, urls: List[str]) -> List[Document]:
        """Fetch, chunk, and index web pages from the given URLs."""
        docs = self._load_websites(urls)
        return self.add_documents(docs)

    def _load_websites(self, urls: List[str]) -> List[Document]:
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        return loader.load()
