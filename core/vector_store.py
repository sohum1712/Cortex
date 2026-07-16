"""
core/vector_store.py — ChromaDB-backed vector store with incremental indexing.

Key design decisions
--------------------
- LangChain wrapper: uses langchain_chroma.Chroma (not raw chromadb client)
  so add_documents / similarity_search_with_score match the prior FAISS API.
- Lazy initialisation: the Chroma collection is only opened when the first
  document is added (or when a persisted store is found on disk), so the app
  starts without requiring Ollama to be up.
- Document hash cache (SHA-256) prevents re-embedding unchanged files.
- TextSplitter instantiated once in __init__, not per call.
- Embedding model + dimension stored in collection metadata — mismatch triggers
  automatic wipe and rebuild (same behaviour as the old FAISS dimension check).
- Chunk size / overlap configurable via config.py (defaults 800 / 100).
- top_k configurable, default 5.
- similarity_search_with_scores() exposes raw L2 distances so the caller
  (LLMRAGHandler) can apply a similarity threshold and skip the LLM call
  entirely when no relevant content is found.
- File-hash registry persisted alongside the Chroma store so the cache
  survives restarts.
"""

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import bs4
import chromadb
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = settings.EMBED_MODEL
EMBED_PROVIDER  = settings.EMBED_PROVIDER  # "huggingface" | "ollama"

# Known embedding dimensions per model name.
_EMBED_DIM_MAP = {
    # HuggingFace / sentence-transformers
    "all-MiniLM-L6-v2":            384,
    "all-mpnet-base-v2":           768,
    "BAAI/bge-small-en-v1.5":      384,
    "BAAI/bge-base-en-v1.5":       768,
    # Ollama (local)
    "mxbai-embed-large":          1024,
    "nomic-embed-text":            768,
    "all-minilm":                  384,
}
DEFAULT_EMBED_DIM = _EMBED_DIM_MAP.get(EMBEDDING_MODEL, 384)


def _build_embeddings(provider: str, model: str):
    """Return the correct LangChain embeddings object based on provider."""
    if provider == "ollama":
        from langchain_ollama.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    # Default: huggingface (works everywhere — no local server required)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

DEFAULT_INDEX_PATH = Path("data/index")

# Legacy FAISS artefacts — removed when a stale index is wiped.
_LEGACY_FAISS_FILES = ("index.faiss", "index.pkl")


class VectorStore:
    """
    Lazy-initialised Chroma vector store with incremental PDF and URL indexing.

    The underlying Chroma collection is not created until the first document is
    added, so constructing VectorStore never requires Ollama to be running.

    Parameters
    ----------
    index_path      : Parent directory for index artefacts (registry JSON, etc.).
    embedding_model : Ollama embedding model name (embeddings only — LLM is Groq).
    chunk_size      : Characters per text chunk (default from config).
    chunk_overlap   : Character overlap between consecutive chunks (default from config).
    persist         : If True, persist registry after every change (Chroma auto-persists).
    retrieval_k     : Default number of chunks returned per query.
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        persist: bool = True,
        retrieval_k: int = settings.TOP_K,
    ) -> None:
        self.index_path       = Path(index_path)
        self.persist          = persist
        self.retrieval_k      = retrieval_k
        self._embedding_model = embedding_model
        self._embed_provider  = EMBED_PROVIDER
        self._collection_name = settings.CHROMA_COLLECTION_NAME
        self._distance_metric = settings.CHROMA_DISTANCE_METRIC
        self._chroma_path     = self._resolve_chroma_path()

        # Resolve embedding dimension for the chosen model.
        self._embed_dim = _EMBED_DIM_MAP.get(embedding_model, DEFAULT_EMBED_DIM)

        # Build embeddings object — HuggingFace (cloud/free) or Ollama (local).
        logger.info(
            "Initialising embeddings: provider=%s model=%s",
            self._embed_provider, self._embedding_model,
        )
        self.embeddings_model = _build_embeddings(self._embed_provider, embedding_model)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._registry_path: Path = self.index_path / "indexed_files.json"
        self._indexed_hashes: Set[str] = self._load_registry()

        # None until _ensure_store() is called or _try_load_existing() succeeds.
        self._store: Optional[Chroma] = None
        self._try_load_existing()

    # ------------------------------------------------------------------
    # Internal: paths & metadata
    # ------------------------------------------------------------------

    def _resolve_chroma_path(self) -> Path:
        """Return the Chroma persist directory (env override or <index_path>/chroma)."""
        override = settings.CHROMA_PERSIST_DIR
        if override is not None:
            return Path(override)
        return self.index_path / "chroma"

    def _collection_metadata(self) -> dict:
        """Metadata written when a new Chroma collection is created."""
        return {
            "hnsw:space":  self._distance_metric,
            "embed_model": self._embedding_model,
            "embed_dim":   str(self._embed_dim),
        }

    def _chroma_has_data(self) -> bool:
        """True when a persisted Chroma database exists on disk."""
        return (self._chroma_path / "chroma.sqlite3").exists()

    # ------------------------------------------------------------------
    # Internal: store lifecycle
    # ------------------------------------------------------------------

    def _read_collection_metadata(self) -> Optional[dict]:
        """Read metadata from the persisted Chroma collection, if it exists."""
        if not self._chroma_has_data():
            return None
        try:
            client = chromadb.PersistentClient(path=str(self._chroma_path))
            collection = client.get_collection(name=self._collection_name)
            return collection.metadata or {}
        except Exception as exc:
            logger.debug("Could not read Chroma collection metadata: %s", exc)
            return None

    def _validate_stored_metadata(self) -> None:
        """
        Raise ValueError when the on-disk collection was built with a different
        embedding model or vector dimension than the current configuration.
        """
        meta = self._read_collection_metadata()
        if not meta:
            return

        stored_model = meta.get("embed_model")
        stored_dim   = meta.get("embed_dim")

        if stored_model and stored_model != self._embedding_model:
            raise ValueError(
                f"Embedding model mismatch: stored={stored_model!r}, "
                f"current={self._embedding_model!r}. "
                f"Deleting stale index and rebuilding."
            )

        if stored_dim is not None:
            try:
                if int(stored_dim) != self._embed_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: stored={stored_dim}, "
                        f"model expects={self._embed_dim}. "
                        f"Deleting stale index and rebuilding."
                    )
            except (TypeError, ValueError) as exc:
                if "mismatch" in str(exc):
                    raise
                logger.debug("Ignoring unparseable embed_dim metadata: %r", stored_dim)

    def _try_load_existing(self) -> None:
        """
        Load a persisted Chroma collection from disk if one exists.

        If the collection was built with a different embedding model or
        dimension, it is automatically deleted and a fresh collection will be
        created on the next document upload.
        """
        if not self._chroma_has_data():
            legacy = self.index_path / "index.faiss"
            if legacy.exists():
                logger.warning(
                    "Legacy FAISS index found at %s but no Chroma store. "
                    "Clearing indexed_files.json so PDFs are re-embedded into Chroma "
                    "(auto-index on next LLM request, or POST /api/rebuild).",
                    legacy,
                )
                # Registry still lists FAISS-era hashes while Chroma is empty —
                # without this, add_document() would skip every file.
                self._indexed_hashes = set()
                if self._registry_path.exists():
                    self._registry_path.unlink()
            else:
                logger.info(
                    "No Chroma index on disk — will create on first document upload."
                )
            return

        t0 = time.perf_counter()
        logger.info("Loading Chroma collection from %s", self._chroma_path)
        try:
            self._validate_stored_metadata()
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self.embeddings_model,
                persist_directory=str(self._chroma_path),
            )
            logger.info("Chroma collection loaded in %.2fs", time.perf_counter() - t0)
        except ValueError as exc:
            logger.warning("Stale Chroma index discarded: %s", exc)
            self._delete_index_files()
        except Exception as exc:
            logger.warning("Could not load Chroma index (%s) — will rebuild.", exc)
            self._delete_index_files()

    def _delete_index_files(self) -> None:
        """Remove all persisted index artefacts so a clean rebuild happens."""
        if self._chroma_path.exists():
            shutil.rmtree(self._chroma_path)
            logger.info("Deleted Chroma persist directory: %s", self._chroma_path)

        for fname in (*_LEGACY_FAISS_FILES, "indexed_files.json"):
            p = self.index_path / fname
            if p.exists():
                p.unlink()
                logger.info("Deleted stale index file: %s", p.name)

        self._indexed_hashes = set()
        self._store = None

    def _ensure_store(self) -> Chroma:
        """
        Return the Chroma store, creating an empty collection if needed.

        No Ollama call is required to construct an empty collection.
        """
        if self._store is not None:
            return self._store

        self._chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Creating new Chroma collection %r (metric=%s, dim=%d) at %s",
            self._collection_name,
            self._distance_metric,
            self._embed_dim,
            self._chroma_path,
        )
        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self.embeddings_model,
            persist_directory=str(self._chroma_path),
            collection_metadata=self._collection_metadata(),
        )
        return self._store

    def _save(self) -> None:
        """Persist the file-hash registry (Chroma writes vectors automatically)."""
        if self.persist:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._save_registry()

    # Expose the underlying store as a property for callers that need it.
    @property
    def vector_store(self) -> Optional[Chroma]:
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

    def similarity_search_with_scores(
        self, question: str, k: int = 0
    ) -> List[Tuple[Document, float]]:
        """
        Return the top-k most similar chunks with their L2 distance scores.

        Lower scores = more similar. Returns an empty list when no index
        has been built yet, allowing the caller to handle the no-context case.

        Score semantics
        ---------------
        The Chroma collection is created with ``hnsw:space = "l2"`` (configurable
        via CHROMA_DISTANCE_METRIC, default ``l2``).  LangChain's
        ``similarity_search_with_score`` returns the raw distance for this metric,
        where lower values mean closer vectors — matching the prior FAISS
        IndexFlatL2 behaviour and LLMRAGHandler's SIMILARITY_THRESHOLD logic.
        No score inversion or normalisation is applied.
        """
        if self._store is None:
            logger.warning("similarity_search called but no index built yet.")
            return []

        effective_k = k if k > 0 else self.retrieval_k
        t0 = time.perf_counter()
        results = self._store.similarity_search_with_score(question, k=effective_k)
        elapsed = time.perf_counter() - t0

        if not results:
            return []

        scores = ", ".join(f"{s:.3f}" for _, s in results)
        logger.debug(
            "Search %.2fs | %d chunks | %s scores: [%s]",
            elapsed,
            len(results),
            self._distance_metric.upper(),
            scores,
        )
        return results

    def similarity_search(self, question: str, k: int = 0) -> List[Document]:
        """
        Return the top-k most similar chunks (no scores).

        Convenience wrapper around similarity_search_with_scores for callers
        that do not need the distance values.
        """
        return [doc for doc, _ in self.similarity_search_with_scores(question, k=k)]

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
