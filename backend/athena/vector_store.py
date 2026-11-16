"""ChromaDB wrapper: embedding, storage and semantic search.

Data flow:
  text --embedder--> vector (384 floats for MiniLM) --> Chroma HNSW index
  query --same embedder--> vector --> nearest stored vectors (cosine distance)

The query and the documents MUST use the same embedding model; vectors from
different models live in different "spaces" and their distances mean nothing.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb

from .config import settings

logger = logging.getLogger(__name__)

# An embedder turns a batch of texts into a batch of vectors.
Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    distance: float = 1.0  # cosine distance: 0 = identical direction, ~1 = unrelated
    # Filled in by the retriever (retriever.py) when it fuses and reranks.
    dense_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None


class SentenceTransformerEmbedder:
    """Loads the model on first use, not at import time."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow import

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()


class VectorStore:
    def __init__(self, path: Path, collection_name: str, embedder: Embedder, query_prefix: str = ""):
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection_name = collection_name
        self._embed = embedder
        # Some models (BGE) are trained with an instruction in front of
        # queries but not passages.
        self._query_prefix = query_prefix
        self._collection = self._open_collection()
        # Bumped on every write, so caches built from the collection (the
        # keyword index) know when they are stale.
        self.version = 0

    def _open_collection(self):
        return self._client.get_or_create_collection(
            self._collection_name, configuration={"hnsw": {"space": "cosine"}}
        )

    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        embed_texts: list[str] | None = None,
    ) -> None:
        """Store chunks. `embed_texts` (e.g. with a title/heading prefix) is
        what gets embedded; `texts` is what is stored and shown."""
        if not ids:
            return
        vectors = []
        source = embed_texts or texts
        for start in range(0, len(source), 256):  # bounded memory on big files
            vectors.extend(self._embed(source[start:start + 256]))
        for start in range(0, len(ids), 2000):  # Chroma's max batch size is ~5k
            end = start + 2000
            self._collection.upsert(
                ids=ids[start:end], documents=texts[start:end],
                embeddings=vectors[start:end], metadatas=metadatas[start:end],
            )
        self.version += 1

    def delete_file(self, filepath: str) -> None:
        self._collection.delete(where={"filepath": filepath})
        self.version += 1

    def get_all(self) -> tuple[list[str], list[str], list[dict]]:
        """Every chunk's id, text and metadata (for the keyword index)."""
        result = self._collection.get(include=["documents", "metadatas"])
        return result["ids"], result["documents"], result["metadatas"]

    def documents(self) -> list[dict]:
        """One summary row per indexed file."""
        result = self._collection.get(include=["metadatas"])
        files: dict[str, dict] = {}
        for m in result["metadatas"]:
            row = files.setdefault(m["filepath"], {
                "filename": m["filename"], "filepath": m["filepath"],
                "title": m.get("title", m["filename"]), "filetype": m.get("filetype"),
                "chunks": 0, "pages": set(), "ocr": False, "source_url": m.get("source_url"),
            })
            row["chunks"] += 1
            if "page" in m:
                row["pages"].add(m["page"])
            row["ocr"] = row["ocr"] or bool(m.get("ocr"))
        rows = []
        for row in files.values():
            row["pages"] = len(row["pages"]) or None
            rows.append(row)
        return sorted(rows, key=lambda r: r["filename"].lower())

    def find_by_hash(self, content_hash: str) -> dict | None:
        """Metadata of one chunk from a file with this content hash, if indexed."""
        result = self._collection.get(
            where={"content_hash": content_hash}, limit=1, include=["metadatas"]
        )
        return result["metadatas"][0] if result["ids"] else None

    def list_files(self) -> list[str]:
        result = self._collection.get(include=["metadatas"])
        return sorted({m["filepath"] for m in result["metadatas"]})

    def search(self, query: str, k: int, where: dict | None = None) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embed([self._query_prefix + query]),
            n_results=min(k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(id=i, text=t, metadata=m, distance=d)
            for i, t, m, d in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._open_collection()
        self.version += 1
        logger.info("Collection %s reset", self._collection_name)


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """The application's shared store, created on first use."""
    return VectorStore(
        settings.chroma_dir,
        settings.collection_name,
        SentenceTransformerEmbedder(settings.embedding_model),
        query_prefix=settings.embedding_query_prefix,
    )
