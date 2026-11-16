"""Ingestion pipeline for one file: read -> sections -> chunk -> embed -> store.

Re-indexing is idempotent: chunk IDs are derived from the file's content
hash, and a file's old chunks are removed before new ones are written.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from enum import StrEnum

from .chunker import chunk_sections, contextualize
from .config import settings
from .document_reader import read_any
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class EmptyDocumentError(Exception):
    """No extractable text, even after OCR."""


class IndexStatus(StrEnum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"  # same file, same content: nothing to do
    DUPLICATE = "duplicate"  # same content already indexed from another path


@dataclass
class IndexResult:
    path: str
    status: IndexStatus
    chunks: int = 0
    duplicate_of: str | None = None
    title: str | None = None
    ocr_pages: list[int] = field(default_factory=list)


def file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def index_document(path: str, store: VectorStore, extra_metadata: dict | None = None) -> IndexResult:
    path = os.path.abspath(path)
    content_hash = file_hash(path)

    existing = store.find_by_hash(content_hash)
    if existing is not None:
        if existing["filepath"] == path:
            return IndexResult(path, IndexStatus.UNCHANGED, title=existing.get("title"))
        return IndexResult(path, IndexStatus.DUPLICATE, duplicate_of=existing["filepath"])

    document = read_any(path)
    if not document.sections:
        raise EmptyDocumentError(f"No extractable text, even with OCR: {path}")

    chunks = chunk_sections(document.sections, settings.chunk_size, settings.chunk_overlap)

    filename = os.path.basename(path)
    ids, texts, embed_texts, metadatas = [], [], [], []
    for i, (page, heading, text) in enumerate(chunks):
        metadata = {
            "filename": filename,
            "filepath": path,
            "filetype": os.path.splitext(path)[1].lower(),
            "title": document.title,
            "content_hash": content_hash,
            "chunk": i,
            "total_chunks": len(chunks),
            **(extra_metadata or {}),
        }
        # Chroma metadata values cannot be None
        if page is not None:
            metadata["page"] = page
            if page in document.ocr_pages:
                metadata["ocr"] = True
        if heading:
            metadata["section"] = heading
        ids.append(f"{content_hash[:16]}-{i:05d}")
        texts.append(text)
        embed_texts.append(contextualize(document.title, heading, text))
        metadatas.append(metadata)

    # The file may have been indexed before with different content.
    store.delete_file(path)
    store.upsert_chunks(ids, texts, metadatas, embed_texts=embed_texts)

    logger.info("Indexed %s: %d sections, %d chunks%s", filename, len(document.sections),
                len(chunks), f", OCR pages {document.ocr_pages}" if document.ocr_pages else "")
    return IndexResult(path, IndexStatus.INDEXED, chunks=len(chunks), title=document.title,
                       ocr_pages=document.ocr_pages)
