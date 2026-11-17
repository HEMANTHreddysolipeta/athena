import os

# Before athena is imported: tests never load the real reranker model, and
# never touch the real library or database.
os.environ.setdefault("ATHENA_RERANK", "false")
os.environ.setdefault("ATHENA_MIN_RERANK_SCORE", "-4.0")
os.environ.setdefault("ATHENA_WARM_UP", "false")
import tempfile  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="athena-tests-")
os.environ["ATHENA_DB_PATH"] = os.path.join(_TMP, "athena.db")
os.environ["ATHENA_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["ATHENA_CHROMA_DIR"] = os.path.join(_TMP, "chroma")
# Athena's private folder, the token, and the "home folder" the assistant may
# look at are all throwaway: tests never see the real user's files.
os.environ["ATHENA_HOME"] = os.path.join(_TMP, "athena-home")
os.environ["ATHENA_TOKEN"] = "test-token-0123456789abcdefghijklmnop"
os.environ["ATHENA_ALLOWED_HOSTS"] = "testserver,127.0.0.1,localhost"
os.makedirs(os.path.join(_TMP, "home"), exist_ok=True)
os.environ["ATHENA_ASSISTANT_ROOT"] = os.path.join(_TMP, "home")
TOKEN_HEADERS = {"x-athena-token": os.environ["ATHENA_TOKEN"]}

import hashlib
import math
import re

import pymupdf
import pytest

from athena.vector_store import VectorStore

DIM = 64


def fake_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words vectors: shared words -> similar vectors.

    Not semantic like MiniLM, but enough to test storage and ranking in
    milliseconds without loading a 90 MB model.
    """
    vectors = []
    for text in texts:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma", "test", fake_embedder)


@pytest.fixture
def make_pdf(tmp_path):
    """make_pdf("a.pdf", ["page one text", "page two text"]) -> path"""

    def _make(name: str, pages: list[str]) -> str:
        path = tmp_path / name
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
        doc.save(path)
        doc.close()
        return str(path)

    return _make
