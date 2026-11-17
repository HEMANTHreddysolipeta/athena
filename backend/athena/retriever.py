"""Hybrid retrieval: embeddings + BM25 keywords, fused, then reranked.

    question ─┬─ dense search (embeddings, top N) ──┐
              └─ BM25 keyword search (top N) ───────┴─ reciprocal rank fusion
                 ─ cross-encoder reranks the best M ─ top k chunks

Why both retrievers: embeddings match meaning ("how do I stop people
guessing passwords" -> "throttling"), but blur exact tokens such as
numbers, acronyms and names ("AAL2", "31", "ClickSlice"). BM25 matches
exact tokens but misses paraphrases. Fusing their *ranks* (not their
scores, which live on different scales) gets most of both.

Why rerank: both retrievers score the question and each chunk separately.
A cross-encoder reads the pair together, which is far more accurate but
too slow to run over every chunk, so it only re-orders the fused shortlist.
Its top score also tells us when nothing relevant was found at all.
"""

import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

from .chunker import contextualize
from .config import settings
from .vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the RRF paper (Cormack et al., 2009)

# Common words that carry no retrieval signal. Kept short on purpose:
# words like "not", "before" and "after" can matter in questions.
_STOPWORDS = frozenset(
    ["a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were", "be", "been", "being", "by", "with", "as", "at", "from", "that", "this", "these", "those", "it", "its", "what", "which", "who", "whom", "how", "does", "do", "did", "can", "could", "should", "would", "will", "about", "into", "than", "then", "there", "their", "they", "them", "i", "me", "my", "we", "our", "you", "your"]
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps '800-63b', '2.0', 'gpt-4' as one token,
    and also emits their parts so 'csf 2.0' matches '2.0' and 'csf'."""
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        if "-" in token or "." in token:
            tokens.extend(p for p in re.split(r"[.\-]", token) if p and p not in _STOPWORDS)
    return tokens


class BM25:
    """Okapi BM25 with Lucene's IDF, over an inverted index.

    Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)), is always positive.
    The classic Okapi IDF goes to zero or negative once a word is in half
    of the chunks, which in a small personal library means a matching word
    can push a chunk *down*. The inverted index also means a query only
    touches chunks that contain one of its words.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.lengths = [len(doc) for doc in corpus]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(corpus):
            for token in doc:
                counts = self.postings.setdefault(token, {})
                counts[i] = counts.get(i, 0) + 1
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        for token in set(query):
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf[token]
            for i, tf in postings.items():
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class KeywordIndex:
    """BM25 over every chunk (title + heading + text), rebuilt when the store changes."""

    def __init__(self, store: VectorStore):
        self._store = store
        self._lock = threading.Lock()
        self._key = None
        self._bm25: BM25 | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []

    def _ensure_fresh(self) -> None:
        key = (self._store.version, self._store.count())
        if key == self._key:
            return
        with self._lock:
            if key == self._key:
                return
            started = time.perf_counter()
            ids, texts, metas = self._store.get_all()
            corpus = [tokenize(contextualize(m.get("title", ""), m.get("section"), t))
                      for t, m in zip(texts, metas, strict=True)]
            self._bm25 = BM25(corpus) if corpus else None
            self._ids, self._texts, self._metas = ids, texts, metas
            self._key = key
            logger.info("Keyword index built: %d chunks in %.2fs", len(ids), time.perf_counter() - started)

    def search(self, query: str, k: int, where: dict | None = None) -> list[RetrievedChunk]:
        self._ensure_fresh()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.scores(tokens)
        results = []
        for i in sorted(scores, key=scores.get, reverse=True):
            if where and not _matches(self._metas[i], where):
                continue
            results.append(RetrievedChunk(id=self._ids[i], text=self._texts[i], metadata=self._metas[i]))
            if len(results) == k:
                break
        return results


def _matches(metadata: dict, where: dict) -> bool:
    """The subset of Chroma's `where` syntax that Athena uses."""
    for key, condition in where.items():
        if key == "$or":
            if not any(_matches(metadata, c) for c in condition):
                return False
        elif isinstance(condition, dict) and "$in" in condition:
            if metadata.get(key) not in condition["$in"]:
                return False
        elif metadata.get(key) != condition:
            return False
    return True


def files_filter(files: list[str] | None) -> dict | None:
    """Restrict retrieval to these filenames (None = whole library)."""
    if not files:
        return None
    return {"filename": files[0]} if len(files) == 1 else {"filename": {"$in": list(files)}}


def reciprocal_rank_fusion(*rankings: list[RetrievedChunk]) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
            chunks.setdefault(chunk.id, chunk)
    return [chunks[i] for i in sorted(scores, key=scores.get, reverse=True)]


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s", self.model_name)
            from . import offline

            self._model = offline.load(CrossEncoder, self.model_name, max_length=512)
        return [float(s) for s in self._model.predict([(query, t) for t in texts], batch_size=16)]


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None = None) -> Reranker:
    return Reranker(model_name or settings.reranker_model)


@dataclass
class Retrieval:
    chunks: list[RetrievedChunk]
    mode: str
    reranked: bool
    seconds: float
    # Best reranker score; None when the reranker didn't run.
    top_score: float | None = None
    timings: dict = field(default_factory=dict)

    @property
    def confident(self) -> bool:
        if not self.chunks:
            return False
        if self.top_score is None:
            return True  # no reranker: no basis for refusing
        return self.top_score >= settings.min_rerank_score


_keyword_indexes: dict[int, KeywordIndex] = {}


def keyword_index(store: VectorStore) -> KeywordIndex:
    index = _keyword_indexes.get(id(store))
    if index is None:
        index = _keyword_indexes[id(store)] = KeywordIndex(store)
    return index


def retrieve(
    query: str,
    store: VectorStore,
    k: int | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    files: list[str] | None = None,
    reranker: Reranker | None = None,
) -> Retrieval:
    k = k or settings.top_k
    mode = mode or settings.retrieval_mode
    rerank = settings.rerank if rerank is None else rerank
    where = files_filter(files)
    n = settings.candidates
    timings = {}
    started = time.perf_counter()

    dense, keyword = [], []
    if mode in ("dense", "hybrid"):
        t = time.perf_counter()
        dense = store.search(query, k=n, where=where)
        timings["dense"] = time.perf_counter() - t
    if mode in ("keyword", "hybrid"):
        t = time.perf_counter()
        keyword = keyword_index(store).search(query, k=n, where=where)
        timings["keyword"] = time.perf_counter() - t
    if mode not in ("dense", "keyword", "hybrid"):
        raise ValueError(f"Unknown retrieval mode: {mode}")

    for rank, chunk in enumerate(dense, start=1):
        chunk.dense_rank = rank
    dense_by_id = {c.id: c for c in dense}
    for rank, chunk in enumerate(keyword, start=1):
        chunk.keyword_rank = rank
        if chunk.id in dense_by_id:
            dense_by_id[chunk.id].keyword_rank = rank
            chunk.dense_rank = dense_by_id[chunk.id].dense_rank
            chunk.distance = dense_by_id[chunk.id].distance

    candidates = reciprocal_rank_fusion(dense, keyword) if mode == "hybrid" else (dense or keyword)

    top_score = None
    if rerank and candidates:
        t = time.perf_counter()
        pool = candidates[: settings.rerank_pool]
        scorer = reranker or get_reranker()
        scores = scorer.score(query, [
            contextualize(c.metadata.get("title", ""), c.metadata.get("section"), c.text) for c in pool
        ])
        for chunk, score in zip(pool, scores, strict=True):
            chunk.rerank_score = score
        candidates = sorted(pool, key=lambda c: c.rerank_score, reverse=True)
        top_score = candidates[0].rerank_score
        timings["rerank"] = time.perf_counter() - t

    return Retrieval(
        chunks=candidates[:k], mode=mode, reranked=bool(rerank and candidates),
        seconds=time.perf_counter() - started, top_score=top_score,
        timings={key: round(v, 3) for key, v in timings.items()},
    )


def sigmoid(x: float) -> float:
    """Reranker logit -> 0..1 relevance, for display."""
    return 1.0 / (1.0 + math.exp(-x))
