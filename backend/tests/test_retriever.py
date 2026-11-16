"""Hybrid retrieval: tokenizer, fusion, filters, reranking, abstention."""

import pytest

from athena.retriever import files_filter, reciprocal_rank_fusion, retrieve, tokenize
from athena.vector_store import RetrievedChunk


def add(store, chunks):
    """chunks: [(id, filename, text)]"""
    store.upsert_chunks(
        [c[0] for c in chunks],
        [c[2] for c in chunks],
        [{"filename": c[1], "filepath": f"/docs/{c[1]}", "title": c[1]} for c in chunks],
    )


class FakeReranker:
    """Scores by how many query words a text contains, minus 2."""

    def score(self, query, texts):
        words = set(tokenize(query))
        return [float(len(words & set(tokenize(t)))) - 2.0 for t in texts]


def test_tokenize_keeps_identifiers_and_their_parts():
    tokens = tokenize("What does SP 800-63B say about AAL2 in CSF 2.0?")
    assert {"800-63b", "800", "63b", "aal2", "2.0", "csf"} <= set(tokens)
    assert "what" not in tokens and "the" not in tokens


def test_rrf_rewards_agreement_between_rankings():
    a, b, c = (RetrievedChunk(id=i, text=i, metadata={}) for i in "abc")
    fused = reciprocal_rank_fusion([a, b, c], [c, a])
    assert [x.id for x in fused] == ["a", "c", "b"]


def test_keyword_search_finds_exact_numbers(store):
    add(store, [
        ("1", "pw.pdf", "Passwords shall be at least 15 characters long."),
        ("2", "pw.pdf", "Passwords should be long and memorable."),
        ("3", "other.pdf", "Cooking rice takes 15 minutes."),
    ])
    hits = retrieve("minimum 15 characters password", store, k=2, mode="keyword", rerank=False).chunks
    assert hits[0].id == "1"


def test_keyword_index_sees_new_chunks(store):
    add(store, [("1", "a.pdf", "alpha beta")])
    assert retrieve("gamma", store, mode="keyword", rerank=False).chunks == []
    add(store, [("2", "b.pdf", "gamma delta")])
    assert [c.id for c in retrieve("gamma", store, mode="keyword", rerank=False).chunks] == ["2"]


@pytest.mark.parametrize("mode", ["dense", "keyword", "hybrid"])
def test_file_filter_applies_to_every_mode(store, mode):
    add(store, [("1", "a.pdf", "risk management framework"), ("2", "b.pdf", "risk management framework")])
    hits = retrieve("risk management", store, k=5, mode=mode, rerank=False, files=["b.pdf"]).chunks
    assert {c.metadata["filename"] for c in hits} == {"b.pdf"}


def test_files_filter_shapes():
    assert files_filter(None) is None
    assert files_filter(["a.pdf"]) == {"filename": "a.pdf"}
    assert files_filter(["a.pdf", "b.pdf"]) == {"filename": {"$in": ["a.pdf", "b.pdf"]}}


def test_reranker_reorders_and_reports_top_score(store):
    add(store, [
        ("1", "a.pdf", "framework framework framework"),
        ("2", "b.pdf", "the cybersecurity framework has six functions"),
    ])
    result = retrieve("cybersecurity framework functions", store, k=2, mode="hybrid",
                      rerank=True, reranker=FakeReranker())
    assert result.chunks[0].id == "2"
    assert result.reranked and result.top_score == pytest.approx(1.0)
    assert result.confident


def test_low_reranker_score_is_not_confident(store):
    add(store, [("1", "a.pdf", "completely unrelated text")])
    result = retrieve("bitcoin price", store, mode="dense", rerank=True, reranker=FakeReranker())
    assert result.top_score == pytest.approx(-2.0)
    assert result.confident is (result.top_score >= __import__("athena.config").config.settings.min_rerank_score)


def test_unknown_mode_is_rejected(store):
    add(store, [("1", "a.pdf", "x")])
    with pytest.raises(ValueError):
        retrieve("x", store, mode="magic", rerank=False)
