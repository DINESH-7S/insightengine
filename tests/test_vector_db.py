import numpy as np
import pytest
from qdrant_client import QdrantClient

import vector_db
from schemas import TextSegment
from vector_db import LexicalScorerBM25, QdrantEngine, make_point_id


@pytest.fixture()
def engine(monkeypatch):
    """A QdrantEngine backed by an in-memory store, isolated per test."""
    client = QdrantClient(location=":memory:")
    monkeypatch.setattr(vector_db, "_qdrant_client_singleton", client)
    monkeypatch.setattr(vector_db, "_active_collections", set())
    monkeypatch.delenv("QDRANT_URL", raising=False)
    return QdrantEngine(collection_name="test", vector_dim=3)


def seg(text, page=1, idx=0, doc="a.pdf"):
    return TextSegment(content=text, doc_id=doc, page_num=page, seq_index=idx)


# ---- BM25 -------------------------------------------------------------
def test_bm25_ranks_matching_doc_first():
    bm25 = LexicalScorerBM25()
    bm25.fit(["revenue grew in the third quarter", "the cat sat on the mat", "quarterly cost report"])
    scores = bm25.score("revenue quarter")
    assert scores.index(max(scores)) == 0


def test_bm25_ignores_punctuation_and_case():
    bm25 = LexicalScorerBM25()
    bm25.fit(["Revenue, up 12%.", "unrelated text here"])
    assert bm25.score("revenue")[0] > 0


def test_bm25_unknown_terms_score_zero():
    bm25 = LexicalScorerBM25()
    bm25.fit(["alpha beta", "gamma delta"])
    assert bm25.score("zzz") == [0.0, 0.0]


def test_bm25_empty_corpus_is_safe():
    bm25 = LexicalScorerBM25()
    bm25.fit([])
    assert bm25.score("anything") == []


# ---- IDs ---------------------------------------------------------------
def test_point_id_is_deterministic_and_content_sensitive():
    assert make_point_id(seg("hello")) == make_point_id(seg("hello"))
    assert make_point_id(seg("hello")) != make_point_id(seg("hello!"))
    assert make_point_id(seg("hello", page=2)) != make_point_id(seg("hello", page=1))


def test_reingesting_same_segments_does_not_duplicate(engine):
    segs = [seg("one", idx=0), seg("two", idx=1)]
    vecs = [[1, 0, 0], [0, 1, 0]]
    engine.store_segments(segs, vecs)
    engine.store_segments(segs, vecs)
    assert engine.client.count("test").count == 2


def test_store_segments_rejects_length_mismatch(engine):
    with pytest.raises(ValueError):
        engine.store_segments([seg("x")], [])


# ---- Search strategies -------------------------------------------------
@pytest.fixture()
def loaded(engine):
    segs = [
        seg("revenue grew strongly this quarter", page=1, idx=0),
        seg("revenue grew strongly this quarter, again", page=2, idx=1),  # near-duplicate
        seg("the staff picnic is on friday", page=3, idx=2),
    ]
    vecs = [[1.0, 0.0, 0.0], [0.99, 0.05, 0.0], [0.0, 0.0, 1.0]]
    engine.store_segments(segs, vecs)
    return engine


def test_dense_returns_closest_first(loaded):
    out = loaded.search_dense([1.0, 0.0, 0.0], limit=3)
    assert out[0].page_num in (1, 2)
    assert out[-1].page_num == 3


def test_mmr_prefers_diverse_results(loaded):
    out = loaded.search_mmr([1.0, 0.0, 0.0], limit=2, diversity_factor=0.4)
    pages = {p.page_num for p in out}
    assert 3 in pages, "MMR should pull in the dissimilar passage over a near-duplicate"


def test_hybrid_uses_lexical_signal(loaded):
    out = loaded.search_hybrid("picnic friday", [0.5, 0.0, 0.5], limit=1)
    assert out[0].page_num == 3


def test_multi_query_dedupes_by_passage(loaded):
    out = loaded.search_multi_query([[1.0, 0.0, 0.0], [0.99, 0.05, 0.0]], limit=5)
    keys = [(p.doc_id, p.page_num, p.seq_index) for p in out]
    assert len(keys) == len(set(keys))


def test_search_dispatch_and_unknown_strategy_falls_back_to_dense(loaded):
    res = loaded.search("q", [1.0, 0.0, 0.0], strategy="not-a-strategy", top_k=2)
    assert len(res.passages) == 2


def test_rerank_falls_back_to_dense_when_model_unavailable(loaded, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # forces ImportError
    out = loaded.search_rerank("revenue", [1.0, 0.0, 0.0], limit=2)
    assert len(out) == 2


def test_reset_collection_empties_store(loaded):
    loaded.reset_collection()
    assert loaded.client.count("test").count == 0
    assert loaded.list_documents() == []


def test_list_documents_is_sorted_and_unique(engine):
    engine.store_segments([seg("a", doc="b.pdf", idx=0), seg("b", doc="a.pdf", idx=1), seg("c", doc="a.pdf", idx=2)],
                          [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert engine.list_documents() == ["a.pdf", "b.pdf"]


def test_mmr_score_math_is_finite(loaded):
    out = loaded.search_mmr([1.0, 0.0, 0.0], limit=3)
    assert all(np.isfinite(p.relevance_score) for p in out)
