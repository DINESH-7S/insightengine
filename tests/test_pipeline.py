from langchain_core.documents import Document

import pipeline
from pipeline import format_passages, generate_queries


def test_generate_queries_without_key_uses_templates(monkeypatch):
    monkeypatch.setattr(pipeline, "groq_key", None)
    out = generate_queries("what is rag")
    assert out[0] == "what is rag" and len(out) == 3


def test_generate_queries_survives_llm_failure(monkeypatch):
    monkeypatch.setattr(pipeline, "groq_key", "fake")

    class Boom:
        def __init__(self, *a, **k): ...
        def invoke(self, *a, **k): raise RuntimeError("network down")

    monkeypatch.setattr(pipeline, "ChatGroq", Boom)
    out = generate_queries("q")
    assert out[0] == "q" and len(out) == 3


def test_generate_queries_parses_fenced_json(monkeypatch):
    monkeypatch.setattr(pipeline, "groq_key", "fake")

    class Reply:
        content = '```json\n["a", "b", "c"]\n```'

    class Fake:
        def __init__(self, *a, **k): ...
        def invoke(self, *a, **k): return Reply()

    monkeypatch.setattr(pipeline, "ChatGroq", Fake)
    assert generate_queries("q") == ["q", "a", "b", "c"]


def test_format_passages_includes_citation_header():
    docs = [Document(page_content="text", metadata={"idx": 1, "source": "r.pdf", "page": 7})]
    out = format_passages(docs)
    assert "r.pdf" in out and "Page: 7" in out and "text" in out


def test_ddg_fallback_returns_placeholder_on_network_error(monkeypatch):
    def boom(*a, **k): raise OSError("offline")
    monkeypatch.setattr(pipeline.urllib.request, "urlopen", boom)
    out = pipeline.execute_duckduckgo_fallback("anything")
    assert out[0].relevance_score == 0.0
