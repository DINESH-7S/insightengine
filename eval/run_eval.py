"""Compare InsightEngine's retrieval strategies on a labelled question set.

Usage:
    python -m eval.run_eval --docs path/to/pdfs --questions eval/questions.jsonl
    python -m eval.run_eval --docs path/to/pdfs --questions q.jsonl --embedder minilm --top-k 5 --md results.md

Each line of the questions file is JSON:
    {"question": "...", "doc": "report.pdf", "pages": [12, 13]}
A hit means at least one of the top-k passages comes from `doc` and one of `pages`.

Reported per strategy: hit@k, MRR, and mean retrieval latency. Retrieval only; no LLM is called.
Use --embedder hash for an offline smoke test. Its numbers mean nothing; do not report them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient  # noqa: E402

import vector_db  # noqa: E402
from parser import DEFAULT_ENCODER, MODEL_DIMENSIONS, DocumentParser, EmbeddingGenerator  # noqa: E402
from schemas import TextSegment  # noqa: E402
from vector_db import QdrantEngine  # noqa: E402

STRATEGIES = ["dense", "mmr", "hybrid", "reranker"]  # multi_query needs an LLM, so it is excluded
HASH_DIM = 384


def hash_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words hashing embedder, for offline smoke tests only."""
    out = []
    for text in texts:
        vec = [0.0] * HASH_DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % HASH_DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        out.append([v / norm for v in vec])
    return out


def load_segments(docs_dir: Path) -> list[TextSegment]:
    segments: list[TextSegment] = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            segments.extend(DocumentParser.parse_pdf(str(path), doc_id=path.name))
        elif path.suffix.lower() == ".txt":
            for i, chunk in enumerate(c for c in path.read_text(encoding="utf-8").split("\n\n") if c.strip()):
                segments.append(TextSegment(content=chunk.strip(), doc_id=path.name, page_num=1, seq_index=i))
    return segments


def load_questions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if not {"question", "doc", "pages"} <= r.keys():
            raise ValueError(f"Bad question row (needs question, doc, pages): {r}")
    return rows


def evaluate(engine: QdrantEngine, questions: list[dict], embed, strategy: str, top_k: int) -> dict:
    hits, rr, latencies = 0, 0.0, []
    for q in questions:
        vec = embed([q["question"]])[0]
        start = time.perf_counter()
        result = engine.search(q["question"], vec, strategy=strategy, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        rank = next(
            (i for i, p in enumerate(result.passages, 1) if p.doc_id == q["doc"] and p.page_num in q["pages"]),
            None,
        )
        if rank:
            hits += 1
            rr += 1.0 / rank
    n = len(questions)
    return {
        "strategy": strategy,
        "hit_rate": hits / n,
        "mrr": rr / n,
        "latency_ms": statistics.mean(latencies),
    }


def to_markdown(rows: list[dict], n_questions: int, top_k: int, embedder: str) -> str:
    lines = [
        f"{n_questions} questions, top-k = {top_k}, embedder = {embedder}",
        "",
        f"| Strategy | Hit@{top_k} | MRR | Mean latency (ms) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['strategy']} | {r['hit_rate']:.2f} | {r['mrr']:.3f} | {r['latency_ms']:.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embedder", choices=["minilm", "hash"], default="minilm")
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--md", type=Path, help="Also write the results table to this file")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    segments = load_segments(args.docs)
    if not segments:
        print("No text found in --docs.", file=sys.stderr)
        return 1

    if args.embedder == "hash":
        embed, dim = hash_embed, HASH_DIM
        print("WARNING: hash embedder is a smoke test only; do not publish these numbers.", file=sys.stderr)
    else:
        def embed(texts):
            return EmbeddingGenerator.embed_texts(texts, model_name=DEFAULT_ENCODER)
        dim = MODEL_DIMENSIONS[DEFAULT_ENCODER]

    vector_db._qdrant_client_singleton = QdrantClient(location=":memory:")
    vector_db._active_collections = set()
    engine = QdrantEngine(collection_name="eval", vector_dim=dim)
    engine.store_segments(segments, embed([s.content for s in segments]))

    rows = [evaluate(engine, questions, embed, s, args.top_k) for s in args.strategies]
    table = to_markdown(rows, len(questions), args.top_k, args.embedder)
    print(table)
    if args.md:
        args.md.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
