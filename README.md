# InsightEngine

[![CI](https://github.com/DINESH-7S/insightengine/actions/workflows/ci.yml/badge.svg)](https://github.com/DINESH-7S/insightengine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A RAG app for asking questions across your PDFs and getting answers that cite the page they came from. I built it to see how retrieval behaves outside a notebook: ingestion, chunking, retrieval, and a way to measure the strategies against each other. It combines a FastAPI backend, a Qdrant vector store and a Streamlit interface, and lets you compare five retrieval strategies side by side.

![Chat view](docs/images/chat.png)

## Features

- **Cited answers**: every answer lists its source passages with document, page number and relevance score.
- **Five retrieval strategies**: dense vector, MMR diversity, lexical + dense hybrid (BM25), multi-query expansion, and cross-encoder reranking.
- **Corrective RAG (CRAG)**: falls back to DuckDuckGo search when local match confidence is low.
- **Async ingestion**: PDFs are parsed, chunked and indexed in the background with live progress.
- **Strategy evaluator**: run one query through several strategies and compare the passages they return.

## Architecture

![Architecture](docs/images/architecture.svg)

| File | Role |
|---|---|
| `server.py` | FastAPI endpoints for ingestion, querying and database wipe |
| `dashboard.py` | Streamlit UI |
| `pipeline.py` | RAG coordinator, custom retriever, query expansion |
| `parser.py` | PDF parsing and embedding generation |
| `vector_db.py` | Qdrant storage layer |
| `schemas.py` | Pydantic request and response models |
| `eval/run_eval.py` | Retrieval evaluation: hit rate, MRR and latency per strategy |
| `tests/` | pytest suite (storage, search strategies, API, pipeline) |

## Screenshots

| Upload and ingest | Strategy evaluator |
|---|---|
| ![Ingest](docs/images/ingest.png) | ![Strategies](docs/images/strategies.png) |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
cp .env.example .env          # then add your Groq API key
```

Run both services (Windows):

```bash
run_app.bat
```

Or manually, in two terminals:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501.

## Configuration

Copy `.env.example` to `.env`. Only `GROQ_API_KEY` is required.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | LLM for answers and query expansion |
| `QDRANT_URL` | embedded store in `./qdrant_storage` | Use a Qdrant server instead |
| `ADMIN_TOKEN` | unset | When set, `DELETE /api/database/wipe` needs an `X-Admin-Token` header |
| `CORS_ORIGINS` | local Streamlit origins | Comma-separated allowed origins |
| `MAX_UPLOAD_MB` | 50 | Upload size limit |

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests use an in-memory Qdrant and need no API key or model download. CI runs them on Python 3.11 and 3.12.

## Evaluating retrieval

Which strategy works best depends on your documents, so measure it on yours. Write a few questions with the page where each answer lives (see `eval/questions.example.jsonl`), then:

```bash
python -m eval.run_eval --docs path/to/pdfs --questions my_questions.jsonl --top-k 5 --md results.md
```

It prints hit@k, MRR and mean retrieval latency for dense, MMR, hybrid and reranker. Multi-query is left out because it needs an LLM call per question. `--embedder hash` runs offline as a smoke test; its numbers are meaningless and should not be published.

## Limitations

- Hybrid search runs BM25 over the dense candidate pool (top `4 * k`), so it re-ranks dense results and cannot surface passages the dense search missed.
- When CRAG triggers, the web snippets replace the local passages rather than adding to them, and the confidence thresholds are hand-set per strategy, not calibrated.
- Ingestion progress is kept in memory, so it is lost on restart and works with a single server process only.
- PDFs only. Scanned PDFs need OCR first.
- There is no user authentication. Run it locally or behind your own gateway.

## Stack

Python, FastAPI, Streamlit, LangChain, Qdrant, pypdf, sentence-transformers (all-MiniLM-L6-v2, bge-small), Groq (Llama 3).

## Design

The UI follows [DESIGN.md](DESIGN.md), adapted from the Linear design language (VoltAgent/awesome-design-md) with a single emerald accent.

## Author

Dinesh N, B.Tech AI and Data Science. [Resume (PDF)](docs/Dinesh_N_Resume.pdf) | [GitHub](https://github.com/DINESH-7S) | [LinkedIn](https://linkedin.com/in/dineshnleo)
