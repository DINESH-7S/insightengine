import importlib

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import vector_db


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(vector_db, "_qdrant_client_singleton", QdrantClient(location=":memory:"))
    monkeypatch.setattr(vector_db, "_active_collections", set())
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    import server
    importlib.reload(server)
    return TestClient(server.app)


def test_wipe_requires_token(client):
    assert client.delete("/api/database/wipe").status_code == 401
    assert client.delete("/api/database/wipe", headers={"X-Admin-Token": "nope"}).status_code == 401


def test_wipe_succeeds_with_token(client):
    r = client.delete("/api/database/wipe", headers={"X-Admin-Token": "s3cret"})
    assert r.status_code == 200


def test_ingest_rejects_non_pdf(client):
    r = client.post("/api/documents/ingest", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_ingest_rejects_oversized_upload(client, monkeypatch):
    import server
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/api/documents/ingest", files={"file": ("a.pdf", b"x" * 100, "application/pdf")})
    assert r.status_code == 413


def test_unknown_task_is_404(client):
    assert client.get("/api/documents/tasks/does-not-exist").status_code == 404


def test_cors_blocks_unlisted_origin(client):
    r = client.options(
        "/api/documents",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in r.headers
