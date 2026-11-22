import asyncio
import hmac
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from parser import DEFAULT_ENCODER, MODEL_DIMENSIONS, DocumentParser, EmbeddingGenerator
from pipeline import RAGCoordinator
from schemas import QueryRequest, QueryResponse, TaskStatusResponse
from vector_db import QdrantEngine

load_dotenv()

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGBackend")

app = FastAPI(title="Insight RAG API Service")

# CORS: local Streamlit UI by default. Set CORS_ORIGINS (comma-separated) to change it.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://127.0.0.1:8501,http://localhost:8501").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)

# Destructive endpoints need X-Admin-Token when ADMIN_TOKEN is set.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024


def require_admin(token: Optional[str]) -> None:
    if ADMIN_TOKEN and not (token and hmac.compare_digest(token, ADMIN_TOKEN)):
        raise HTTPException(status_code=401, detail="Missing or invalid admin token.")

# In-memory background task status store
tasks_status: Dict[str, Dict[str, Any]] = {}
# Concurrency lock to prevent thread/GPU starvation during heavy embedding generation
processing_semaphore = asyncio.Semaphore(2)

async def background_ingest_worker(
    task_id: str,
    file_path: str,
    doc_id: str,
    embedding_model: str
):
    """Background task processing segment extraction and vector database ingestion."""
    async with processing_semaphore:
        try:
            tasks_status[task_id].update({
                "status": "extracting",
                "progress": 25.0,
                "message": "Extracting text passages from document..."
            })

            # Parse PDF
            segments = DocumentParser.parse_pdf(file_path, doc_id=doc_id)

            if not segments:
                tasks_status[task_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "message": "No text content found in document.",
                    "processed_count": 0
                })
                return

            tasks_status[task_id].update({
                "status": "vectorizing",
                "progress": 60.0,
                "message": f"Generating vector embeddings using {embedding_model}..."
            })

            # Embed and Index
            texts = [s.content for s in segments]
            # Since generating embeddings is a blocking call (running on CPU/GPU), run in a separate executor to prevent blocking the event loop
            loop = asyncio.get_running_loop()
            vectors = await loop.run_in_executor(None, lambda: EmbeddingGenerator.embed_texts(texts, model_name=embedding_model))

            dim = MODEL_DIMENSIONS.get(embedding_model, 384)
            db_engine = QdrantEngine(vector_dim=dim)
            db_engine.store_segments(segments, vectors)

            tasks_status[task_id].update({
                "status": "completed",
                "progress": 100.0,
                "message": f"Successfully ingested {len(segments)} segments.",
                "processed_count": len(segments)
            })
        except Exception as e:
            logger.error(f"Failed to ingest document task {task_id}: {str(e)}")
            tasks_status[task_id].update({
                "status": "failed",
                "progress": 100.0,
                "message": "Internal processing failed.",
                "error": str(e)
            })
        finally:
            # Remove temporary uploaded file
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Could not remove %s", file_path, exc_info=True)

@app.post("/api/documents/ingest", response_model=TaskStatusResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_model: str = Form(DEFAULT_ENCODER)
):
    """Endpoint to upload a PDF file and trigger background indexing pipeline."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if embedding_model not in MODEL_DIMENSIONS:
        embedding_model = DEFAULT_ENCODER

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    task_id = str(uuid.uuid4())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (".", "-", "_"))
    temp_path = os.path.join("uploads", f"{task_id}-{safe_filename}")

    # Write file to disk temporarily
    written = 0
    with open(temp_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                f.close()
                os.unlink(temp_path)
                raise HTTPException(status_code=413, detail="File too large.")
            f.write(chunk)

    tasks_status[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Enqueued for ingestion...",
        "processed_count": 0,
        "error": None
    }

    # Queue background task
    background_tasks.add_task(
        background_ingest_worker,
        task_id=task_id,
        file_path=temp_path,
        doc_id=file.filename,
        embedding_model=embedding_model
    )

    return tasks_status[task_id]

@app.get("/api/documents/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Retrieve current processing progress of an ingestion task."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found.")
    return tasks_status[task_id]

@app.post("/api/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    """Execute synchronous RAG question answering pipeline."""
    coordinator = RAGCoordinator()
    response = coordinator.process_query(
        question=request.question,
        strategy=request.strategy,
        top_k=request.top_k,
        embedding_model=request.embedding_model,
        chat_history=request.chat_history,
        enable_crag=request.enable_crag,
        answer_length=request.answer_length
    )
    return response

@app.get("/api/documents", response_model=List[str])
async def list_ingested_documents():
    """Retrieve list of unique document sources stored in the system."""
    db_engine = QdrantEngine()
    return db_engine.list_documents()

@app.delete("/api/database/wipe")
async def wipe_database(
    embedding_model: str = DEFAULT_ENCODER,
    x_admin_token: Optional[str] = Header(default=None),
):
    """Clears uploaded logs, deletes temporary files, and wipes vector store."""
    require_admin(x_admin_token)
    try:
        # Wipe Qdrant indexes
        dim = MODEL_DIMENSIONS.get(embedding_model, 384)
        db_engine = QdrantEngine(vector_dim=dim)
        db_engine.reset_collection()

        # Clean local storage
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                filepath = os.path.join(uploads_dir, filename)
                try:
                    if os.path.isfile(filepath) or os.path.islink(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                except OSError:
                    logger.warning("Could not remove %s", filepath, exc_info=True)

        # Reset background statuses
        tasks_status.clear()

        return {"status": "success", "message": "Database and temporary storage successfully cleared."}
    except Exception:
        logger.exception("Database wipe failed")
        raise HTTPException(status_code=500, detail="Failed to reset database.") from None
