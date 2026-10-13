import logging
import time
import uuid
from pathlib import Path
from typing import Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, status
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.services.ingestion import chunk_document, sanitize_filename, SUPPORTED_EXTENSIONS
from app.rag.vectorstore import upsert_chunks
from app.rag.workflow import run_agentic_workflow
from app.services.audit import record_audit_entry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000, description="Employee question")


class IngestionResponse(BaseModel):
    message: str
    file: str
    document_id: str
    chunks: int
    dimension: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    source_used: str
    kb_grade: str | None = None
    web_grade: str | None = None
    retry_count: int = 0
    trace: list[str] = []
    rewritten_query: str | None = None
    latency_seconds: float = 0.0


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """
    Execute the Agentic RAG decision pipeline for an employee query.
    Performs audit logging of the outcome, evidence grades, and latency.
    """
    clean_question = payload.question.strip()
    if len(clean_question) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must contain at least 2 characters.",
        )

    req_id = f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    try:
        final_state = run_agentic_workflow(clean_question)
    except Exception as exc:
        logger.error("Agentic workflow execution error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during agentic reasoning.",
        ) from exc

    duration = time.time() - start_time
    trace = final_state.get("execution_trace", [])
    source = final_state.get("source_used", "unknown")
    kb_grade = final_state.get("kb_grade")
    web_grade = final_state.get("web_grade")
    retries = final_state.get("retry_count", 0)
    answer = final_state.get("answer", "")
    citations = final_state.get("citations", [])

    # Record audit log
    try:
        record_audit_entry(
            request_id=req_id,
            question=clean_question,
            stages=trace,
            source_used=source,
            kb_grade=kb_grade,
            web_grade=web_grade,
            retry_count=retries,
            duration_seconds=duration,
            final_outcome=source,
        )
    except Exception as exc:
        logger.error("Audit logging failed: %s", exc)

    return ChatResponse(
        answer=answer,
        citations=citations,
        source_used=source,
        kb_grade=kb_grade,
        web_grade=web_grade,
        retry_count=retries,
        trace=trace,
        rewritten_query=final_state.get("search_query"),
        latency_seconds=round(duration, 3),
    )


async def _handle_document_ingestion(
    file: UploadFile,
    x_admin_key: str,
) -> IngestionResponse:
    # 1. Admin authorization check
    if not x_admin_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid admin credentials.",
        )

    # 2. File extension check
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Allowed formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # 3. Read content in-memory with file size limit enforcement
    content_bytes = await file.read()
    if len(content_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(content_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum size limit of {settings.max_upload_size_bytes // (1024 * 1024)}MB.",
        )

    # 4. Parse, chunk, and embed
    clean_name = sanitize_filename(filename)
    try:
        chunks = chunk_document(
            filename=clean_name,
            content_bytes=content_bytes,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Document parsing error for '%s': %s", clean_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse document content.",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid text chunks could be extracted from document.",
        )

    # 5. Upsert to Pinecone
    try:
        upserted_count = upsert_chunks(chunks)
    except Exception as exc:
        logger.error("Pinecone vector indexing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vector store indexing failed: {str(exc)}",
        ) from exc

    doc_id = chunks[0].metadata.get("document_id", "unknown")
    return IngestionResponse(
        message="Document successfully processed and indexed into Pinecone.",
        file=clean_name,
        document_id=doc_id,
        chunks=upserted_count,
        dimension=settings.embedding_dimension,
    )


@router.post("/documents/ingest", response_model=IngestionResponse)
async def ingest_document(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default="", alias="x-admin-key"),
):
    return await _handle_document_ingestion(file, x_admin_key)


@router.post("/ingest", response_model=IngestionResponse, include_in_schema=False)
async def ingest_document_alias(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default="", alias="x-admin-key"),
):
    """Compatibility alias for the frontend upload modal."""
    return await _handle_document_ingestion(file, x_admin_key)
