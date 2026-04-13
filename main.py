"""
Legal Contract Analyzer - FastAPI application.

Exposes a REST API for:
  - POST /api/analyze         - Full contract analysis (split + RAG per clause)
  - POST /api/analyze/clause  - Analyze a single clause
  - POST /api/preview         - Preview clause splitting without analysis
  - GET  /api/kb              - List all knowledge-base documents
  - GET  /api/kb/stats        - Knowledge-base statistics
  - GET  /api/status          - Pipeline readiness and model info
  - GET  /                    - Health-check / liveness probe
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any, AsyncGenerator
import uuid

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from google.genai import types

from app.rag_core import (
    LegalRAGPipeline,
    _EMBEDDING_MODEL_NAME,
    _GEMINI_MODEL_NAME,
)
from app.schemas import (
    ClauseAnalysis,
    ClausePreviewResponse,
    ClauseResponse,
    ChatRequest,
    ChatResponse,
    ContractAnalysisResponse,
    KBDocumentResponse,
    KBStatsResponse,
    PipelineStatusResponse,
    SingleClauseRequest,
)
from app.session_store import create_chat_session_store, ChatSessionState, SessionStore
from app.settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
pipeline: LegalRAGPipeline | None = None
chat_sessions: SessionStore = create_chat_session_store(settings)


def _request_id_from(request: Request) -> str:
    """Return the request id assigned by middleware."""
    return getattr(request.state, "request_id", "unknown")


# ---------------------------------------------------------------------------
# Serialization Helpers for Redis
# ---------------------------------------------------------------------------
def dump_history(history: list[types.Content] | None) -> list[dict[str, Any]]:
    """Converts Gemini Content objects into JSON-safe dictionaries."""
    if not history:
        return []
    return [
        {
            "role": h.role,
            "parts": [{"text": p.text} for p in h.parts if p.text]
        }
        for h in history
    ]


def load_history(raw_history: list[dict[str, Any]]) -> list[types.Content]:
    """Converts JSON dictionaries back into Gemini Content objects."""
    return [
        types.Content(
            role=msg["role"],
            parts=[types.Part.from_text(text=part["text"]) for part in msg.get("parts", [])]
        )
        for msg in raw_history
    ]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the RAG pipeline and Redis connection once on startup."""
    global pipeline

    if not settings.gemini_api_key or settings.gemini_api_key == "your_api_key_here":
        logger.error("GEMINI_API_KEY is not set.")
        raise RuntimeError("Missing GEMINI_API_KEY.")

    logger.info(
        "Initialising %s in %s mode",
        settings.app_name,
        settings.environment,
    )
    pipeline = LegalRAGPipeline(gemini_api_key=settings.gemini_api_key)
    
    # Start Redis connection
    await chat_sessions.startup()
    
    app.state.started_at = time.time()
    logger.info("Pipeline ready - server accepting requests.")

    yield

    # Clean up Redis connection on shutdown
    await chat_sessions.close()
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    description=(
        "RAG-powered API that analyses Indian employment contracts for "
        "potential violations of Fundamental Rights and the Indian Contract Act."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.allow_credentials and "*" not in settings.cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_minimum_size)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach a request id and timing headers to every response."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled exception for %s %s [request_id=%s]",
            request.method,
            request.url.path,
            request_id,
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    logger.info(
        "%s %s -> %s [request_id=%s duration_ms=%.2f]",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_pipeline() -> LegalRAGPipeline:
    """Return the pipeline or raise HTTP 503 if not ready."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is not initialised. Check server logs.",
        )
    return pipeline


def _validate_pdf_upload(file: UploadFile) -> None:
    """Ensure only PDF uploads are accepted."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")


def _validate_clause_count(clauses: list[str]) -> None:
    """Reject unexpectedly large documents that would overload the service."""
    if len(clauses) > settings.max_clauses_per_document:
        raise HTTPException(
            status_code=413,
            detail=(
                "The document contains too many clauses to process safely. "
                f"Maximum supported clauses: {settings.max_clauses_per_document}."
            ),
        )


async def _extract_text_from_pdf(file: UploadFile) -> str:
    """Read text from an uploaded PDF file using pypdf."""
    try:
        content = await file.read()
        if len(content) > settings.max_pdf_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "PDF file is too large. "
                    f"Maximum supported size: {settings.max_pdf_size_bytes} bytes."
                ),
            )

        reader = PdfReader(io.BytesIO(content))
        text_parts: list[str] = []
        for page in reader.pages:
            if page_text := page.extract_text():
                text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to extract text from PDF: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Could not read the PDF file. Please ensure it is a valid text-based PDF.",
        ) from exc
    finally:
        await file.close()


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a clean 500 response for unhandled exceptions."""
    payload: dict[str, Any] = {
        "detail": "An internal server error occurred. Please try again later.",
        "request_id": _request_id_from(request),
    }
    if settings.debug:
        payload["error"] = str(exc)
    return JSONResponse(status_code=500, content=payload)


# ===========================================================================
# ROUTES - Health & Status
# ===========================================================================
@app.get("/", tags=["health"])
async def health_check() -> dict[str, str]:
    """Simple liveness probe."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.environment,
    }


@app.get(
    "/api/status",
    response_model=PipelineStatusResponse,
    tags=["health"],
    summary="Pipeline readiness & model info",
)
async def pipeline_status() -> PipelineStatusResponse:
    """Return detailed status of the RAG pipeline."""
    pipe = _require_pipeline()
    return PipelineStatusResponse(
        status="ready",
        embedding_model=_EMBEDDING_MODEL_NAME,
        llm_model=pipe.model_name or _GEMINI_MODEL_NAME,
        faiss_index_size=pipe.index.ntotal,
        kb_documents_loaded=len(pipe.kb_documents),
    )


# ===========================================================================
# ROUTES - Knowledge Base
# ===========================================================================
@app.get(
    "/api/kb",
    response_model=list[KBDocumentResponse],
    tags=["knowledge-base"],
    summary="List all KB documents",
)
async def list_kb_documents() -> list[KBDocumentResponse]:
    """Return every document in the loaded knowledge base."""
    pipe = _require_pipeline()
    return [
        KBDocumentResponse(text=doc["text"], metadata=doc["metadata"])
        for doc in pipe.kb_documents
    ]


@app.get(
    "/api/kb/stats",
    response_model=KBStatsResponse,
    tags=["knowledge-base"],
    summary="Knowledge-base statistics",
)
async def kb_stats() -> KBStatsResponse:
    """Aggregate stats about the knowledge base."""
    pipe = _require_pipeline()
    sources = Counter(doc["metadata"].get("source", "Unknown") for doc in pipe.kb_documents)
    types = Counter(doc["metadata"].get("type", "unknown") for doc in pipe.kb_documents)
    return KBStatsResponse(
        total_documents=len(pipe.kb_documents),
        sources=dict(sources),
        types=dict(types),
    )


# ===========================================================================
# ROUTES - Clause Preview
# ===========================================================================
@app.post(
    "/api/preview",
    response_model=ClausePreviewResponse,
    tags=["analysis"],
    summary="Preview clause splitting from a PDF without analysis",
)
async def preview_clauses(file: UploadFile = File(...)) -> ClausePreviewResponse:
    """Extract text from the uploaded PDF and preview the resulting clauses."""
    pipe = _require_pipeline()
    _validate_pdf_upload(file)

    contract_text = await _extract_text_from_pdf(file)
    if not contract_text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    clauses = pipe.split_clauses(contract_text)
    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any clauses from the provided text.",
        )
    _validate_clause_count(clauses)
    return ClausePreviewResponse(total_clauses=len(clauses), clauses=clauses)


# ===========================================================================
# ROUTES - Analysis
# ===========================================================================
@app.post(
    "/api/analyze",
    response_model=ContractAnalysisResponse,
    tags=["analysis"],
    summary="Analyze a full contract PDF for legal violations",
)
async def analyze_contract(file: UploadFile = File(...)) -> ContractAnalysisResponse:
    """Extract text from a PDF, split it into clauses, and analyse each one."""
    pipe = _require_pipeline()
    _validate_pdf_upload(file)

    contract_text = await _extract_text_from_pdf(file)
    if not contract_text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    clauses = pipe.split_clauses(contract_text)
    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any clauses from the provided text.",
        )
    _validate_clause_count(clauses)

    logger.info("Analysing %d clause(s) from PDF concurrently", len(clauses))
    semaphore = asyncio.Semaphore(settings.analysis_concurrency)

    async def process_clause(clause_text: str) -> ClauseResponse | None:
        async with semaphore:
            try:
                analysis = await pipe.analyze_clause(clause_text)
                return ClauseResponse(clause=clause_text, analysis=analysis)
            except Exception:
                logger.exception(
                    "Failed to analyse clause: %.60s",
                    clause_text,
                )
                return None

    tasks = [process_clause(clause) for clause in clauses]
    completed_results = await asyncio.gather(*tasks)
    results = [res for res in completed_results if res is not None]

    if not results:
        raise HTTPException(
            status_code=500,
            detail="Analysis failed for all clauses. Please check server logs.",
        )

    if len(results) < len(clauses):
        logger.warning(
            "Partial clause analysis completed: %d/%d succeeded",
            len(results),
            len(clauses),
        )

    # ---------------------------------------------------------
    # NEW PDF FLOW: Build a fresh session state for Redis
    # ---------------------------------------------------------
    session_id = str(uuid.uuid4())
    system_instruction = (
        "You are an expert Indian legal assistant. The user has uploaded an "
        "employment contract. Answer any questions about the contract. "
        f"Here is the contract text:\n\n{contract_text}"
    )

    initial_state: ChatSessionState = {
        "model_name": pipe.model_name or _GEMINI_MODEL_NAME,
        "chat_config": {"system_instruction": system_instruction},
        "history": [],
        "created_at": time.time(),
        "updated_at": time.time()
    }

    await chat_sessions.set(session_id, initial_state)
    
    return ContractAnalysisResponse(session_id=session_id, results=results)


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["analysis"],
    summary="Chat with an analyzed contract",
)
async def chat_with_contract(request: ChatRequest) -> ChatResponse:
    """Send a question within an active document chat session."""
    pipe = _require_pipeline()
    
    # 1. Fetch JSON state from Redis
    state = await chat_sessions.get(request.session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. It may have expired or not exist.",
        )

    try:
        # 2. Reconstruct the Gemini Chat Session from the saved JSON
        history = load_history(state["history"])
        config = types.GenerateContentConfig(
            system_instruction=state["chat_config"]["system_instruction"],
            temperature=0.3,
        )
        chat = pipe.client.aio.chats.create(
            model=state["model_name"],
            config=config,
            history=history
        )

        # 3. Process the new user message
        response = await chat.send_message_async(request.message)

        # 4. Serialize the updated history back to Redis
        state["history"] = dump_history(chat.history)
        state["updated_at"] = time.time()
        await chat_sessions.set(request.session_id, state)

        return ChatResponse(response=response.text)
    except Exception as exc:
        logger.error("Failed to process chat message: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to process the chat message.",
        ) from exc


@app.post(
    "/api/analyze/clause",
    response_model=ClauseResponse,
    tags=["analysis"],
    summary="Analyze a single clause",
)
async def analyze_single_clause(request: SingleClauseRequest) -> ClauseResponse:
    """Analyze one clause directly without splitting."""
    pipe = _require_pipeline()

    try:
        analysis: ClauseAnalysis = await pipe.analyze_clause(request.clause_text)
        return ClauseResponse(clause=request.clause_text, analysis=analysis)
    except Exception as exc:
        logger.error("Failed to analyse clause: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Analysis failed for the clause.",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )