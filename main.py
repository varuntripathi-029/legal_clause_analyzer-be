"""
Legal Contract Analyzer - FastAPI application.

Exposes a REST API for:
  - POST /api/analyze           - Full contract analysis from PDF (split + RAG per clause)
  - POST /api/analyze/clause    - Analyze a single clause
  - POST /api/analyze/text      - Analyze confirmed/edited text (OCR workflow)
  - POST /api/upload            - Upload document + extract text (OCR if needed)
  - POST /api/preview           - Preview clause splitting from PDF
  - POST /api/preview/text      - Preview clause splitting from confirmed text
  - GET  /api/kb                - List all knowledge-base documents
  - GET  /api/kb/stats          - Knowledge-base statistics
  - GET  /api/status            - Pipeline readiness and model info
  - GET  /                      - Health-check / liveness probe
"""

from __future__ import annotations

import asyncio
import io
import json
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
from fastapi.responses import JSONResponse, StreamingResponse
from pypdf import PdfReader
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware


from app.rag_core import (
    LegalRAGPipeline,
    _EMBEDDING_MODEL_NAME,
    _GENERATIVE_MODEL_NAME,
)
from app.schemas import (
    AnalyzeFromTextRequest,
    ClauseAnalysis,
    ClausePreviewResponse,
    ClauseResponse,
    ChatRequest,
    ChatResponse,
    ContractAnalysisResponse,
    DocumentUploadResponse,
    KBDocumentResponse,
    KBStatsResponse,
    PipelineStatusResponse,
    SingleClauseRequest,
)
from ingestion import extract_text_from_upload, DocumentType, SUPPORTED_EXTENSIONS
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
# Serialization Helpers for Redis (plain dict format for Groq messages API)
# ---------------------------------------------------------------------------
def dump_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Pass-through: history is already a list of {role, content} dicts."""
    return list(history) if history else []


def load_history(raw_history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Pass-through: history is already a list of {role, content} dicts."""
    return list(raw_history) if raw_history else []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the RAG pipeline and Redis connection once on startup."""
    global pipeline

    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        logger.error("GROQ_API_KEY is not set.")
        raise RuntimeError("Missing GROQ_API_KEY.")

    logger.info(
        "Initialising %s in %s mode",
        settings.app_name,
        settings.environment,
    )
    pipeline = LegalRAGPipeline(groq_api_key=settings.groq_api_key)
    await pipeline.startup()
    
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
    allow_origin_regex=r"https://.*\.hf\.space|http://(localhost|127\.0\.0\.1)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
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


def _validate_upload(file: UploadFile) -> None:
    """Ensure the uploaded file has a supported extension."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )


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
@app.head("/", tags=["health"])
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
        llm_model=pipe.model_name or _GENERATIVE_MODEL_NAME,
        faiss_index_size=pipe.index.ntotal if pipe.index else 0,
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

    cleaned_text = pipe.preprocess_contract_text(contract_text)
    clauses = pipe.split_clauses(cleaned_text)
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
    tags=["analysis"],
    summary="Analyze a full contract PDF for legal violations and stream results",
)
async def analyze_contract(file: UploadFile = File(...)) -> StreamingResponse:
    """Extract text from a PDF, split it into clauses, and analyse each one."""
    pipe = _require_pipeline()
    _validate_pdf_upload(file)

    contract_text = await _extract_text_from_pdf(file)
    if not contract_text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    cleaned_text = pipe.preprocess_contract_text(contract_text)
    clauses = pipe.split_clauses(cleaned_text)
    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any clauses from the provided text.",
        )
    _validate_clause_count(clauses)

    logger.info("Analysing %d clause(s) from PDF sequentially via SSE with context", len(clauses))
    
    # ---------------------------------------------------------
    # NEW PDF FLOW: Build a fresh session state for Redis first
    # ---------------------------------------------------------
    session_id = str(uuid.uuid4())
    system_instruction = (
        "You are an expert Indian legal assistant. The user has uploaded an "
        "employment contract. Answer any questions about the contract. "
        "Use the retrieved legal context provided with each question to "
        "ground your answers in specific laws and provisions."
    )

    initial_state: ChatSessionState = {
        "model_name": pipe.model_name or _GENERATIVE_MODEL_NAME,
        "chat_config": {"system_instruction": system_instruction},
        "history": [],
        "contract_text": contract_text,
        "created_at": time.time(),
        "updated_at": time.time()
    }

    await chat_sessions.set(session_id, initial_state)

    async def sse_generator() -> AsyncGenerator[str, None]:
        # Yield initialization event
        init_data = {"type": "init", "session_id": session_id, "total_clauses": len(clauses)}
        yield f"data: {json.dumps(init_data)}\n\n"

        # 1. Generate contract summary first
        contract_summary = await pipe.generate_contract_summary(contract_text)
        yield f"data: {json.dumps({'type': 'summary', 'text': contract_summary})}\n\n"

        previous_analyses: list[dict] = []

        # 2. Process clauses sequentially with context
        for i, clause in enumerate(clauses):
            try:
                analysis_dict = await pipe.analyze_clause(
                    clause_text=clause,
                    clause_index=i,
                    total_clauses=len(clauses),
                    contract_summary=contract_summary,
                    previous_analyses=previous_analyses,
                    all_clauses=clauses,
                )
                previous_analyses.append(analysis_dict)

                clause_data = {
                    "type": "clause",
                    "index": i,
                    "clause": clause,
                    "analysis": analysis_dict,
                }
                yield f"data: {json.dumps(clause_data)}\n\n"
            except Exception:
                logger.exception("Failed to analyse clause %d: %.60s", i, clause)

        # 3. Generate executive summary across all analyzed clauses
        try:
            exec_summary_dict = await pipe.generate_executive_summary(
                contract_summary=contract_summary,
                clause_analyses=previous_analyses,
                all_clauses=clauses,
            )
            yield f"data: {json.dumps({'type': 'executive_summary', 'summary': exec_summary_dict})}\n\n"
        except Exception:
            logger.exception("Failed to generate executive summary")

        # Yield completion event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post(
    "/api/chat",
    tags=["analysis"],
    summary="Chat with an analyzed contract (SSE streaming)",
)
async def chat_with_contract(request: ChatRequest) -> StreamingResponse:
    """Stream a chat response within an active document chat session via SSE."""
    pipe = _require_pipeline()

    # 1. Fetch JSON state from Redis
    state = await chat_sessions.get(request.session_id)
    if state is None:
        async def error_stream() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type': 'error', 'text': 'Session expired. Please start a new analysis.'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    history = load_history(state["history"])
    system_instruction = state["chat_config"]["system_instruction"]

    async def sse_chat_generator() -> AsyncGenerator[str, None]:
        full_response = ""
        try:
            # Stream tokens from Groq LPU via RAG-augmented context
            async for token in pipe.stream_chat(
                query=request.message,
                history=history,
                system_instruction=system_instruction,
            ):
                full_response += token
                yield f"data: {json.dumps({'type': 'chunk', 'text': token})}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # Persist updated history back to Redis
            state["history"] = dump_history(history + [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": full_response},
            ])
            state["updated_at"] = time.time()
            await chat_sessions.set(request.session_id, state)

        except Exception as exc:
            logger.error("Failed to stream chat message: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'text': 'Failed to process the chat message.'})}\n\n"

    return StreamingResponse(sse_chat_generator(), media_type="text/event-stream")


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
        analysis_dict = await pipe.analyze_clause(
            clause_text=request.clause_text,
            clause_index=0,
            total_clauses=1,
            contract_summary=request.clause_text,
            previous_analyses=[],
            all_clauses=[request.clause_text],
        )
        analysis = ClauseAnalysis.model_validate(analysis_dict)
        return ClauseResponse(clause=request.clause_text, analysis=analysis)
    except Exception as exc:
        logger.error("Failed to analyse clause: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Analysis failed for the clause.",
        ) from exc


# ===========================================================================
# ROUTES - OCR Upload & Text-Based Analysis (new two-step workflow)
# ===========================================================================
@app.post(
    "/api/upload",
    response_model=DocumentUploadResponse,
    tags=["ingestion"],
    summary="Upload a document and extract text (OCR if needed)",
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Accept a PDF, image, or text file.  Extract text (with OCR for
    scanned documents / images) and return it for user review before
    analysis."""
    _validate_upload(file)

    try:
        content = await file.read()
    finally:
        await file.close()

    # Size guard
    suffix = Path(file.filename or "").suffix.lower()
    if suffix == ".pdf" and len(content) > settings.max_pdf_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "PDF file is too large. "
                f"Maximum supported size: {settings.max_pdf_size_bytes} bytes."
            ),
        )
    if suffix in {".jpg", ".jpeg", ".png"} and len(content) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "Image file is too large. "
                f"Maximum supported size: {settings.max_image_size_bytes} bytes."
            ),
        )

    try:
        result = extract_text_from_upload(
            file_bytes=content,
            filename=file.filename or "upload",
            ocr_confidence_threshold=settings.ocr_confidence_warning_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document extraction failed for %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail="Failed to process the uploaded document.",
        ) from exc

    if not result.extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text could be extracted from the uploaded document.",
        )

    # Convert OCR confidence from 0.0-1.0 float to 0-100 int for the API
    ocr_conf_int: int | None = None
    if result.ocr_confidence is not None:
        ocr_conf_int = int(result.ocr_confidence * 100)

    extraction_id = str(uuid.uuid4())
    logger.info(
        "Document uploaded: type=%s pages=%d ocr_confidence=%s extraction_id=%s",
        result.document_type.value,
        result.page_count,
        ocr_conf_int,
        extraction_id,
    )

    return DocumentUploadResponse(
        extraction_id=extraction_id,
        extracted_text=result.extracted_text,
        document_type=result.document_type.value,
        page_count=result.page_count,
        ocr_confidence=ocr_conf_int,
        warnings=result.warnings,
    )


@app.post(
    "/api/analyze/text",
    tags=["analysis"],
    summary="Analyze confirmed/edited contract text and stream results",
)
async def analyze_from_text(request: AnalyzeFromTextRequest) -> StreamingResponse:
    """Run the full RAG analysis pipeline on user-confirmed text.

    This is the second step of the OCR workflow: the user has reviewed
    (and optionally edited) the extracted text and confirmed it.
    """
    pipe = _require_pipeline()
    contract_text = request.contract_text

    cleaned_text = pipe.preprocess_contract_text(contract_text)
    clauses = pipe.split_clauses(cleaned_text)
    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any clauses from the provided text.",
        )
    _validate_clause_count(clauses)

    logger.info("Analysing %d clause(s) from confirmed text sequentially via SSE with context", len(clauses))

    # Build a chat session for follow-up questions
    session_id = str(uuid.uuid4())
    system_instruction = (
        "You are an expert Indian legal assistant. The user has uploaded an "
        "employment contract. Answer any questions about the contract. "
        "Use the retrieved legal context provided with each question to "
        "ground your answers in specific laws and provisions."
    )

    initial_state: ChatSessionState = {
        "model_name": pipe.model_name or _GENERATIVE_MODEL_NAME,
        "chat_config": {"system_instruction": system_instruction},
        "history": [],
        "contract_text": contract_text,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    await chat_sessions.set(session_id, initial_state)

    async def sse_generator() -> AsyncGenerator[str, None]:
        init_data = {
            "type": "init",
            "session_id": session_id,
            "total_clauses": len(clauses),
        }
        yield f"data: {json.dumps(init_data)}\n\n"

        contract_summary = await pipe.generate_contract_summary(contract_text)
        yield f"data: {json.dumps({'type': 'summary', 'text': contract_summary})}\n\n"

        previous_analyses: list[dict] = []

        for i, clause in enumerate(clauses):
            try:
                analysis_dict = await pipe.analyze_clause(
                    clause_text=clause,
                    clause_index=i,
                    total_clauses=len(clauses),
                    contract_summary=contract_summary,
                    previous_analyses=previous_analyses,
                    all_clauses=clauses,
                )
                previous_analyses.append(analysis_dict)

                clause_data = {
                    "type": "clause",
                    "index": i,
                    "clause": clause,
                    "analysis": analysis_dict,
                }
                yield f"data: {json.dumps(clause_data)}\n\n"
            except Exception:
                logger.exception("Failed to analyse clause %d: %.60s", i, clause)

        try:
            exec_summary_dict = await pipe.generate_executive_summary(
                contract_summary=contract_summary,
                clause_analyses=previous_analyses,
                all_clauses=clauses,
            )
            yield f"data: {json.dumps({'type': 'executive_summary', 'summary': exec_summary_dict})}\n\n"
        except Exception:
            logger.exception("Failed to generate executive summary")

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post(
    "/api/preview/text",
    response_model=ClausePreviewResponse,
    tags=["analysis"],
    summary="Preview clause splitting from confirmed text",
)
async def preview_clauses_from_text(
    request: AnalyzeFromTextRequest,
) -> ClausePreviewResponse:
    """Split user-confirmed text into clauses without running analysis."""
    pipe = _require_pipeline()

    cleaned_text = pipe.preprocess_contract_text(request.contract_text)
    clauses = pipe.split_clauses(cleaned_text)
    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any clauses from the provided text.",
        )
    _validate_clause_count(clauses)
    return ClausePreviewResponse(total_clauses=len(clauses), clauses=clauses)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )