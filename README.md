---
title: Legal RAG Assistant
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---
# Legal Contract Analyzer — Backend

A RAG-powered FastAPI service that analyzes Indian employment contracts (PDFs, images, plain text), splits them into clauses, and evaluates each against the Constitution of India (Fundamental Rights) and the Indian Contract Act, 1872.

**Live Frontend Application:** [legal-rag-frontend-silk.vercel.app](https://legal-rag-frontend-silk.vercel.app/)

---

## Technical Stack

* **FastAPI**: Asynchronous web framework exposing REST endpoints and Server-Sent Events (SSE) streaming.
* **Groq (Llama-3.3-70b-versatile)**: Generative LLM for legal reasoning, delivering lightning-fast inference.
* **FastEmbed (BAAI/bge-base-en-v1.5)**: High-performance CPU-optimized local embeddings (768-dim) for semantic search.
* **FAISS (L2 Index)**: High-performance vector database for retrieving legal contexts.
* **Redis**: Persistent multi-worker storage for active chat sessions (TTL-based).
* **PaddleOCR (CPU-only)**: Standalone, lazily loaded OCR engine for scanned PDFs and image files.
* **PyMuPDF & pdfplumber**: Pure-Python document parsing and page-to-image extraction (zero system dependencies, no Poppler).

---

## Key Features

1. **Multi-Format Input Ingestion**: Supports `.txt`, searchable `.pdf`, scanned `.pdf` (automatically converted to images and OCR'd), and `.jpg/.jpeg/.png` images.
2. **Two-Step Verification Workflow**: Extracts text and returns it with a confidence score/low-confidence warnings. The user can review, edit, or confirm the text before running the heavy legal analyzer.
3. **SSE Clause-Level Streaming**: Returns analysis per-clause concurrently as they complete, preventing UX freezes.
4. **Context-Aware Follow-Up Chat**: Keeps conversational state via Redis for document-specific Q&A after analysis.
5. **Calibrated Hybrid Confidence**: Combines vector match score (40%) and LLM certainty (60%) for realistic confidence metrics.

---

## Performance & Optimization Metrics

| Parameter | Gemini API (Old Pipeline) | FastEmbed + Groq (New Pipeline) | Speedup / Impact |
|-----------|---------------------------|----------------------------------|-------------------|
| **Embedding Latency** | ~800ms - 1200ms / request | **~8ms - 15ms / request** (Local CPU) | **~100x Faster** (No API costs, offline capable) |
| **LLM Inference Latency** | ~2.5s - 3.5s / clause | **~350ms - 500ms / clause** | **~7x Faster** |
| **Total Analysis Time (10 Clauses)** | ~25 seconds | **~3 - 4 seconds** (with Concurrency) | **~6-8x Faster** overall |
| **Active Session Persistence** | In-Memory (No multi-worker safety) | **Redis Cache Store** (TTL Bounded) | Highly scalable, survives container restarts |
| **Ingestion Engine** | Searchable PDF only | **Modular OCR (PaddleOCR)** | Expands support to scanned docs / image files |

---

## Problems Solved (OCR Ingestion & Platform Integration)

### 1. PaddleOCR 3.7.0 API Breaking Changes
* **Problem**: PaddleOCR recently moved to PaddX pipelines under the hood. Constructing `PaddleOCR(show_log=False)` raised `ValueError: Unknown argument: show_log`. Additionally, calling `.ocr(img, cls=True)` raised `TypeError: PaddleOCR.predict() got an unexpected keyword argument 'cls'`.
* **Solution**: Developed a robust dual-compatibility driver in `ocr_provider.py`. It tries 3.x constructor/method interfaces first (using `use_textline_orientation=True` and `device="cpu"`), catching errors and falling back to 2.x arguments (`use_gpu=False`, `cls=True`) on older systems.

### 2. oneDNN / MKL-DNN Runtime Crash on Windows
* **Problem**: PaddlePaddle 3.3.1 static runner throws a `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support` on CPU when oneDNN (MKL-DNN) optimizations are turned on.
* **Solution**: Explicitly disabled oneDNN during initialization by setting `enable_mkldnn=False` in the PaddleOCR arguments. This bypassed compilation errors and allowed smooth CPU execution.

### 3. Poppler System Dependency
* **Problem**: Converting scanned PDFs to images typically requires `pdf2image`, which depends on compiling the C++ library `Poppler` (causing complex Docker and multi-platform setups).
* **Solution**: Used **PyMuPDF's** native `page.get_pixmap(dpi=300)` to render pages as raw pixel maps in memory and load them straight into Pillow, removing Poppler entirely.

### 4. Lazy-loading Heavy Models
* **Problem**: PaddleOCR and PaddlePaddle total ~1.2 GB in memory/dependencies, which slows server boot time and bloats lightweight text analysis runs.
* **Solution**: Isolated the OCR components into a modular `ingestion/` package and lazily imported PaddleOCR inside execution scopes. If a user uploads plain text or a searchable PDF, PaddleOCR is never imported or loaded.

---

## API Routes

| Method | Path | Payload | Output |
|--------|------|---------|--------|
| `POST` | `/api/upload` | Multipart File | `DocumentUploadResponse` (Extracted Text, Confidence, Warnings) |
| `POST` | `/api/analyze/text` | JSON `AnalyzeFromTextRequest` | SSE Stream (`init` -> `clause` -> `done` events) |
| `POST` | `/api/preview/text` | JSON `AnalyzeFromTextRequest` | JSON `ClausePreviewResponse` (Total clauses, splits) |
| `POST` | `/api/chat` | JSON `ChatRequest` | JSON `ChatResponse` |
| `GET` | `/api/status` | - | Pipeline Status, Model Details, FAISS size |
| `GET` | `/` | - | Health check |

---

## Local Development

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Environment
# Create .env with GROQ_API_KEY, REDIS_URL (optional)
echo "GROQ_API_KEY=your_key_here" > .env

# Run FastAPI Server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```