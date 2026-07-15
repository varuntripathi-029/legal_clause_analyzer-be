"""
Ingestion package — modular document loading and OCR.

Public API
----------
.. code-block:: python

    from ingestion import extract_text_from_upload, clean_ocr_text, DocumentType

The package is completely independent of the RAG pipeline and can be
reused in any project that needs PDF / image → text conversion.
"""

from ingestion.document_loader import (
    DocumentExtractionResult,
    DocumentType,
    SUPPORTED_EXTENSIONS,
    extract_text_from_upload,
)
from ingestion.text_cleaner import clean_ocr_text
from ingestion.ocr_provider import OCRProvider, PaddleOCRProvider, OCRResult

__all__ = [
    # Core entry point
    "extract_text_from_upload",
    # Result / enum types
    "DocumentExtractionResult",
    "DocumentType",
    "SUPPORTED_EXTENSIONS",
    # Text cleaning (usable standalone)
    "clean_ocr_text",
    # OCR provider interface (for extensibility)
    "OCRProvider",
    "PaddleOCRProvider",
    "OCRResult",
]
