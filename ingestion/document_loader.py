"""
Document loader — top-level orchestrator for file ingestion.

Accepts raw file bytes and a filename, determines the document type,
and dispatches to the appropriate reader.  The result is a unified
``DocumentExtractionResult`` ready for the API layer to return.

Supported file types
--------------------
* ``.txt``   — Plain text (direct passthrough)
* ``.pdf``   — Searchable or scanned (auto-detected)
* ``.jpg``, ``.jpeg``, ``.png`` — Image files (always OCR'd)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document type classification
# ---------------------------------------------------------------------------
class DocumentType(str, Enum):
    """Classification of the uploaded document."""

    PLAIN_TEXT = "plain_text"
    SEARCHABLE_PDF = "searchable_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE = "image"


# Map of extension → high-level category used for routing
_EXTENSION_MAP: dict[str, str] = {
    ".txt": "text",
    ".pdf": "pdf",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
}

SUPPORTED_EXTENSIONS: set[str] = set(_EXTENSION_MAP.keys())


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class DocumentExtractionResult:
    """Unified output regardless of file type."""

    extracted_text: str
    """The extracted (or OCR'd, then cleaned) text."""

    document_type: DocumentType
    """How the document was classified."""

    page_count: int
    """Number of pages (1 for images and plain text)."""

    ocr_confidence: float | None
    """Average OCR confidence in [0.0, 1.0] — ``None`` if OCR was skipped."""

    warnings: list[str] = field(default_factory=list)
    """User-facing warnings (e.g. low OCR confidence)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_upload(
    file_bytes: bytes,
    filename: str,
    *,
    ocr_confidence_threshold: float = 0.85,
) -> DocumentExtractionResult:
    """Detect file type and extract text.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded file.
    filename:
        Original filename (used for extension detection).
    ocr_confidence_threshold:
        If OCR confidence falls below this value a warning is appended
        to the result.  Defaults to 0.85 (85%).

    Returns
    -------
    DocumentExtractionResult

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    """
    ext = PurePosixPath(filename).suffix.lower()

    if ext not in _EXTENSION_MAP:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    category = _EXTENSION_MAP[ext]

    if category == "text":
        return _handle_plain_text(file_bytes)

    if category == "pdf":
        return _handle_pdf(file_bytes, ocr_confidence_threshold)

    if category == "image":
        return _handle_image(file_bytes, ocr_confidence_threshold)

    # Should never reach here — belt-and-suspenders
    raise ValueError(f"No handler for category '{category}'.")


# ---------------------------------------------------------------------------
# Private handlers
# ---------------------------------------------------------------------------

def _handle_plain_text(file_bytes: bytes) -> DocumentExtractionResult:
    """Decode and return raw text with no processing."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    return DocumentExtractionResult(
        extracted_text=text.strip(),
        document_type=DocumentType.PLAIN_TEXT,
        page_count=1,
        ocr_confidence=None,
    )


def _handle_pdf(
    file_bytes: bytes,
    confidence_threshold: float,
) -> DocumentExtractionResult:
    """Classify a PDF as searchable or scanned, then extract text."""
    from ingestion.pdf_reader import read_pdf  # noqa: WPS433

    result = read_pdf(file_bytes)
    warnings: list[str] = []

    if result.is_scanned:
        doc_type = DocumentType.SCANNED_PDF
        if (
            result.ocr_confidence is not None
            and result.ocr_confidence < confidence_threshold
        ):
            warnings.append(
                f"OCR confidence is low ({result.ocr_confidence:.0%}). "
                "Please review the extracted text carefully and consider "
                "editing it or uploading a higher-quality scan."
            )
    else:
        doc_type = DocumentType.SEARCHABLE_PDF

    return DocumentExtractionResult(
        extracted_text=result.text,
        document_type=doc_type,
        page_count=result.page_count,
        ocr_confidence=result.ocr_confidence,
        warnings=warnings,
    )


def _handle_image(
    file_bytes: bytes,
    confidence_threshold: float,
) -> DocumentExtractionResult:
    """Run OCR on an image file."""
    from ingestion.image_reader import read_image  # noqa: WPS433

    result = read_image(file_bytes)
    warnings: list[str] = []

    if result.ocr_confidence < confidence_threshold:
        warnings.append(
            f"OCR confidence is low ({result.ocr_confidence:.0%}). "
            "Please review the extracted text carefully and consider "
            "editing it or uploading a higher-quality image."
        )

    return DocumentExtractionResult(
        extracted_text=result.text,
        document_type=DocumentType.IMAGE,
        page_count=1,
        ocr_confidence=result.ocr_confidence,
        warnings=warnings,
    )
