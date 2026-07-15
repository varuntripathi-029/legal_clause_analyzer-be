"""
PDF text extraction with scanned-PDF detection.

Strategy
--------
1. Try **pdfplumber** to extract selectable text.
2. If pdfplumber is not installed, fall back to **PyMuPDF** text extraction.
3. Evaluate a per-page character-density heuristic.  If the average is
   below ``MIN_CHARS_PER_PAGE`` the PDF is classified as *scanned*.
4. For scanned PDFs, convert every page to a PIL image via
   ``PyMuPDF.page.get_pixmap()`` and delegate to ``ocr_processor``.

No external system dependencies (no Poppler / Tesseract).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Below this threshold (characters per page on average) the PDF is
# treated as a scanned document and OCR is triggered.
MIN_CHARS_PER_PAGE = 100


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PDFReadResult:
    """Result of reading a PDF file."""

    text: str
    """Extracted (or OCR'd) text."""

    is_scanned: bool
    """``True`` when the text came from OCR rather than selectable text."""

    page_count: int
    """Total number of pages in the PDF."""

    ocr_confidence: float | None
    """Average OCR confidence (0.0–1.0) if OCR was used, else ``None``."""


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(pdf_bytes: bytes) -> tuple[str, int] | None:
    """Attempt text extraction with pdfplumber.

    Returns ``(full_text, page_count)`` on success, or ``None`` if
    pdfplumber is not installed.
    """
    try:
        import pdfplumber  # noqa: WPS433
    except ImportError:
        logger.debug("pdfplumber not installed — skipping.")
        return None

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    return "\n".join(text_parts), page_count


def _extract_with_pymupdf(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract text using PyMuPDF (fitz).

    Always available since we depend on PyMuPDF for page-to-image
    conversion anyway.
    """
    import fitz  # noqa: WPS433 — PyMuPDF

    text_parts: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = len(doc)
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts), page_count


def _pdf_pages_to_images(pdf_bytes: bytes) -> list[PILImage.Image]:
    """Convert every PDF page to an RGB PIL image using PyMuPDF.

    Uses ``page.get_pixmap(dpi=300)`` for OCR-quality resolution.
    """
    import fitz  # noqa: WPS433
    from PIL import Image as PILImage  # noqa: WPS433

    images: list[PILImage.Image] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
            logger.debug(
                "Converted PDF page %d to image (%dx%d)",
                page_num + 1,
                pix.width,
                pix.height,
            )

    return images


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_pdf(pdf_bytes: bytes) -> PDFReadResult:
    """Extract text from a PDF, using OCR only when necessary.

    Parameters
    ----------
    pdf_bytes:
        Raw bytes of the uploaded PDF file.

    Returns
    -------
    PDFReadResult
    """
    # 1. Try selectable-text extraction
    plumber_result = _extract_with_pdfplumber(pdf_bytes)

    if plumber_result is not None:
        text, page_count = plumber_result
    else:
        text, page_count = _extract_with_pymupdf(pdf_bytes)

    # 2. Decide whether the PDF is scanned
    char_count = len(text.strip())
    avg_chars = char_count / max(page_count, 1)

    if avg_chars >= MIN_CHARS_PER_PAGE:
        # Sufficient selectable text — no OCR needed
        logger.info(
            "PDF classified as searchable (avg %.0f chars/page across %d pages).",
            avg_chars,
            page_count,
        )
        return PDFReadResult(
            text=text.strip(),
            is_scanned=False,
            page_count=page_count,
            ocr_confidence=None,
        )

    # 3. Scanned PDF — convert pages to images and OCR
    logger.info(
        "PDF classified as scanned (avg %.0f chars/page across %d pages). "
        "Starting OCR …",
        avg_chars,
        page_count,
    )

    from ingestion.ocr_processor import ocr_multiple_images  # noqa: WPS433

    images = _pdf_pages_to_images(pdf_bytes)
    ocr_result = ocr_multiple_images(images)

    return PDFReadResult(
        text=ocr_result.cleaned_text,
        is_scanned=True,
        page_count=page_count,
        ocr_confidence=ocr_result.confidence,
    )
