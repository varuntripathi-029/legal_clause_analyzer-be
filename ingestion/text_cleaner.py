"""
Post-OCR text cleaning utilities.

Addresses common OCR artefacts:
* Excessive / duplicate whitespace
* Broken lines that split a sentence across two lines
* Stray page numbers (``- 3 -``, ``Page 5 of 12``, etc.)
* Repeated headers and footers across pages

The cleaner is intentionally stateless — every function takes a string
and returns a cleaned string so they compose easily.
"""

from __future__ import annotations

import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning passes
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace (spaces / tabs) into one space.

    Preserves newlines so paragraph structure is maintained at this stage.
    """
    # Replace runs of spaces/tabs with a single space (line-by-line)
    lines = text.split("\n")
    cleaned = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned)


def _remove_page_numbers(text: str) -> str:
    """Strip common page-number patterns from the text.

    Matched patterns (case-insensitive):
    * ``- 3 -``
    * ``Page 5``
    * ``Page 5 of 12``
    * ``5 / 12``
    * A bare number on its own line
    """
    lines = text.split("\n")
    cleaned: list[str] = []

    page_num_patterns = [
        re.compile(r"^\s*-\s*\d+\s*-\s*$"),                     # - 3 -
        re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I),  # Page 5 / Page 5 of 12
        re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),                    # 5 / 12
        re.compile(r"^\s*\d{1,4}\s*$"),                          # bare number (≤4 digits)
    ]

    for line in lines:
        if any(pat.match(line) for pat in page_num_patterns):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def _remove_repeated_headers_footers(text: str) -> str:
    """Detect and remove lines that repeat identically across ≥ 3 page breaks.

    A "page break" is approximated by a double-newline boundary.  Lines that
    appear in the first or last position of ≥ 3 consecutive page-blocks and
    are identical are treated as running headers / footers and stripped.
    """
    pages = re.split(r"\n{2,}", text)
    if len(pages) < 3:
        return text  # Too few pages to detect repetition

    # Collect first and last non-empty lines per page
    first_lines: list[str] = []
    last_lines: list[str] = []
    for page in pages:
        stripped_lines = [l.strip() for l in page.split("\n") if l.strip()]
        if stripped_lines:
            first_lines.append(stripped_lines[0])
            last_lines.append(stripped_lines[-1])

    # Lines that appear as first/last in ≥ 3 pages → likely header/footer
    header_counts = Counter(first_lines)
    footer_counts = Counter(last_lines)

    to_remove: set[str] = set()
    threshold = min(3, len(pages))
    for line, count in header_counts.items():
        if count >= threshold and line:
            to_remove.add(line)
    for line, count in footer_counts.items():
        if count >= threshold and line:
            to_remove.add(line)

    if not to_remove:
        return text

    logger.debug(
        "Stripping %d detected header/footer pattern(s): %s",
        len(to_remove),
        to_remove,
    )

    cleaned_lines = [
        line for line in text.split("\n") if line.strip() not in to_remove
    ]
    return "\n".join(cleaned_lines)


def _merge_broken_lines(text: str) -> str:
    """Join lines that were broken mid-sentence by the OCR engine.

    Heuristic: if a line does **not** end with sentence-ending punctuation
    (``.``, ``:``, ``;``, ``?``, ``!``) or a list marker, and the next line
    starts with a lowercase letter, merge them.
    """
    lines = text.split("\n")
    if not lines:
        return text

    merged: list[str] = []
    buffer = lines[0]

    sentence_end = re.compile(r"[.;:?!)\]\"']\s*$")
    list_marker = re.compile(r"^\s*(?:\d+[.)]\s|[a-zA-Z][.)]\s|\([a-zA-Z0-9]+\)\s|[-•▪])")

    for next_line in lines[1:]:
        stripped_next = next_line.strip()

        # Blank lines → paragraph boundary — flush buffer
        if not stripped_next:
            merged.append(buffer)
            merged.append("")
            buffer = ""
            continue

        # If current buffer is empty, start fresh
        if not buffer:
            buffer = stripped_next
            continue

        # Merge conditions: current line doesn't end with punctuation AND
        # next line starts with a lowercase letter (continuation)
        if (
            not sentence_end.search(buffer)
            and not list_marker.match(stripped_next)
            and stripped_next[0].islower()
        ):
            buffer = buffer.rstrip() + " " + stripped_next
        else:
            merged.append(buffer)
            buffer = stripped_next

    if buffer:
        merged.append(buffer)

    return "\n".join(merged)


def _collapse_blank_lines(text: str) -> str:
    """Reduce three-or-more consecutive blank lines to exactly two (paragraph break)."""
    return re.sub(r"\n{3,}", "\n\n", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """Run the full cleaning pipeline on raw OCR output.

    Order matters — whitespace normalisation runs first so downstream
    passes see consistent spacing.

    Parameters
    ----------
    text:
        Raw OCR-extracted text.

    Returns
    -------
    str
        Cleaned text ready for clause splitting.
    """
    if not text or not text.strip():
        return ""

    result = text
    result = _normalize_whitespace(result)
    result = _remove_page_numbers(result)
    result = _remove_repeated_headers_footers(result)
    result = _merge_broken_lines(result)
    result = _collapse_blank_lines(result)
    return result.strip()
