"""PDF metadata extraction integration.

Provides PDF metadata extraction using PyMuPDF (fitz).
Gracefully degrades when PyMuPDF is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("filehub")

try:
    import fitz

    PDF_AVAILABLE = True
except ImportError:
    fitz = None  # noqa: F841
    PDF_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class PdfMetadata:
    """Immutable PDF document metadata."""

    title: str
    author: str
    page_count: int
    creation_date: str | None
    file_size: int
    has_text: bool


class PdfMetadataExtractor:
    """Extract metadata from PDF files using PyMuPDF.

    Falls back gracefully when PyMuPDF is not installed.
    """

    def __init__(self) -> None:
        if not PDF_AVAILABLE:
            logger.warning(
                "PyMuPDF is not installed. PDF metadata extraction is unavailable. "
                "Install with: pip install filehub[pdf]"
            )

    @property
    def is_available(self) -> bool:
        """Check whether PDF extraction is available."""
        return PDF_AVAILABLE

    def extract(self, path: Path) -> PdfMetadata | None:
        """Extract metadata from a PDF file.

        Args:
            path: Path to the PDF file.

        Returns:
            PdfMetadata if extraction succeeds, None otherwise.
        """
        if not PDF_AVAILABLE:
            logger.warning("PDF extraction unavailable: PyMuPDF not installed")
            return None

        if not path.exists():
            logger.warning("PDF file not found: %s", path)
            return None

        if path.suffix.lower() != ".pdf":
            logger.warning("Not a PDF file: %s", path)
            return None

        try:
            return self._extract_with_fitz(path)
        except Exception:
            logger.exception("Failed to extract PDF metadata: %s", path)
            return None

    def _extract_with_fitz(self, path: Path) -> PdfMetadata:
        """Internal extraction using fitz (PyMuPDF).

        Args:
            path: Path to the PDF file (already validated).

        Returns:
            PdfMetadata with extracted information.
        """
        doc = fitz.open(str(path))
        try:
            metadata = doc.metadata or {}
            title = metadata.get("title", "") or ""
            author = metadata.get("author", "") or ""
            creation_date = metadata.get("creationDate") or None

            # Check if any page has extractable text
            has_text = False
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                if text:
                    has_text = True
                    break

            file_size = path.stat().st_size

            return PdfMetadata(
                title=title,
                author=author,
                page_count=len(doc),
                creation_date=creation_date,
                file_size=file_size,
                has_text=has_text,
            )
        finally:
            doc.close()
