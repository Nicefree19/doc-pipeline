"""FileHub Integrations Module.

Optional integrations for extended functionality:
- PDF metadata extraction (requires PyMuPDF)
"""

from .pdf import PDF_AVAILABLE, PdfMetadata, PdfMetadataExtractor

__all__ = ["PdfMetadataExtractor", "PdfMetadata", "PDF_AVAILABLE"]
