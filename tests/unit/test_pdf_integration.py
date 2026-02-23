"""Tests for PDF metadata extraction integration."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPdfAvailableFlag:
    """Test PDF_AVAILABLE flag behavior."""

    def test_pdf_available_true_when_fitz_installed(self):
        """Test that PDF_AVAILABLE is True when fitz can be imported."""
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            # Re-import to trigger the try/except with fitz available
            import importlib

            import src.filehub.integrations.pdf as pdf_mod

            importlib.reload(pdf_mod)
            assert pdf_mod.PDF_AVAILABLE is True

    def test_pdf_available_false_when_fitz_missing(self):
        """Test that PDF_AVAILABLE is False when fitz import fails."""
        with patch.dict("sys.modules", {"fitz": None}):
            import importlib

            import src.filehub.integrations.pdf as pdf_mod

            importlib.reload(pdf_mod)
            assert pdf_mod.PDF_AVAILABLE is False


class TestPdfMetadata:
    """Test PdfMetadata dataclass."""

    def test_create_with_all_fields(self):
        """Test creating PdfMetadata with all fields populated."""
        from src.filehub.integrations.pdf import PdfMetadata

        metadata = PdfMetadata(
            title="Test Document",
            author="Test Author",
            page_count=10,
            creation_date="D:20240101120000",
            file_size=1024,
            has_text=True,
        )
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.page_count == 10
        assert metadata.creation_date == "D:20240101120000"
        assert metadata.file_size == 1024
        assert metadata.has_text is True

    def test_create_with_none_creation_date(self):
        """Test creating PdfMetadata with None creation_date."""
        from src.filehub.integrations.pdf import PdfMetadata

        metadata = PdfMetadata(
            title="",
            author="",
            page_count=0,
            creation_date=None,
            file_size=0,
            has_text=False,
        )
        assert metadata.creation_date is None

    def test_frozen_immutability(self):
        """Test that PdfMetadata is immutable (frozen dataclass)."""
        from src.filehub.integrations.pdf import PdfMetadata

        metadata = PdfMetadata(
            title="Test",
            author="Author",
            page_count=1,
            creation_date=None,
            file_size=100,
            has_text=False,
        )
        with pytest.raises(FrozenInstanceError):
            metadata.title = "Changed"  # type: ignore[misc]


class TestPdfMetadataExtractor:
    """Test PdfMetadataExtractor class."""

    def test_is_available_when_pdf_available(self):
        """Test is_available property returns True when PDF_AVAILABLE is True."""
        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", True):
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            extractor = PdfMetadataExtractor()
            assert extractor.is_available is True

    def test_is_available_when_pdf_unavailable(self):
        """Test is_available property returns False when PDF_AVAILABLE is False."""
        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", False):
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            extractor = PdfMetadataExtractor()
            assert extractor.is_available is False

    def test_extract_when_pdf_unavailable_returns_none(self):
        """Test extract() returns None when PDF_AVAILABLE is False."""
        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", False):
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            extractor = PdfMetadataExtractor()
            result = extractor.extract(Path("test.pdf"))
            assert result is None

    def test_extract_file_not_found_returns_none(self):
        """Test extract() returns None when file does not exist."""
        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", True):
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            extractor = PdfMetadataExtractor()
            result = extractor.extract(Path("/nonexistent/path/test.pdf"))
            assert result is None

    def test_extract_non_pdf_file_returns_none(self, temp_dir):
        """Test extract() returns None for non-PDF file extensions."""
        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", True):
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            txt_file = temp_dir / "document.txt"
            txt_file.write_text("not a pdf")

            extractor = PdfMetadataExtractor()
            result = extractor.extract(txt_file)
            assert result is None

    def test_extract_corrupted_pdf_returns_none(self, temp_dir):
        """Test extract() returns None when fitz raises an exception."""
        mock_fitz = MagicMock()
        mock_fitz.open.side_effect = RuntimeError("corrupted PDF")

        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", True), \
             patch("src.filehub.integrations.pdf.fitz", mock_fitz):
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            pdf_file = temp_dir / "corrupted.pdf"
            pdf_file.write_bytes(b"not a real pdf")

            extractor = PdfMetadataExtractor()
            result = extractor.extract(pdf_file)
            assert result is None

    def test_extract_success_with_mocked_fitz(self, temp_dir):
        """Test extract() returns PdfMetadata with mocked fitz document."""
        # Set up mock page
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Hello World"

        # Set up mock document
        mock_doc = MagicMock()
        mock_doc.metadata = {
            "title": "Test PDF",
            "author": "Jane Doe",
            "creationDate": "D:20240315100000",
        }
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", True), \
             patch("src.filehub.integrations.pdf.fitz", mock_fitz):
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            pdf_file = temp_dir / "document.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 fake content")

            extractor = PdfMetadataExtractor()
            result = extractor.extract(pdf_file)

            assert result is not None
            assert result.title == "Test PDF"
            assert result.author == "Jane Doe"
            assert result.page_count == 5
            assert result.creation_date == "D:20240315100000"
            assert result.file_size == pdf_file.stat().st_size
            assert result.has_text is True
            mock_doc.close.assert_called_once()

    def test_extract_no_text_content(self, temp_dir):
        """Test extract() correctly detects PDF with no text content."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "   "

        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "", "author": ""}
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", True), \
             patch("src.filehub.integrations.pdf.fitz", mock_fitz):
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            pdf_file = temp_dir / "scanned.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 scanned")

            extractor = PdfMetadataExtractor()
            result = extractor.extract(pdf_file)

            assert result is not None
            assert result.has_text is False
            assert result.title == ""
            assert result.author == ""

    def test_init_logs_warning_when_unavailable(self):
        """Test that __init__ logs a warning when PDF is unavailable."""
        with patch("src.filehub.integrations.pdf.PDF_AVAILABLE", False), \
             patch("src.filehub.integrations.pdf.logger") as mock_logger:
            from src.filehub.integrations.pdf import PdfMetadataExtractor

            PdfMetadataExtractor()
            mock_logger.warning.assert_called_once()
            assert "PyMuPDF" in mock_logger.warning.call_args[0][0]
