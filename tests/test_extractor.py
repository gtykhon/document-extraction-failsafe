"""Unit tests for extraction engine cascade logic."""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extractor import ExtractionEngine
from models import ExtractionMethod, ExtractionResult
from quality import QualityScorer


class TestExtractionEngine:
    """Tests for ExtractionEngine cascade logic."""

    @pytest.fixture
    def engine(self):
        """Create an ExtractionEngine instance."""
        return ExtractionEngine(
            pypdf2_quality_threshold=0.5,
            pymupdf_quality_threshold=0.5,
            min_words=50,
        )

    @pytest.fixture
    def sample_pdf_path(self):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_nonexistent_file(self, engine):
        """Extraction of nonexistent file should fail gracefully."""
        result = engine.extract("/nonexistent/path/file.pdf")
        assert result.success is False
        assert result.text == ""
        assert result.confidence_score == 0.0
        assert "not found" in result.error.lower()

    def test_extraction_returns_result_object(self, engine, sample_pdf_path):
        """Extraction should return ExtractionResult object."""
        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2:
            mock_pypdf2.return_value = ("test text", MagicMock(overall=0.8, word_count=100))
            result = engine.extract(sample_pdf_path)
            assert isinstance(result, ExtractionResult)

    def test_cascade_order_pypdf2_success(self, engine):
        """If PyPDF2 succeeds, should not advance to PyMuPDF."""
        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2, \
             patch.object(engine, "_extract_pymupdf") as mock_pymupdf, \
             patch.object(engine, "_get_page_count") as mock_page_count:

            mock_page_count.return_value = 1
            quality_score = MagicMock(overall=0.8, word_count=100)
            mock_pypdf2.return_value = ("good text", quality_score)

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.close()

            result = engine.extract(temp_file.name)

            assert result.method_used == ExtractionMethod.PYPDF2
            assert result.text == "good text"
            mock_pymupdf.assert_not_called()

            Path(temp_file.name).unlink(missing_ok=True)

    def test_cascade_order_pymupdf_on_pypdf2_failure(self, engine):
        """If PyPDF2 fails, should advance to PyMuPDF."""
        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2, \
             patch.object(engine, "_extract_pymupdf") as mock_pymupdf, \
             patch.object(engine, "_get_page_count") as mock_page_count:

            mock_page_count.return_value = 1
            poor_quality = MagicMock(overall=0.3, word_count=10)
            good_quality = MagicMock(overall=0.8, word_count=100)

            mock_pypdf2.return_value = ("poor text", poor_quality)
            mock_pymupdf.return_value = ("good text", good_quality, False)

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.close()

            with patch.object(engine.quality_scorer, "is_acceptable") as mock_acceptable:
                mock_acceptable.side_effect = [False, True]
                result = engine.extract(temp_file.name)

                assert result.method_used == ExtractionMethod.PYMUPDF
                assert result.text == "good text"

            Path(temp_file.name).unlink(missing_ok=True)

    def test_cascade_advances_to_tesseract(self, engine):
        """If PyMuPDF fails, should advance to Tesseract."""
        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2, \
             patch.object(engine, "_extract_pymupdf") as mock_pymupdf, \
             patch.object(engine, "_extract_tesseract") as mock_tesseract, \
             patch.object(engine, "_get_page_count") as mock_page_count:

            mock_page_count.return_value = 1
            poor_quality = MagicMock(overall=0.3, word_count=10)
            good_quality = MagicMock(overall=0.8, word_count=100)

            mock_pypdf2.return_value = ("poor", poor_quality)
            mock_pymupdf.return_value = ("poor", poor_quality, True)
            mock_tesseract.return_value = ("good ocr text", good_quality)

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.close()

            with patch.object(engine.quality_scorer, "is_acceptable") as mock_acceptable:
                mock_acceptable.side_effect = [False, False]
                result = engine.extract(temp_file.name)

                assert result.method_used == ExtractionMethod.TESSERACT
                mock_tesseract.assert_called_once()

            Path(temp_file.name).unlink(missing_ok=True)

    def test_quality_threshold_applied(self, engine):
        """Quality threshold should be compared against result."""
        engine.pypdf2_threshold = 0.7

        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2, \
             patch.object(engine, "_extract_pymupdf") as mock_pymupdf, \
             patch.object(engine, "_get_page_count") as mock_page_count:

            mock_page_count.return_value = 1
            below_threshold = MagicMock(overall=0.6, word_count=100)
            mock_pypdf2.return_value = ("text", below_threshold)

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.close()

            with patch.object(engine.quality_scorer, "is_acceptable") as mock_acceptable:
                mock_acceptable.return_value = True
                result = engine.extract(temp_file.name)

                assert result.method_used == ExtractionMethod.PYMUPDF

            Path(temp_file.name).unlink(missing_ok=True)

    def test_extraction_timing(self, engine):
        """Extraction should measure processing time."""
        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2, \
             patch.object(engine, "_get_page_count") as mock_page_count:

            mock_page_count.return_value = 1
            good_quality = MagicMock(overall=0.8, word_count=100)
            mock_pypdf2.return_value = ("text", good_quality)

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.close()

            with patch.object(engine.quality_scorer, "is_acceptable") as mock_acceptable:
                mock_acceptable.return_value = True
                result = engine.extract(temp_file.name)

                assert result.processing_time >= 0

            Path(temp_file.name).unlink(missing_ok=True)

    def test_extraction_includes_metadata(self, engine):
        """Extraction result should include page count and image info."""
        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2, \
             patch.object(engine, "_get_page_count") as mock_page_count:

            mock_page_count.return_value = 5
            good_quality = MagicMock(overall=0.8, word_count=100)
            mock_pypdf2.return_value = ("text", good_quality)

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.close()

            with patch.object(engine.quality_scorer, "is_acceptable") as mock_acceptable:
                mock_acceptable.return_value = True
                result = engine.extract(temp_file.name)

                assert result.page_count == 5
                assert result.quality_score is not None

            Path(temp_file.name).unlink(missing_ok=True)

    def test_pypdf2_extraction_failure_graceful(self, engine):
        """PyPDF2 extraction exceptions should be handled gracefully."""
        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2:
            mock_pypdf2.side_effect = RuntimeError("PDF read error")

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.close()

            result = engine.extract(temp_file.name)
            assert result.error is not None

            Path(temp_file.name).unlink(missing_ok=True)

    def test_confidence_score_from_quality(self, engine):
        """Confidence score should come from quality assessment."""
        with patch.object(engine, "_extract_pypdf2") as mock_pypdf2, \
             patch.object(engine, "_get_page_count") as mock_page_count:

            mock_page_count.return_value = 1
            good_quality = MagicMock(overall=0.85, word_count=200)
            mock_pypdf2.return_value = ("text", good_quality)

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.close()

            with patch.object(engine.quality_scorer, "is_acceptable") as mock_acceptable:
                mock_acceptable.return_value = True
                result = engine.extract(temp_file.name)

                assert result.confidence_score == 0.85

            Path(temp_file.name).unlink(missing_ok=True)


class TestExtractionModule:
    """Tests for module-level extraction functions."""

    def test_extract_text_function(self):
        """Module function should work with default settings."""
        from extractor import extract_text

        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_file.close()

        with patch.object(ExtractionEngine, "extract") as mock_extract:
            mock_result = ExtractionResult(
                text="test",
                method_used=ExtractionMethod.PYPDF2,
                confidence_score=0.8,
                processing_time=0.1,
            )
            mock_extract.return_value = mock_result

            result = extract_text(temp_file.name)
            assert isinstance(result, ExtractionResult)

        Path(temp_file.name).unlink(missing_ok=True)
