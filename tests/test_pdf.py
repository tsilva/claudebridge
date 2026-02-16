"""
PDF extraction tests for claudebridge.

Prerequisites:
- Server must be running for integration tests: claudebridge
- fpdf2 must be installed for PDF fixture generation

Usage:
- Unit tests: pytest tests/test_pdf.py::TestPdfUtils tests/test_pdf.py::TestSlugification -v
- Integration tests: pytest tests/test_pdf.py::TestPdfIntegration -v
"""

import base64
from pathlib import Path

import pytest

from claudebridge.server import (
    parse_data_url,
    openai_image_to_claude,
    openai_content_to_claude,
    extract_text_from_content,
)
from claudebridge.models import (
    TextContent,
    ImageUrlContent,
    ImageUrl,
)

from .test_utils import slugify_text, text_similarity


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDF_TEST_DOCUMENT = FIXTURES_DIR / "pdf_test_document.pdf"

# Expected text content in the PDF fixture
EXPECTED_PDF_TEXT = """CLAUDE CODE BRIDGE
PDF Extraction Test Document
This document tests PDF extraction accuracy.
Version: 2025-01
Status: ACTIVE"""


class TestPdfUtils:
    """Unit tests for PDF handling in image_utils."""

    def test_parse_data_url_pdf(self):
        """Parse PDF data URL."""
        url = "data:application/pdf;base64,JVBERi0xLjQK"
        media_type, data = parse_data_url(url)

        assert media_type == "application/pdf"
        assert data == "JVBERi0xLjQK"

    def test_openai_pdf_to_claude_document(self):
        """Convert PDF data URL to Claude document format."""
        pdf_content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:application/pdf;base64,JVBERi0xLjQK")
        )
        result = openai_image_to_claude(pdf_content)

        assert result == {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjQK",
            }
        }

    def test_openai_content_with_pdf(self):
        """Convert mixed content with PDF to Claude format."""
        content = [
            TextContent(type="text", text="Extract text from this PDF:"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:application/pdf;base64,JVBERi0xLjQK")
            ),
        ]
        result = openai_content_to_claude(content)

        assert result == [
            {"type": "text", "text": "Extract text from this PDF:"},
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjQK",
                }
            },
        ]

    def test_extract_text_pdf_placeholder(self):
        """Extract text shows PDF placeholder for logging."""
        content = [
            TextContent(type="text", text="Extract this:"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:application/pdf;base64,JVBERi0xLjQK")
            ),
        ]
        result = extract_text_from_content(content)

        assert "Extract this:" in result
        assert "[document: PDF base64 data]" in result

    def test_image_still_produces_image_block(self):
        """Verify images still produce image blocks, not document blocks."""
        image_content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:image/png;base64,iVBORw0KGgo=")
        )
        result = openai_image_to_claude(image_content)

        assert result["type"] == "image"
        assert result["source"]["media_type"] == "image/png"


class TestSlugification:
    """Unit tests for text slugification and similarity functions."""

    def test_slugify_basic(self):
        """Slugify removes punctuation and lowercases."""
        assert slugify_text("Hello, World!") == "hello world"
        assert slugify_text("Test: 123") == "test 123"
        assert slugify_text("UPPERCASE") == "uppercase"

    def test_slugify_multiline(self):
        """Slugify normalizes whitespace including newlines."""
        text = "Line 1\n\nLine 2\tLine 3"
        assert slugify_text(text) == "line 1 line 2 line 3"

    def test_slugify_unicode(self):
        """Slugify handles unicode characters."""
        assert slugify_text("café") == "cafe"
        assert slugify_text("naïve") == "naive"

    def test_text_similarity_exact_match(self):
        """Exact match returns 1.0 similarity."""
        assert text_similarity("hello world", "hello world") == 1.0

    def test_text_similarity_case_insensitive(self):
        """Similarity is case insensitive."""
        assert text_similarity("Hello World", "hello world") == 1.0

    def test_text_similarity_partial_match(self):
        """Partial match returns proportional similarity."""
        # 2 of 4 words match
        assert text_similarity("a b c d", "a b x y") == 0.5

    def test_text_similarity_no_match(self):
        """No matching words returns 0.0."""
        assert text_similarity("hello world", "foo bar") == 0.0

    def test_text_similarity_empty_expected(self):
        """Empty expected text returns 1.0."""
        assert text_similarity("", "anything") == 1.0

    def test_text_similarity_superset(self):
        """Actual text containing all expected words returns 1.0."""
        assert text_similarity("a b", "a b c d e f") == 1.0


