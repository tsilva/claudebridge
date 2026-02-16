"""
Comprehensive unit tests for image utilities.

These tests cover all image format conversions, URL handling, and edge cases.

Usage:
- pytest tests/test_image_utils.py -v
"""

import pytest

from claudebridge.server import (
    parse_data_url,
    is_http_url,
    is_data_url,
    openai_image_to_claude,
    openai_content_to_claude,
    has_multimodal_content,
    extract_text_from_content,
)
from claudebridge.models import (
    Message,
    TextContent,
    ImageUrlContent,
    ImageUrl,
)


@pytest.mark.unit
class TestParseDataUrl:
    """Tests for parse_data_url function."""

    def test_parse_png(self):
        """Parse PNG data URL."""
        url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        media_type, data = parse_data_url(url)
        assert media_type == "image/png"
        assert data.startswith("iVBORw0KGgo")

    def test_parse_jpeg(self):
        """Parse JPEG data URL."""
        url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD"
        media_type, data = parse_data_url(url)
        assert media_type == "image/jpeg"
        assert data.startswith("/9j/")

    def test_parse_gif(self):
        """Parse GIF data URL."""
        url = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        media_type, data = parse_data_url(url)
        assert media_type == "image/gif"
        assert data.startswith("R0lGODlh")

    def test_parse_webp(self):
        """Parse WebP data URL."""
        url = "data:image/webp;base64,UklGRiYAAABXRUJQVlA4IBoAAAAwAQCdASoBAAEAAUAmJYgCdAEO"
        media_type, data = parse_data_url(url)
        assert media_type == "image/webp"
        assert data.startswith("UklGR")

    def test_parse_pdf(self):
        """Parse PDF data URL."""
        url = "data:application/pdf;base64,JVBERi0xLjQKJeLjz9M="
        media_type, data = parse_data_url(url)
        assert media_type == "application/pdf"
        assert data.startswith("JVBERi0")

    def test_parse_svg(self):
        """Parse SVG data URL."""
        url = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg=="
        media_type, data = parse_data_url(url)
        assert media_type == "image/svg+xml"

    def test_invalid_url_not_data(self):
        """Non-data URL raises ValueError."""
        with pytest.raises(ValueError):
            parse_data_url("https://example.com/image.png")

    def test_invalid_url_no_base64(self):
        """Data URL without base64 encoding raises ValueError."""
        with pytest.raises(ValueError):
            parse_data_url("data:image/png,rawdata")

    def test_invalid_url_malformed(self):
        """Malformed data URL raises ValueError."""
        with pytest.raises(ValueError):
            parse_data_url("data:;base64,abc")

    def test_empty_data(self):
        """Data URL with empty data section raises ValueError (regex requires data)."""
        url = "data:image/png;base64,"
        with pytest.raises(ValueError):
            parse_data_url(url)


@pytest.mark.unit
class TestIsHttpUrl:
    """Tests for is_http_url function."""

    def test_https_url(self):
        """HTTPS URL returns True."""
        assert is_http_url("https://example.com/image.png") is True

    def test_http_url(self):
        """HTTP URL returns True."""
        assert is_http_url("http://example.com/image.png") is True

    def test_data_url(self):
        """Data URL returns False."""
        assert is_http_url("data:image/png;base64,abc") is False

    def test_file_path(self):
        """File path returns False."""
        assert is_http_url("/path/to/image.png") is False

    def test_relative_path(self):
        """Relative path returns False."""
        assert is_http_url("images/photo.jpg") is False

    def test_ftp_url(self):
        """FTP URL returns False."""
        assert is_http_url("ftp://example.com/file.txt") is False

    def test_empty_string(self):
        """Empty string returns False."""
        assert is_http_url("") is False


@pytest.mark.unit
class TestIsDataUrl:
    """Tests for is_data_url function."""

    def test_data_url_image(self):
        """Image data URL returns True."""
        assert is_data_url("data:image/png;base64,abc") is True

    def test_data_url_pdf(self):
        """PDF data URL returns True."""
        assert is_data_url("data:application/pdf;base64,abc") is True

    def test_http_url(self):
        """HTTP URL returns False."""
        assert is_data_url("https://example.com/image.png") is False

    def test_file_path(self):
        """File path returns False."""
        assert is_data_url("/path/to/file.pdf") is False

    def test_empty_string(self):
        """Empty string returns False."""
        assert is_data_url("") is False

    def test_data_prefix_only(self):
        """Just 'data:' returns True (though invalid)."""
        assert is_data_url("data:") is True


@pytest.mark.unit
class TestOpenaiImageToClaude:
    """Tests for openai_image_to_claude function."""

    def test_base64_png_to_image_block(self):
        """Base64 PNG converts to image block."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:image/png;base64,abc123"),
        )
        result = openai_image_to_claude(content)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == "abc123"

    def test_base64_jpeg_to_image_block(self):
        """Base64 JPEG converts to image block."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:image/jpeg;base64,xyz789"),
        )
        result = openai_image_to_claude(content)
        assert result["type"] == "image"
        assert result["source"]["media_type"] == "image/jpeg"

    def test_base64_gif_to_image_block(self):
        """Base64 GIF converts to image block."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:image/gif;base64,R0lGODlh"),
        )
        result = openai_image_to_claude(content)
        assert result["type"] == "image"
        assert result["source"]["media_type"] == "image/gif"

    def test_base64_webp_to_image_block(self):
        """Base64 WebP converts to image block."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:image/webp;base64,UklGR"),
        )
        result = openai_image_to_claude(content)
        assert result["type"] == "image"
        assert result["source"]["media_type"] == "image/webp"

    def test_base64_pdf_to_document_block(self):
        """Base64 PDF converts to document block (not image)."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:application/pdf;base64,JVBERi0xLjQ="),
        )
        result = openai_image_to_claude(content)
        assert result["type"] == "document"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "application/pdf"

    def test_http_url_to_url_source(self):
        """HTTP URL converts to url source."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="https://example.com/photo.jpg"),
        )
        result = openai_image_to_claude(content)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/photo.jpg"

    def test_https_url_to_url_source(self):
        """HTTPS URL converts to url source."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="https://cdn.example.com/images/photo.png"),
        )
        result = openai_image_to_claude(content)
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://cdn.example.com/images/photo.png"

    def test_unsupported_url_raises_error(self):
        """Unsupported URL format raises ValueError."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="/local/path/image.png"),
        )
        with pytest.raises(ValueError) as exc_info:
            openai_image_to_claude(content)
        assert "Unsupported image URL format" in str(exc_info.value)


@pytest.mark.unit
class TestOpenaiContentToClaude:
    """Tests for openai_content_to_claude function."""

    def test_string_content(self):
        """String content becomes text block."""
        result = openai_content_to_claude("Hello world")
        assert result == [{"type": "text", "text": "Hello world"}]

    def test_single_text_part(self):
        """Single text part converted."""
        content = [TextContent(type="text", text="Hello")]
        result = openai_content_to_claude(content)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_multiple_text_parts(self):
        """Multiple text parts converted."""
        content = [
            TextContent(type="text", text="First"),
            TextContent(type="text", text="Second"),
        ]
        result = openai_content_to_claude(content)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "First"}
        assert result[1] == {"type": "text", "text": "Second"}

    def test_text_and_image(self):
        """Mixed text and image content."""
        content = [
            TextContent(type="text", text="What's in this image?"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,abc"),
            ),
        ]
        result = openai_content_to_claude(content)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"

    def test_multiple_images(self):
        """Multiple images in content."""
        content = [
            TextContent(type="text", text="Compare these:"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,img1"),
            ),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:image/jpeg;base64,img2"),
            ),
        ]
        result = openai_content_to_claude(content)
        assert len(result) == 3
        assert result[1]["source"]["data"] == "img1"
        assert result[2]["source"]["data"] == "img2"

    def test_empty_content_list(self):
        """Empty content list returns empty result."""
        result = openai_content_to_claude([])
        assert result == []

    def test_pdf_and_image_mixed(self):
        """PDF and image in same content."""
        content = [
            TextContent(type="text", text="Review these documents:"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:application/pdf;base64,pdf1"),
            ),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,img1"),
            ),
        ]
        result = openai_content_to_claude(content)
        assert result[1]["type"] == "document"  # PDF
        assert result[2]["type"] == "image"  # Image


@pytest.mark.unit
class TestHasMultimodalContent:
    """Tests for has_multimodal_content function."""

    def test_text_only_messages(self):
        """Text-only messages return False."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        assert has_multimodal_content(messages) is False

    def test_single_image_message(self):
        """Message with image returns True."""
        messages = [
            Message(
                role="user",
                content=[
                    TextContent(type="text", text="What's this?"),
                    ImageUrlContent(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,abc"),
                    ),
                ],
            )
        ]
        assert has_multimodal_content(messages) is True

    def test_image_in_later_message(self):
        """Image in later message returns True."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(
                role="user",
                content=[
                    ImageUrlContent(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,abc"),
                    ),
                ],
            ),
        ]
        assert has_multimodal_content(messages) is True

    def test_mixed_text_array_no_image(self):
        """Content array with only text is not multimodal."""
        messages = [
            Message(
                role="user",
                content=[
                    TextContent(type="text", text="Part 1"),
                    TextContent(type="text", text="Part 2"),
                ],
            )
        ]
        assert has_multimodal_content(messages) is False

    def test_empty_messages(self):
        """Empty messages list returns False."""
        assert has_multimodal_content([]) is False

    def test_system_message_with_text(self):
        """System message with text only returns False."""
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        assert has_multimodal_content(messages) is False


@pytest.mark.unit
class TestExtractTextFromContent:
    """Tests for extract_text_from_content function."""

    def test_string_content(self):
        """String content returned as-is."""
        assert extract_text_from_content("Hello world") == "Hello world"

    def test_single_text_part(self):
        """Single text part extracted."""
        content = [TextContent(type="text", text="Hello")]
        assert extract_text_from_content(content) == "Hello"

    def test_multiple_text_parts(self):
        """Multiple text parts joined."""
        content = [
            TextContent(type="text", text="Hello"),
            TextContent(type="text", text="World"),
        ]
        result = extract_text_from_content(content)
        assert "Hello" in result
        assert "World" in result

    def test_image_placeholder(self):
        """Image gets placeholder text."""
        content = [
            TextContent(type="text", text="Look at this:"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,abc"),
            ),
        ]
        result = extract_text_from_content(content)
        assert "Look at this:" in result
        assert "[image: base64 data]" in result

    def test_pdf_placeholder(self):
        """PDF gets document placeholder text."""
        content = [
            TextContent(type="text", text="Check this PDF:"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:application/pdf;base64,abc"),
            ),
        ]
        result = extract_text_from_content(content)
        assert "[document: PDF base64 data]" in result

    def test_http_image_shows_url(self):
        """HTTP image shows URL in placeholder."""
        content = [
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="https://example.com/photo.jpg"),
            ),
        ]
        result = extract_text_from_content(content)
        assert "[image: https://example.com/photo.jpg]" in result

    def test_empty_content(self):
        """Empty content returns empty string."""
        assert extract_text_from_content([]) == ""


@pytest.mark.unit
class TestImageBlockTypeRegression:
    """Regression tests ensuring images produce correct block types."""

    @pytest.mark.parametrize("media_type,url_prefix", [
        ("image/png", "data:image/png;base64,iVBORw0KGgo="),
        ("image/jpeg", "data:image/jpeg;base64,/9j/4AAQ"),
        ("image/gif", "data:image/gif;base64,R0lGODlh"),
        ("image/webp", "data:image/webp;base64,UklGR"),
        ("image/svg+xml", "data:image/svg+xml;base64,PHN2Zw=="),
    ])
    def test_images_produce_image_blocks(self, media_type, url_prefix):
        """All image types produce 'image' blocks, not 'document'."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url=url_prefix),
        )
        result = openai_image_to_claude(content)
        assert result["type"] == "image", f"{media_type} should produce 'image' block"
        assert result["source"]["media_type"] == media_type

    def test_pdf_produces_document_block(self):
        """PDFs produce 'document' blocks, not 'image'."""
        content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:application/pdf;base64,JVBERi0="),
        )
        result = openai_image_to_claude(content)
        assert result["type"] == "document", "PDF should produce 'document' block"
        assert result["source"]["media_type"] == "application/pdf"
