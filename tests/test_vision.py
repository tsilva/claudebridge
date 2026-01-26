"""
Vision/OCR tests for claude-code-bridge.

Prerequisites:
- Server must be running for integration tests: claude-code-bridge

Usage:
- Unit tests: pytest tests/test_vision.py::TestImageUtils -v
- Integration tests: pytest tests/test_vision.py::TestVisionIntegration -v
"""

import base64
from pathlib import Path

import pytest

from claude_code_bridge.image_utils import (
    parse_data_url,
    is_http_url,
    is_data_url,
    openai_image_to_claude,
    openai_content_to_claude,
    has_multimodal_content,
    extract_text_from_content,
)
from claude_code_bridge.models import (
    Message,
    TextContent,
    ImageUrlContent,
    ImageUrl,
)
from claude_code_bridge.client import BridgeClient


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures"
OCR_TEST_IMAGE = FIXTURES_DIR / "ocr_test_document.png"


class TestImageUtils:
    """Unit tests for image utility functions."""

    def test_parse_data_url_png(self):
        """Parse PNG data URL."""
        url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        media_type, data = parse_data_url(url)

        assert media_type == "image/png"
        assert data == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    def test_parse_data_url_jpeg(self):
        """Parse JPEG data URL."""
        url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        media_type, data = parse_data_url(url)

        assert media_type == "image/jpeg"
        assert data == "/9j/4AAQSkZJRg=="

    def test_parse_data_url_invalid(self):
        """Parsing invalid data URL raises ValueError."""
        with pytest.raises(ValueError):
            parse_data_url("not-a-data-url")

    def test_is_http_url(self):
        """Check HTTP URL detection."""
        assert is_http_url("https://example.com/image.png")
        assert is_http_url("http://example.com/image.png")
        assert not is_http_url("data:image/png;base64,...")
        assert not is_http_url("/local/path/image.png")

    def test_is_data_url(self):
        """Check data URL detection."""
        assert is_data_url("data:image/png;base64,...")
        assert not is_data_url("https://example.com/image.png")
        assert not is_data_url("/local/path/image.png")

    def test_openai_image_to_claude_base64(self):
        """Convert base64 image URL to Claude format."""
        image_content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="data:image/png;base64,abc123")
        )
        result = openai_image_to_claude(image_content)

        assert result == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123",
            }
        }

    def test_openai_image_to_claude_http(self):
        """Convert HTTP image URL to Claude format."""
        image_content = ImageUrlContent(
            type="image_url",
            image_url=ImageUrl(url="https://example.com/image.png")
        )
        result = openai_image_to_claude(image_content)

        assert result == {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.png",
            }
        }

    def test_openai_content_to_claude_text_only(self):
        """Convert text-only content."""
        result = openai_content_to_claude("Hello, world!")

        assert result == [{"type": "text", "text": "Hello, world!"}]

    def test_openai_content_to_claude_text_parts(self):
        """Convert list of text content parts."""
        content = [
            TextContent(type="text", text="First part."),
            TextContent(type="text", text="Second part."),
        ]
        result = openai_content_to_claude(content)

        assert result == [
            {"type": "text", "text": "First part."},
            {"type": "text", "text": "Second part."},
        ]

    def test_openai_content_to_claude_with_image(self):
        """Convert content with image."""
        content = [
            TextContent(type="text", text="What's in this image?"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,abc123")
            ),
        ]
        result = openai_content_to_claude(content)

        assert result == [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abc123",
                }
            },
        ]

    def test_has_multimodal_content_text_only(self):
        """Text-only messages are not multimodal."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        assert not has_multimodal_content(messages)

    def test_has_multimodal_content_with_image(self):
        """Messages with images are multimodal."""
        messages = [
            Message(
                role="user",
                content=[
                    TextContent(type="text", text="What's this?"),
                    ImageUrlContent(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,abc")
                    ),
                ]
            ),
        ]
        assert has_multimodal_content(messages)

    def test_extract_text_from_content_string(self):
        """Extract text from string content."""
        assert extract_text_from_content("Hello") == "Hello"

    def test_extract_text_from_content_parts(self):
        """Extract text from content parts with image placeholder."""
        content = [
            TextContent(type="text", text="Look at this:"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,abc")
            ),
        ]
        result = extract_text_from_content(content)

        assert "Look at this:" in result
        assert "[image: base64 data]" in result


@pytest.fixture(scope="module")
def client():
    """Create BridgeClient and skip if server not running."""
    c = BridgeClient()
    if not c.health_check():
        pytest.skip(f"Server not running at {c.base_url}")
    yield c
    c.close_sync()


class TestVisionIntegration:
    """Integration tests for vision/OCR functionality (requires running server)."""

    @pytest.fixture
    def test_image_base64(self):
        """Load test image as base64."""
        if not OCR_TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {OCR_TEST_IMAGE}")
        with open(OCR_TEST_IMAGE, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def test_image_ocr(self, client, test_image_base64):
        """Test OCR with the generated document image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What text do you see in this image? Reply with only the text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_base64}"}}
                ]
            }
        ]

        response = client.complete_messages_sync(messages, stream=False)

        # The test image contains "HELLO WORLD 2025"
        assert "HELLO" in response.upper()
        assert "WORLD" in response.upper()
        assert "2025" in response

    def test_image_ocr_streaming(self, client, test_image_base64):
        """Test OCR with streaming response."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What text do you see? Reply with only the text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_base64}"}}
                ]
            }
        ]

        response = client.complete_messages_sync(messages, stream=True)

        assert "HELLO" in response.upper()
        assert "WORLD" in response.upper()

    def test_mixed_text_and_image(self, client, test_image_base64):
        """Test message with both text context and image."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts text from images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a test document. Extract the text shown in the image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_base64}"}}
                ]
            }
        ]

        response = client.complete_messages_sync(messages, stream=False)

        # Should still extract the text
        assert "HELLO" in response.upper() or "2025" in response

    def test_text_only_still_works(self, client):
        """Ensure text-only messages still work after multimodal changes."""
        messages = [
            {"role": "user", "content": "Say 'vision test passed' and nothing else."}
        ]

        response = client.complete_messages_sync(messages, stream=False)

        assert "vision" in response.lower() or "test" in response.lower() or "passed" in response.lower()
