"""
Vision/OCR integration tests for claudebridge.

Prerequisites:
- Server must be running: claudebridge

Unit tests for image utilities are in test_image_utils.py.

Usage:
- pytest tests/test_vision.py -v
"""

import base64
from pathlib import Path

import pytest

from claudebridge.client import BridgeClient


# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures"
OCR_TEST_IMAGE = FIXTURES_DIR / "ocr_test_document.png"


@pytest.fixture(scope="module")
def client():
    """Create BridgeClient for testing."""
    c = BridgeClient()
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
