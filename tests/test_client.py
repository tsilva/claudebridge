"""
Client integration tests for claude-code-bridge.

Prerequisites:
- Server must be running: claude-code-bridge

Usage:
- pytest tests/test_client.py -v
"""

import pytest

from claude_code_bridge.client import BridgeClient


@pytest.fixture(scope="module")
def client():
    """Create BridgeClient and skip if server not running."""
    c = BridgeClient()
    if not c.health_check():
        pytest.skip(f"Server not running at {c.base_url}")
    yield c
    c.close_sync()


class TestHealthCheck:
    """Tests for the /health endpoint."""

    def test_health_check(self, client):
        """Verify health_check returns True when server is up."""
        assert client.health_check()


class TestListModels:
    """Tests for the /v1/models endpoint."""

    def test_list_models(self, client):
        """Verify list_models returns opus, sonnet, haiku."""
        models = client.list_models()

        assert "opus" in models
        assert "sonnet" in models
        assert "haiku" in models


class TestChatCompletion:
    """Tests for the /v1/chat/completions endpoint."""

    def test_sync_non_streaming(self, client):
        """Test synchronous non-streaming completion."""
        response = client.complete_sync("Say 'hello' and nothing else.", stream=False)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_sync_streaming(self, client):
        """Test synchronous streaming completion (collects chunks)."""
        response = client.complete_sync("Say 'hi' and nothing else.", stream=True)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_with_system_message(self, client):
        """Test completion with system message."""
        messages = [
            {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            {"role": "user", "content": "Say hello."},
        ]
        response = client.complete_messages_sync(messages, stream=False)

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_streaming(self, client):
        """Test async streaming response."""
        chunks = []
        async for chunk in client.stream("Say 'hello'"):
            chunks.append(chunk)

        content = "".join(chunks)
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_async_complete(self, client):
        """Test async complete method."""
        response = await client.complete("Say 'hi'", stream=True)

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_non_streaming(self, client):
        """Test async non-streaming completion."""
        response = await client.complete("Say 'ok'", stream=False)

        assert isinstance(response, str)
        assert len(response) > 0


class TestModelSelection:
    """Tests for model selection."""

    def test_model_parameter(self, client):
        """Test that model parameter is accepted."""
        response = client.complete_sync("Say 'test'", model="haiku", stream=False)

        assert isinstance(response, str)
        assert len(response) > 0
