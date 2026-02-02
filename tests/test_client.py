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
    """Create BridgeClient for testing."""
    c = BridgeClient()
    yield c
    c.close_sync()


class TestHealthCheck:
    """Tests for the /health endpoint."""

    def test_health_check(self, client):
        """Verify health_check returns True when server is up."""
        assert client.health_check()


class TestListModels:
    """Tests for the /api/v1/models endpoint."""

    def test_list_models(self, client):
        """Verify list_models returns OpenRouter-style model slugs."""
        models = client.list_models()

        # Check for OpenRouter-style slugs
        assert any("opus" in m for m in models)
        assert any("sonnet" in m for m in models)
        assert any("haiku" in m for m in models)


class TestChatCompletion:
    """Tests for the /api/v1/chat/completions endpoint."""

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

    @pytest.mark.parametrize("model,expected_name", [
        ("haiku", "haiku"),
        ("sonnet", "sonnet"),
        ("opus", "opus"),
    ])
    def test_model_switching(self, client, model, expected_name):
        """Test that model switching works by asking the model to identify itself."""
        prompt = (
            "What is your exact model name and version? "
            "Reply with ONLY the model name, nothing else. "
            "For example: 'Claude 3.5 Haiku' or 'Claude Sonnet 4'"
        )
        response = client.complete_sync(prompt, model=model, stream=False)

        # The response should contain the expected model family name
        assert expected_name.lower() in response.lower(), (
            f"Expected '{expected_name}' in response when using model='{model}', "
            f"but got: {response}"
        )
