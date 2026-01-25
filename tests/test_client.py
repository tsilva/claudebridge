"""
Client integration tests for claude-bridge.

Prerequisites:
- Server must be running: claude-bridge

Usage:
- pytest tests/test_client.py -v
"""

import os

import httpx
import pytest
from openai import OpenAI

BASE_URL = os.environ.get("PROXY_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def server_available():
    """Check if server is running, skip tests if not."""
    try:
        with httpx.Client() as client:
            response = client.get(f"{BASE_URL}/health", timeout=2.0)
            if response.status_code == 200:
                return True
    except httpx.ConnectError:
        pass
    pytest.skip(f"Server not running at {BASE_URL}")


@pytest.fixture
def openai_client():
    """Create OpenAI client configured for the proxy."""
    return OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")


class TestHealthCheck:
    """Tests for the /health endpoint."""

    def test_health_check(self, server_available):
        """Verify /health endpoint returns {"status": "ok"}."""
        with httpx.Client() as client:
            response = client.get(f"{BASE_URL}/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestListModels:
    """Tests for the /v1/models endpoint."""

    def test_list_models(self, server_available):
        """Verify /v1/models returns opus, sonnet, haiku."""
        with httpx.Client() as client:
            response = client.get(f"{BASE_URL}/v1/models")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        model_ids = [model["id"] for model in data["data"]]
        assert "opus" in model_ids
        assert "sonnet" in model_ids
        assert "haiku" in model_ids

    def test_list_models_openai_client(self, server_available, openai_client):
        """Verify models can be listed via OpenAI client."""
        models = openai_client.models.list()

        model_ids = [model.id for model in models.data]
        assert "opus" in model_ids
        assert "sonnet" in model_ids
        assert "haiku" in model_ids


class TestChatCompletion:
    """Tests for the /v1/chat/completions endpoint."""

    def test_chat_completion(self, server_available, openai_client):
        """Send a simple prompt, verify response structure."""
        response = openai_client.chat.completions.create(
            model="haiku",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )

        # Verify response structure
        assert response.id.startswith("chatcmpl-")
        assert response.model == "haiku"
        assert response.object == "chat.completion"
        assert len(response.choices) == 1

        choice = response.choices[0]
        assert choice.index == 0
        assert choice.finish_reason == "stop"
        assert choice.message.role == "assistant"
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0

    def test_chat_completion_streaming(self, server_available, openai_client):
        """Test streaming response format."""
        stream = openai_client.chat.completions.create(
            model="haiku",
            messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
            stream=True,
        )

        chunks = list(stream)

        # Should have at least initial chunk, content, and final chunk
        assert len(chunks) >= 2

        # First chunk should have role
        first_chunk = chunks[0]
        assert first_chunk.id.startswith("chatcmpl-")
        assert first_chunk.object == "chat.completion.chunk"

        # Collect all content
        content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        assert len(content) > 0

        # Last chunk should have finish_reason
        last_chunk = chunks[-1]
        assert last_chunk.choices[0].finish_reason == "stop"

    def test_chat_completion_with_system_message(self, server_available, openai_client):
        """Test chat completion with system message."""
        response = openai_client.chat.completions.create(
            model="haiku",
            messages=[
                {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
                {"role": "user", "content": "Say hello."},
            ],
        )

        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0

    def test_model_name_mapping(self, server_available):
        """Test that various model name formats are accepted."""
        model_names = ["sonnet", "claude-sonnet", "claude-3-sonnet", "claude-3.5-sonnet"]

        with httpx.Client(timeout=60.0) as client:
            for model_name in model_names:
                response = client.post(
                    f"{BASE_URL}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "Say 'ok'"}],
                    },
                )
                assert response.status_code == 200, f"Failed for model: {model_name}"
