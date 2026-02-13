"""
Unit tests for server functions and endpoint handlers.

These tests mock the Claude SDK and test server logic in isolation.

Usage:
- pytest tests/test_server.py -v
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from claudebridge.models import (
    Message,
    TextContent,
    ImageUrlContent,
    ImageUrl,
    Tool,
    FunctionDefinition,
)
from claudebridge.server import (
    format_messages,
    format_multimodal_messages,
    build_tool_prompt,
    parse_tool_response,
    ClaudeResponse,
    app,
)
from claudebridge.image_utils import extract_text_from_content


@pytest.mark.unit
class TestFormatMessages:
    """Tests for format_messages function."""

    def test_single_user_message(self):
        """Single user message formats correctly."""
        messages = [Message(role="user", content="Hello")]
        result = format_messages(messages)
        assert "User: Hello" in result

    def test_single_system_message(self):
        """System message appears at start."""
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        result = format_messages(messages)
        assert result.startswith("System: Be helpful")
        assert "User: Hello" in result

    def test_conversation_format(self):
        """Multi-turn conversation formats correctly."""
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="How are you?"),
        ]
        result = format_messages(messages)
        assert "User: Hi" in result
        assert "Assistant: Hello!" in result
        assert "User: How are you?" in result

    def test_empty_assistant_message_skipped(self):
        """Empty assistant messages are skipped."""
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content=""),
            Message(role="user", content="Hello?"),
        ]
        result = format_messages(messages)
        # Should not have "Assistant:" for empty message
        assert result.count("Assistant:") == 0

    def test_none_content_handled(self):
        """None content (for tool calls) handled gracefully."""
        messages = [
            Message(role="assistant", content=None),
            Message(role="user", content="Continue"),
        ]
        result = format_messages(messages)
        # Should not crash, assistant line might be empty
        assert "User: Continue" in result


@pytest.mark.unit
class TestExtractTextContent:
    """Tests for extract_text_from_content used by server formatting functions."""

    def test_single_text_part(self):
        """Single text part extracted."""
        content = [TextContent(type="text", text="Hello world")]
        result = extract_text_from_content(content)
        assert result == "Hello world"

    def test_multiple_text_parts(self):
        """Multiple text parts joined with space."""
        content = [
            TextContent(type="text", text="Hello"),
            TextContent(type="text", text="World"),
        ]
        result = extract_text_from_content(content)
        assert result == "Hello World"

    def test_mixed_content_text_and_image(self):
        """Mixed content extracts text and image placeholders."""
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

    def test_empty_content_list(self):
        """Empty content list returns empty string."""
        result = extract_text_from_content([])
        assert result == ""


@pytest.mark.unit
class TestBuildToolPrompt:
    """Tests for build_tool_prompt function."""

    def test_single_tool_prompt(self):
        """Single tool creates appropriate prompt."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                ),
            )
        ]
        result = build_tool_prompt(tools)
        assert "JSON object" in result
        assert '"type": "object"' in result

    def test_multiple_tools_prompt(self):
        """Multiple tools creates choice prompt."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(name="tool1"),
            ),
            Tool(
                type="function",
                function=FunctionDefinition(name="tool2"),
            ),
        ]
        result = build_tool_prompt(tools)
        assert "tool1" in result
        assert "tool2" in result
        assert "function" in result.lower()

    def test_tool_prompt_contains_schema(self):
        """Tool prompt includes function schema."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="calculate",
                    parameters={
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string"},
                            "numbers": {"type": "array"},
                        },
                    },
                ),
            )
        ]
        result = build_tool_prompt(tools)
        assert "operation" in result
        assert "numbers" in result


@pytest.mark.unit
class TestParseToolResponse:
    """Tests for parse_tool_response function."""

    def test_parse_json_block(self):
        """Parse JSON in code block."""
        text = '```json\n{"city": "NYC"}\n```'
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(name="get_weather"),
            )
        ]
        remaining, tool_calls = parse_tool_response(text, tools)
        assert remaining == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["city"] == "NYC"

    def test_parse_raw_json(self):
        """Parse raw JSON object."""
        text = '{"value": 42}'
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(name="process"),
            )
        ]
        remaining, tool_calls = parse_tool_response(text, tools)
        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function.arguments)
        assert args["value"] == 42

    def test_parse_multi_tool_response(self):
        """Parse response with function name for multiple tools."""
        text = '{"function": "tool2", "arguments": {"x": 1}}'
        tools = [
            Tool(type="function", function=FunctionDefinition(name="tool1")),
            Tool(type="function", function=FunctionDefinition(name="tool2")),
        ]
        remaining, tool_calls = parse_tool_response(text, tools)
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "tool2"

    def test_no_json_returns_original_text(self):
        """No JSON returns original text."""
        text = "This is just plain text with no JSON."
        tools = [
            Tool(type="function", function=FunctionDefinition(name="test")),
        ]
        remaining, tool_calls = parse_tool_response(text, tools)
        assert remaining == text
        assert len(tool_calls) == 0

    def test_invalid_json_returns_text(self):
        """Invalid JSON returns original text."""
        text = "{invalid json here}"
        tools = [
            Tool(type="function", function=FunctionDefinition(name="test")),
        ]
        remaining, tool_calls = parse_tool_response(text, tools)
        # Should return original text since JSON is invalid
        assert remaining == text or len(tool_calls) == 0

    def test_tool_call_has_id(self):
        """Tool calls have unique IDs."""
        text = '{"key": "value"}'
        tools = [
            Tool(type="function", function=FunctionDefinition(name="test")),
        ]
        _, tool_calls = parse_tool_response(text, tools)
        assert tool_calls[0].id.startswith("call_")


@pytest.mark.unit
class TestClaudeResponse:
    """Tests for ClaudeResponse container class."""

    def test_empty_response(self):
        """Empty response defaults."""
        resp = ClaudeResponse()
        assert resp.text == ""
        assert resp.tool_calls == []
        assert resp.usage is None
        assert not resp.has_tool_calls
        assert resp.finish_reason == "stop"

    def test_response_with_text(self):
        """Response with text."""
        resp = ClaudeResponse()
        resp.text = "Hello world"
        assert resp.text == "Hello world"
        assert resp.finish_reason == "stop"

    def test_response_with_tool_calls(self):
        """Response with tool calls."""
        from claudebridge.models import ToolCall, FunctionCall

        resp = ClaudeResponse()
        resp.tool_calls = [
            ToolCall(
                id="call_123",
                function=FunctionCall(name="test", arguments="{}"),
            )
        ]
        assert resp.has_tool_calls
        assert resp.finish_reason == "tool_calls"

    def test_get_usage_with_data(self):
        """get_usage returns OpenAI-format usage."""
        resp = ClaudeResponse()
        resp.usage = {"input_tokens": 100, "output_tokens": 50}
        usage = resp.get_usage()
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_get_usage_without_data(self):
        """get_usage returns zeros when no data."""
        resp = ClaudeResponse()
        usage = resp.get_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_get_usage_partial_data(self):
        """get_usage handles partial data."""
        resp = ClaudeResponse()
        resp.usage = {"input_tokens": 50}  # Missing output_tokens
        usage = resp.get_usage()
        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 50


@pytest.mark.unit
class TestFormatMultimodalMessages:
    """Tests for format_multimodal_messages function."""

    def test_text_with_image(self):
        """Text and image content formatted correctly."""
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
        result = format_multimodal_messages(messages)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert "User:" in result[0]["text"]
        assert result[1]["type"] == "image"

    def test_system_message_prepended(self):
        """System message prepended to content."""
        messages = [
            Message(role="system", content="Be helpful"),
            Message(
                role="user",
                content=[
                    TextContent(type="text", text="Hi"),
                    ImageUrlContent(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,abc"),
                    ),
                ],
            ),
        ]
        result = format_multimodal_messages(messages)
        assert result[0]["text"].startswith("System:")

    def test_multiple_images(self):
        """Multiple images in content."""
        messages = [
            Message(
                role="user",
                content=[
                    TextContent(type="text", text="Compare these:"),
                    ImageUrlContent(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,abc"),
                    ),
                    ImageUrlContent(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,xyz"),
                    ),
                ],
            )
        ]
        result = format_multimodal_messages(messages)
        image_blocks = [b for b in result if b["type"] == "image"]
        assert len(image_blocks) == 2


# Test the FastAPI app with TestClient
@pytest.fixture
def test_client():
    """Create test client with mocked pool."""
    # We need to mock the pool to avoid connecting to real SDK
    with patch("claudebridge.server.pool") as mock_pool:
        # Create a mock that can be used with async context manager
        mock_client = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        yield TestClient(app, raise_server_exceptions=False)


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, test_client):
        """Health endpoint returns status ok with version and pool info."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


@pytest.mark.unit
class TestModelsEndpoint:
    """Tests for /api/v1/models endpoint."""

    def test_models_returns_list(self, test_client):
        """Models endpoint returns list."""
        response = test_client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0

    def test_models_have_openrouter_format(self, test_client):
        """Models have OpenRouter-style slugs."""
        response = test_client.get("/api/v1/models")
        data = response.json()
        for model in data["data"]:
            assert model["id"].startswith("anthropic/claude-")
            assert model["object"] == "model"


@pytest.mark.unit
class TestChatCompletionsValidation:
    """Tests for /api/v1/chat/completions request validation."""

    def test_missing_model_error(self, test_client):
        """Missing model returns error."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_missing_messages_error(self, test_client):
        """Missing messages returns error."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={"model": "sonnet"},
        )
        assert response.status_code == 422

    def test_invalid_model_error(self, test_client):
        """Invalid model returns OpenAI-format error."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "invalid-model-xyz",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == "model"

    def test_empty_messages_accepted(self, test_client):
        """Empty messages list is accepted by Pydantic but may fail later."""
        # Note: This tests Pydantic validation, not business logic
        response = test_client.post(
            "/api/v1/chat/completions",
            json={"model": "sonnet", "messages": []},
        )
        # Empty messages might be accepted by validation
        # The actual error would come from the SDK call

    def test_extra_openrouter_params_accepted(self, test_client):
        """Extra OpenRouter/OpenAI parameters don't cause request rejection."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.5,
                "seed": 42,
                "n": 1,
                "user": "test-user",
                "response_format": {"type": "json_object"},
                "logprobs": True,
                "top_logprobs": 5,
            },
        )
        # Should not be 422 (validation error)
        assert response.status_code != 422


@pytest.mark.unit
class TestErrorResponseFormat:
    """Tests for error response formatting."""

    def test_invalid_model_error_format(self, test_client):
        """Invalid model error has correct format."""
        response = test_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "not-a-real-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "type" in data["error"]
        assert data["error"]["code"] == "model_not_found"
