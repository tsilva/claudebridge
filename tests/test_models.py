"""
Unit tests for Pydantic models.

These tests verify model validation, serialization, and edge cases.

Usage:
- pytest tests/test_models.py -v
"""

import json

import pytest

from claudebridge.models import (
    # Request models
    ChatCompletionRequest,
    Message,
    TextContent,
    ImageUrlContent,
    ImageUrl,
    Tool,
    FunctionDefinition,
    ToolChoiceObject,
    ToolChoiceFunction,
    # Response models
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    StreamChoice,
    DeltaMessage,
    Usage,
    ModelInfo,
    ModelList,
    # Tool call models
    ToolCall,
    FunctionCall,
    # Error models
    ErrorDetail,
    ErrorResponse,
)


@pytest.mark.unit
class TestMessageModel:
    """Tests for Message model validation."""

    def test_simple_text_message(self):
        """Basic text message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_assistant_message(self):
        """Assistant message."""
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_system_message(self):
        """System message."""
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_message_with_none_content(self):
        """Message with None content (for tool calls)."""
        msg = Message(role="assistant", content=None)
        assert msg.content is None

    def test_message_with_tool_calls(self):
        """Message with tool_calls."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        )
        msg = Message(role="assistant", content=None, tool_calls=[tool_call])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_message_with_multimodal_content(self):
        """Message with text and image content."""
        content = [
            TextContent(type="text", text="What's in this image?"),
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrl(url="data:image/png;base64,abc123"),
            ),
        ]
        msg = Message(role="user", content=content)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image_url"

    def test_message_serialization(self):
        """Verify message serializes to JSON correctly."""
        msg = Message(role="user", content="Test")
        data = msg.model_dump()
        assert data["role"] == "user"
        assert data["content"] == "Test"
        assert data["tool_calls"] is None


@pytest.mark.unit
class TestImageUrlModel:
    """Tests for ImageUrl model."""

    def test_simple_url(self):
        """Basic URL."""
        img = ImageUrl(url="https://example.com/image.png")
        assert img.url == "https://example.com/image.png"
        assert img.detail is None

    def test_data_url(self):
        """Base64 data URL."""
        img = ImageUrl(url="data:image/png;base64,iVBORw0KGgo=")
        assert img.url.startswith("data:")

    def test_detail_parameter_auto(self):
        """Detail parameter set to auto."""
        img = ImageUrl(url="https://example.com/image.png", detail="auto")
        assert img.detail == "auto"

    def test_detail_parameter_low(self):
        """Detail parameter set to low."""
        img = ImageUrl(url="https://example.com/image.png", detail="low")
        assert img.detail == "low"

    def test_detail_parameter_high(self):
        """Detail parameter set to high."""
        img = ImageUrl(url="https://example.com/image.png", detail="high")
        assert img.detail == "high"


@pytest.mark.unit
class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_minimal_request(self):
        """Minimal valid request."""
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.model == "sonnet"
        assert len(req.messages) == 1
        assert req.stream is False
        assert req.temperature is None
        assert req.max_tokens is None

    def test_request_with_streaming(self):
        """Request with streaming enabled."""
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[Message(role="user", content="Hello")],
            stream=True,
        )
        assert req.stream is True

    def test_request_with_all_optional_fields(self):
        """Request with all optional fields."""
        req = ChatCompletionRequest(
            model="anthropic/claude-sonnet-4",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=1000,
            stream=True,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=["END"],
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 1000
        assert req.top_p == 0.9
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.5
        assert req.stop == ["END"]

    def test_request_with_tools(self):
        """Request with tool definitions."""
        tool = Tool(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get current weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        )
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[Message(role="user", content="Weather in NYC?")],
            tools=[tool],
        )
        assert len(req.tools) == 1
        assert req.tools[0].function.name == "get_weather"

    def test_request_with_tool_choice_auto(self):
        """Request with tool_choice as string."""
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[Message(role="user", content="Hello")],
            tool_choice="auto",
        )
        assert req.tool_choice == "auto"

    def test_request_with_tool_choice_specific(self):
        """Request with specific tool choice."""
        tool_choice = ToolChoiceObject(
            type="function",
            function=ToolChoiceFunction(name="get_weather"),
        )
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[Message(role="user", content="Hello")],
            tool_choice=tool_choice,
        )
        assert req.tool_choice.function.name == "get_weather"

    def test_request_with_stop_sequence_string(self):
        """Stop sequence as string."""
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[Message(role="user", content="Hello")],
            stop="END",
        )
        assert req.stop == "END"

    def test_request_with_stop_sequence_list(self):
        """Stop sequence as list."""
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[Message(role="user", content="Hello")],
            stop=["END", "STOP"],
        )
        assert req.stop == ["END", "STOP"]

    def test_request_with_openrouter_compat_fields(self):
        """Request accepts all OpenRouter/OpenAI compatibility fields."""
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[Message(role="user", content="Hello")],
            n=1,
            seed=42,
            user="test-user",
            response_format={"type": "json_object"},
            logit_bias={"123": -100},
            logprobs=True,
            top_logprobs=5,
            parallel_tool_calls=True,
            stream_options={"include_usage": True},
        )
        assert req.n == 1
        assert req.seed == 42
        assert req.user == "test-user"
        assert req.response_format == {"type": "json_object"}
        assert req.logprobs is True
        assert req.top_logprobs == 5
        assert req.parallel_tool_calls is True
        assert req.stream_options == {"include_usage": True}


@pytest.mark.unit
class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse model."""

    def test_basic_response(self):
        """Basic completion response."""
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[
                Choice(
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
        )
        assert resp.id == "chatcmpl-123"
        assert resp.object == "chat.completion"
        assert resp.model == "sonnet"
        assert len(resp.choices) == 1
        assert resp.choices[0].finish_reason == "stop"

    def test_response_with_usage(self):
        """Response with token usage."""
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[
                Choice(
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_response_with_tool_calls(self):
        """Response with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        )
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[
                Choice(
                    message=Message(role="assistant", content=None, tool_calls=[tool_call]),
                    finish_reason="tool_calls",
                )
            ],
        )
        assert resp.choices[0].finish_reason == "tool_calls"
        assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"

    def test_default_usage(self):
        """Default usage is zero."""
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[
                Choice(
                    message=Message(role="assistant", content="Hi"),
                    finish_reason="stop",
                )
            ],
        )
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0
        assert resp.usage.total_tokens == 0

    def test_system_fingerprint_field(self):
        """Response accepts system_fingerprint."""
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[
                Choice(
                    message=Message(role="assistant", content="Hi"),
                    finish_reason="stop",
                )
            ],
            system_fingerprint="fp_abc123",
        )
        assert resp.system_fingerprint == "fp_abc123"

    def test_system_fingerprint_default_none(self):
        """system_fingerprint defaults to None."""
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[
                Choice(
                    message=Message(role="assistant", content="Hi"),
                    finish_reason="stop",
                )
            ],
        )
        assert resp.system_fingerprint is None


@pytest.mark.unit
class TestChatCompletionChunk:
    """Tests for streaming chunk model."""

    def test_initial_chunk(self):
        """Initial chunk with role."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.role == "assistant"

    def test_content_chunk(self):
        """Chunk with content."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[StreamChoice(delta=DeltaMessage(content="Hello"))],
        )
        assert chunk.choices[0].delta.content == "Hello"

    def test_final_chunk(self):
        """Final chunk with finish_reason."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        )
        assert chunk.choices[0].finish_reason == "stop"

    def test_chunk_with_usage(self):
        """Chunk with usage data (for final streaming chunk)."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert chunk.usage.prompt_tokens == 10
        assert chunk.usage.total_tokens == 15

    def test_chunk_usage_default_none(self):
        """Chunk usage defaults to None."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[StreamChoice(delta=DeltaMessage(content="Hi"))],
        )
        assert chunk.usage is None

    def test_chunk_system_fingerprint(self):
        """Chunk accepts system_fingerprint."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1234567890,
            model="sonnet",
            choices=[StreamChoice(delta=DeltaMessage(content="Hi"))],
            system_fingerprint="fp_abc",
        )
        assert chunk.system_fingerprint == "fp_abc"


@pytest.mark.unit
class TestErrorModels:
    """Tests for error response models."""

    def test_error_detail_minimal(self):
        """Minimal error detail."""
        error = ErrorDetail(
            message="Invalid model",
            type="invalid_request_error",
        )
        assert error.message == "Invalid model"
        assert error.type == "invalid_request_error"
        assert error.param is None
        assert error.code is None

    def test_error_detail_full(self):
        """Full error detail with all fields."""
        error = ErrorDetail(
            message="Model not found",
            type="invalid_request_error",
            param="model",
            code="model_not_found",
        )
        assert error.param == "model"
        assert error.code == "model_not_found"

    def test_error_response(self):
        """Error response wrapper."""
        resp = ErrorResponse(
            error=ErrorDetail(
                message="Test error",
                type="server_error",
            )
        )
        assert resp.error.message == "Test error"

    def test_error_response_serialization(self):
        """Error response serializes correctly."""
        resp = ErrorResponse(
            error=ErrorDetail(
                message="Bad request",
                type="invalid_request_error",
                param="model",
            )
        )
        data = resp.model_dump()
        assert "error" in data
        assert data["error"]["message"] == "Bad request"
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["param"] == "model"


@pytest.mark.unit
class TestToolModels:
    """Tests for tool-related models."""

    def test_function_definition_minimal(self):
        """Minimal function definition."""
        func = FunctionDefinition(name="my_function")
        assert func.name == "my_function"
        assert func.description is None
        assert func.parameters is None

    def test_function_definition_full(self):
        """Full function definition."""
        func = FunctionDefinition(
            name="get_weather",
            description="Get the current weather",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        )
        assert func.name == "get_weather"
        assert "Get the current weather" in func.description
        assert func.parameters["type"] == "object"

    def test_tool_model(self):
        """Tool model."""
        tool = Tool(
            type="function",
            function=FunctionDefinition(name="test_func"),
        )
        assert tool.type == "function"
        assert tool.function.name == "test_func"

    def test_tool_call_model(self):
        """Tool call model."""
        call = ToolCall(
            id="call_abc123",
            function=FunctionCall(
                name="get_weather",
                arguments='{"city": "New York"}',
            ),
        )
        assert call.id == "call_abc123"
        assert call.type == "function"
        assert call.function.name == "get_weather"
        # Verify arguments is valid JSON
        args = json.loads(call.function.arguments)
        assert args["city"] == "New York"


@pytest.mark.unit
class TestModelInfo:
    """Tests for model listing models."""

    def test_model_info_minimal(self):
        """Model info with defaults."""
        info = ModelInfo(id="anthropic/claude-sonnet-4")
        assert info.id == "anthropic/claude-sonnet-4"
        assert info.object == "model"
        assert info.created == 0
        assert info.owned_by == "claude-code"

    def test_model_list(self):
        """Model list."""
        models = ModelList(
            data=[
                ModelInfo(id="anthropic/claude-sonnet-4"),
                ModelInfo(id="anthropic/claude-opus-4.5"),
            ]
        )
        assert models.object == "list"
        assert len(models.data) == 2


@pytest.mark.unit
class TestUsageModel:
    """Tests for Usage model."""

    def test_default_usage(self):
        """Default values are zero."""
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_usage(self):
        """Custom usage values."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
