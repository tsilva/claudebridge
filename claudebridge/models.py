"""OpenAI-compatible request/response models and model mapping."""

import re
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


# Tool-related types (OpenAI format)
class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None


class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"]
    function: ToolChoiceFunction


# Tool call types for responses
class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# Multimodal content types (OpenAI format)
class ImageUrl(BaseModel):
    url: str  # data:image/xxx;base64,... or https://...
    detail: Literal["auto", "low", "high"] | None = None  # For future SDK support


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


ContentPart = Annotated[Union[TextContent, ImageUrlContent], Field(discriminator='type')]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart] | None = None  # None when tool_calls present
    tool_calls: list[ToolCall] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # Tool calling support
    tools: list[Tool] | None = None
    tool_choice: ToolChoiceObject | str | None = None  # "auto", "none", or specific
    # Additional fields accepted for OpenRouter/OpenAI compatibility
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    seed: int | None = None
    user: str | None = None
    response_format: dict | None = None
    logit_bias: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    parallel_tool_calls: bool | None = None
    stream_options: dict | None = None


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"  # "stop" or "tool_calls"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: str | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "claude-code"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# Error response models (OpenAI format)
class ErrorDetail(BaseModel):
    message: str
    type: str  # "invalid_request_error", "server_error", etc.
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Model mapping (OpenRouter slug → Claude Code model identifier)
# ---------------------------------------------------------------------------

# Simple names that map directly to Claude Code model identifiers
SIMPLE_NAMES: set[str] = {"opus", "sonnet", "haiku"}

# Word-boundary pattern for matching model names in slugs
_MODEL_PATTERN = re.compile(
    r'(?:^|[^a-zA-Z])(' + '|'.join(sorted(SIMPLE_NAMES)) + r')(?:[^a-zA-Z]|$)',
    re.IGNORECASE,
)

# Available models for /api/v1/models endpoint (OpenRouter-style)
AVAILABLE_MODELS: list[dict[str, str]] = [
    {"slug": f"anthropic/claude-{name}", "name": f"Claude {name.capitalize()}"}
    for name in sorted(SIMPLE_NAMES)
]


class UnsupportedModelError(ValueError):
    """Raised when an unsupported model identifier is provided."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"Unsupported model: '{model}'. "
            f"Supported models: {', '.join(sorted(SIMPLE_NAMES))}, "
            f"or any slug containing 'opus', 'sonnet', or 'haiku'"
        )


def resolve_model(model: str) -> str:
    """Resolve an OpenRouter-style slug or simple name to a Claude Code model.

    Uses word-boundary matching to prevent false positives. Model names must appear
    as distinct segments separated by non-alpha characters (/, -, _, ., etc.).

    Args:
        model: Model identifier (OpenRouter slug or simple name)

    Returns:
        Claude Code model identifier (opus, sonnet, haiku)

    Raises:
        UnsupportedModelError: If model is not recognized
    """
    model_stripped = model.strip()
    model_lower = model_stripped.lower()

    # Already a simple Claude Code name (exact match)
    if model_lower in SIMPLE_NAMES:
        return model_lower

    # Word-boundary match: find model name as a distinct segment in the slug
    match = _MODEL_PATTERN.search(model_lower)
    if match:
        return match.group(1).lower()

    # Unknown model - raise error
    raise UnsupportedModelError(model)
