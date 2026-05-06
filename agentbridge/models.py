"""OpenAI-compatible request/response models and provider model mapping."""

import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

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

ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None  # None when tool_calls present
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # Present when role="tool"
    name: str | None = None  # Tool name when role="tool"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    reasoning: dict[str, Any] | None = None
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
    owned_by: str = "agentbridge"


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
# Provider model mapping
# ---------------------------------------------------------------------------

# Simple names that map directly to Claude Code model identifiers
CLAUDE_SIMPLE_NAMES: set[str] = {"opus", "sonnet", "haiku"}

# Backwards-compatible export used by tests and callers.
SIMPLE_NAMES = CLAUDE_SIMPLE_NAMES

PROVIDER_NAMES: set[str] = {"claudecode", "codex", "openrouter"}

CODEX_MODEL_SLUGS: set[str] = {
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2",
}

OPENROUTER_EXAMPLE_SLUGS: set[str] = {
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
    "openai/gpt-5",
}


@dataclass(frozen=True)
class ModelResolution:
    """Resolved provider and backend model identifier."""

    provider: Literal["claudecode", "codex", "openrouter"]
    model: str

# Word-boundary pattern for matching model names in slugs
_MODEL_PATTERN = re.compile(
    r'(?:^|[^a-zA-Z])(' + '|'.join(sorted(CLAUDE_SIMPLE_NAMES)) + r')(?:[^a-zA-Z]|$)',
    re.IGNORECASE,
)
# Available models for /api/v1/models endpoint.
AVAILABLE_MODELS: list[dict[str, str]] = [
    *[
        {
            "slug": f"claudecode/{name}",
            "name": f"Claude {name.capitalize()}",
            "owned_by": "claude-code",
        }
        for name in sorted(CLAUDE_SIMPLE_NAMES)
    ],
    *[
        {"slug": f"codex/{name}", "name": name.upper(), "owned_by": "codex-cli"}
        for name in sorted(CODEX_MODEL_SLUGS)
    ],
    *[
        {
            "slug": f"openrouter/{name}",
            "name": name,
            "owned_by": "openrouter",
        }
        for name in sorted(OPENROUTER_EXAMPLE_SLUGS)
    ],
]


class UnsupportedModelError(ValueError):
    """Raised when an unsupported model identifier is provided."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"Unsupported model: '{model}'. "
            "Model IDs must start with a provider namespace. "
            "Use claudecode/<opus|sonnet|haiku>, codex/<model>, "
            "or openrouter/<provider>/<model>."
        )


def resolve_model_request(model: str) -> ModelResolution:
    """Resolve a namespaced model ID to a backend provider.

    Model IDs must begin with an AgentBridge provider namespace:
    claudecode/<model>, codex/<model>, or openrouter/<provider>/<model>.

    Args:
        model: Namespaced model identifier.

    Returns:
        Provider and backend model identifier.

    Raises:
        UnsupportedModelError: If model is not recognized
    """
    model_stripped = model.strip()

    provider, sep, provider_model = model_stripped.partition("/")
    if not sep:
        raise UnsupportedModelError(model)

    provider_lower = provider.lower()
    provider_model = provider_model.strip()
    provider_model_lower = provider_model.lower()

    if provider_lower not in PROVIDER_NAMES or not provider_model:
        raise UnsupportedModelError(model)

    if provider_lower == "codex":
        return ModelResolution(provider="codex", model=provider_model)

    if provider_lower == "openrouter":
        if "/" not in provider_model:
            raise UnsupportedModelError(model)
        return ModelResolution(provider="openrouter", model=provider_model)

    if provider_model_lower in CLAUDE_SIMPLE_NAMES:
        return ModelResolution(provider="claudecode", model=provider_model_lower)

    # Word-boundary match: find model name as a distinct segment in the slug
    match = _MODEL_PATTERN.search(provider_model_lower)
    if match:
        return ModelResolution(provider="claudecode", model=match.group(1).lower())

    # Unknown model - raise error
    raise UnsupportedModelError(model)


def resolve_model(model: str) -> str:
    """Resolve a model identifier to its backend model name.

    Kept for backwards compatibility. Use resolve_model_request() when the
    provider is needed.
    """
    resolution = resolve_model_request(model)
    return resolution.model
