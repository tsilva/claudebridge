"""OpenAI-compatible request/response models."""

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
    # Additional fields ignored but accepted for compatibility
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None


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
