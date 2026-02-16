"""FastAPI server exposing Claude Code SDK as OpenAI-compatible API."""

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Claude palette (24-bit true color)
_CLAUDE = "\033[38;2;218;119;86m"   # Terracotta — Claude's signature orange
_CLAUDE_DIM = "\033[38;2;171;93;67m"  # Muted terracotta for secondary accents
_DIM = "\033[2m"
_BOLD = "\033[1m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_RESET = "\033[0m"


class _BridgeFormatter(logging.Formatter):
    """Colored log output: dim timestamps, yellow warnings, red errors."""

    def format(self, record: logging.LogRecord) -> str:
        ts = f"{_DIM}{self.formatTime(record, '%H:%M:%S')}{_RESET}"
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"{ts} {_RED}{record.levelname}{_RESET} {msg}"
        if record.levelno >= logging.WARNING:
            return f"{ts} {_YELLOW}{record.levelname}{_RESET} {msg}"
        return f"{ts} {msg}"


class _SuppressSDKNoise(logging.Filter):
    """Drop noisy SDK messages like 'Using bundled Claude Code CLI: ...'."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Using bundled Claude Code CLI" in msg:
            return False
        return True


def _configure_logging() -> None:
    """Set up bridge-style logging: clean timestamps, suppressed SDK noise."""
    handler = logging.StreamHandler()
    handler.setFormatter(_BridgeFormatter())
    handler.addFilter(_SuppressSDKNoise())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Suppress verbose SDK internals
    logging.getLogger("claude_agent_sdk").setLevel(logging.WARNING)

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    StreamChoice,
    Message,
    DeltaMessage,
    ModelList,
    ModelInfo,
    Tool,
    ToolCall,
    FunctionCall,
    Usage,
    ErrorDetail,
    ErrorResponse,
)
from .models import (
    resolve_model,
    AVAILABLE_MODELS,
    UnsupportedModelError,
    ContentPart,
    TextContent,
    ImageUrlContent,
)
from .pool import ClientPool
from .dashboard import DashboardState
from . import __version__


# ---------------------------------------------------------------------------
# Image format conversion utilities (OpenAI → Claude)
# ---------------------------------------------------------------------------

EXTENSION_MAP = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


@dataclass
class AttachmentInfo:
    """Metadata for an attachment extracted from a message."""

    msg_index: int
    att_index: int
    media_type: str
    content_type: str  # "base64" or "url"
    data: bytes | None  # decoded binary data (base64 only)
    url: str | None  # original HTTP URL (url only)
    filename: str  # e.g. "msg0_att0.png"


def extract_attachments_from_messages(
    messages: list,
) -> list[AttachmentInfo]:
    """Extract attachment info from multimodal messages."""
    attachments: list[AttachmentInfo] = []

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg.content, list):
            continue

        att_idx = 0
        for part in msg.content:
            if not isinstance(part, ImageUrlContent):
                continue

            url = part.image_url.url

            if is_data_url(url):
                media_type, b64_data = parse_data_url(url)
                ext = EXTENSION_MAP.get(media_type, ".bin")
                filename = f"msg{msg_idx}_att{att_idx}{ext}"
                attachments.append(
                    AttachmentInfo(
                        msg_index=msg_idx,
                        att_index=att_idx,
                        media_type=media_type,
                        content_type="base64",
                        data=base64.b64decode(b64_data),
                        url=None,
                        filename=filename,
                    )
                )
            elif is_http_url(url):
                ext = ".png"
                for mt, e in EXTENSION_MAP.items():
                    if e[1:] in url.lower():
                        ext = e
                        break
                filename = f"msg{msg_idx}_att{att_idx}{ext}"
                attachments.append(
                    AttachmentInfo(
                        msg_index=msg_idx,
                        att_index=att_idx,
                        media_type="image/unknown",
                        content_type="url",
                        data=None,
                        url=url,
                        filename=filename,
                    )
                )

            att_idx += 1

    return attachments


def parse_data_url(url: str) -> tuple[str, str]:
    """Extract media type and base64 data from a data URL."""
    match = re.match(r"data:([^;]+);base64,(.+)", url)
    if not match:
        raise ValueError(f"Invalid data URL format: {url[:50]}...")
    return match.group(1), match.group(2)


def is_http_url(url: str) -> bool:
    """Check if URL is an HTTP/HTTPS URL."""
    return url.startswith("http://") or url.startswith("https://")


def is_data_url(url: str) -> bool:
    """Check if URL is a data URL."""
    return url.startswith("data:")


def openai_image_to_claude(image_content: ImageUrlContent) -> dict[str, Any]:
    """Convert OpenAI image_url content block to Claude image/document format."""
    url = image_content.image_url.url

    if is_data_url(url):
        media_type, data = parse_data_url(url)
        block_type = "document" if media_type == "application/pdf" else "image"
        return {
            "type": block_type,
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }
        }

    if is_http_url(url):
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            }
        }

    raise ValueError(f"Unsupported image URL format: {url[:50]}...")


def openai_content_to_claude(content: str | list[ContentPart]) -> list[dict[str, Any]]:
    """Convert OpenAI message content to Claude content array format."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    result = []
    for part in content:
        if isinstance(part, TextContent):
            result.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageUrlContent):
            result.append(openai_image_to_claude(part))

    return result


def has_multimodal_content(messages: list) -> bool:
    """Check if any message contains image content."""
    return any(
        isinstance(part, ImageUrlContent)
        for msg in messages
        if isinstance(msg.content, list)
        for part in msg.content
    )


def extract_text_from_content(content: str | list[ContentPart]) -> str:
    """Extract text from message content for logging."""
    if isinstance(content, str):
        return content

    parts = []
    for part in content:
        if isinstance(part, TextContent):
            parts.append(part.text)
        elif isinstance(part, ImageUrlContent):
            url = part.image_url.url
            if is_data_url(url):
                media_type, _ = parse_data_url(url)
                if media_type == "application/pdf":
                    parts.append("[document: PDF base64 data]")
                else:
                    parts.append("[image: base64 data]")
            else:
                parts.append(f"[image: {url}]")

    return " ".join(parts)

# ---------------------------------------------------------------------------
# Session logging (JSON format)
# ---------------------------------------------------------------------------

# Maximum number of log files to keep
MAX_LOG_FILES = int(os.environ.get("MAX_LOG_FILES", 1000))


class SessionLogger:
    """Logs a single Claude request/response session to a JSON file."""

    def __init__(self, request_id: str, model: str, api_key: str | None = None):
        self.request_id = request_id
        self.model = model
        self.api_key = api_key
        self.start_time = datetime.now(timezone.utc)
        self.chunks: list[tuple[datetime, str]] = []
        self.finish_reason: str | None = None
        self.error: str | None = None
        self.acquire_ms: int | None = None
        self.query_ms: int | None = None
        self.pool_snapshot: dict | None = None
        self.exception_type: str | None = None
        self.traceback_str: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

        # Ensure log directory exists
        self.log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{request_id}.json"

    def log_chunk(self, content: str) -> None:
        """Record a streaming chunk with timestamp."""
        self.chunks.append((datetime.now(timezone.utc), content))

    def log_finish(self, reason: str) -> None:
        """Record the finish reason."""
        self.finish_reason = reason

    def log_timing(self, acquire_ms: int, query_ms: int) -> None:
        """Record timing breakdown."""
        self.acquire_ms = acquire_ms
        self.query_ms = query_ms

    def log_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def log_error(self, error: str, *, exception_type: str | None = None,
                  traceback_str: str | None = None, pool_snapshot: dict | None = None) -> None:
        """Record an error with optional diagnostic details."""
        self.error = error
        if exception_type is not None:
            self.exception_type = exception_type
        if traceback_str is not None:
            self.traceback_str = traceback_str
        if pool_snapshot is not None:
            self.pool_snapshot = pool_snapshot

    def write(self, messages: list, stream: bool, temperature: float | None, max_tokens: int | None) -> None:
        """Write the complete session log as JSON."""
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - self.start_time).total_seconds() * 1000)
        full_response = "".join(content for _, content in self.chunks)

        # Format messages for JSON
        msg_list = []
        for msg in messages:
            msg_list.append({
                "role": msg.role,
                "content": extract_text_from_content(msg.content),
            })

        # Build timing dict
        timing: dict[str, int] = {"duration_ms": duration_ms}
        if self.acquire_ms is not None:
            timing["acquire_ms"] = self.acquire_ms
        if self.query_ms is not None:
            timing["query_ms"] = self.query_ms

        # Build usage dict
        usage: dict[str, int] = {}
        if self.input_tokens is not None:
            usage["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            usage["output_tokens"] = self.output_tokens

        # Build attachments metadata
        att_meta = []
        try:
            attachments = extract_attachments_from_messages(messages)
            for att in attachments:
                entry = {
                    "msg_index": att.msg_index,
                    "att_index": att.att_index,
                    "media_type": att.media_type,
                    "filename": att.filename,
                }
                att_meta.append(entry)
        except Exception:
            pass

        data = {
            "request_id": self.request_id,
            "model": self.model,
            "api_key": self.api_key,
            "timestamp": self.start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "messages": msg_list,
            "parameters": {
                "stream": stream,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "response": full_response,
            "finish_reason": self.finish_reason,
            "timing": timing,
            "usage": usage,
            "error": self.error,
            "attachments": att_meta,
        }

        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save binary attachments
        self._save_attachments(messages)
        self._cleanup_old_logs()

    def _save_attachments(self, messages: list) -> None:
        """Save binary attachments alongside the log file."""
        try:
            attachments = extract_attachments_from_messages(messages)
            if not attachments:
                return

            att_dir = self.log_dir / f"{self.request_id}_attachments"
            att_dir.mkdir(parents=True, exist_ok=True)

            for att in attachments:
                if att.content_type == "base64" and att.data:
                    (att_dir / att.filename).write_bytes(att.data)

            logging.info(f"[session_logger] Saved {len(attachments)} attachment(s) for {self.request_id}")
        except Exception as e:
            logging.warning(f"[session_logger] Failed to save attachments: {e}")

    def _cleanup_old_logs(self) -> None:
        """Delete oldest log files if count exceeds MAX_LOG_FILES."""
        try:
            log_files = sorted(
                self.log_dir.glob("*.json"),
                key=lambda f: f.stat().st_mtime,
            )
            if len(log_files) > MAX_LOG_FILES:
                to_delete = log_files[:len(log_files) - MAX_LOG_FILES]
                for f in to_delete:
                    stem = f.stem
                    att_dir = self.log_dir / f"{stem}_attachments"
                    if att_dir.is_dir():
                        shutil.rmtree(att_dir)
                    f.unlink()
                logging.info(f"[session_logger] Cleaned up {len(to_delete)} old log files")
        except Exception as e:
            logging.warning(f"[session_logger] Log cleanup failed: {e}")


# Pool configuration
pool: ClientPool | None = None
dashboard_state = DashboardState()

# Track which unsupported parameter warnings have been shown (log once per param)
_warned_params: set[str] = set()

# Parameters accepted for compatibility but not supported by Claude SDK
_UNSUPPORTED_PARAMS = {
    "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "stop", "n", "seed", "response_format", "logit_bias", "logprobs",
    "top_logprobs", "parallel_tool_calls", "stream_options", "user",
}


def _warn_unsupported_params(request: "ChatCompletionRequest") -> None:
    """Log a warning for each unsupported parameter that has a non-None value, once per param."""
    for param in _UNSUPPORTED_PARAMS:
        if param not in _warned_params:
            value = getattr(request, param, None)
            if value is not None:
                _warned_params.add(param)
                logging.warning(
                    f"Parameter '{param}' is accepted but not supported by Claude SDK — value ignored"
                )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and shutdown pool."""
    global pool
    pool_size = int(os.environ.get("POOL_SIZE", 1))
    timeout = int(os.environ.get("CLAUDE_TIMEOUT", 120))
    port = int(os.environ.get("PORT", 8082))

    pool = ClientPool(size=pool_size, default_model="opus", on_change=dashboard_state.notify_pool_change)
    try:
        await pool.initialize()
    except Exception as e:
        logging.error(f"Failed to initialize pool: {e}")
        raise

    yield
    await pool.shutdown()


app = FastAPI(title="Claude Code Bridge", version=__version__, lifespan=lifespan)

# Mount dashboard
from .dashboard import create_dashboard_router

app.include_router(create_dashboard_router(
    dashboard_state,
    pool_status_fn=lambda: pool.status() if pool else {"size": 0, "available": 0, "in_use": 0, "models": []},
))

# Timeout for Claude SDK calls (in seconds)
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", 120))


class BridgeHTTPException(HTTPException):
    """HTTPException with request_id for error tracing."""

    def __init__(self, status_code: int, detail: str, request_id: str | None = None):
        super().__init__(status_code=status_code, detail=detail)
        self.request_id = request_id


# Exception handlers for OpenAI-format error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to OpenAI error format."""
    # Map status codes to error types
    error_types = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "server_error",
        504: "timeout_error",
    }
    error_type = error_types.get(exc.status_code, "server_error")

    # Extract request_id from BridgeHTTPException
    code = getattr(exc, "request_id", None)

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc.detail),
                type=error_type,
                code=code,
            )
        ).model_dump(),
    )


@app.exception_handler(UnsupportedModelError)
async def unsupported_model_handler(request: Request, exc: UnsupportedModelError):
    """Handle unsupported model errors with OpenAI error format."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc),
                type="invalid_request_error",
                param="model",
                code="model_not_found",
            )
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with OpenAI error format."""
    logging.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                message=f"Internal error: {type(exc).__name__}: {str(exc)}",
                type="server_error",
            )
        ).model_dump(),
    )


def format_messages(messages: list[Message]) -> str | list[dict]:
    """Convert OpenAI-style messages to Claude format.

    Returns:
        str: For text-only messages, returns formatted prompt string
        list[dict]: For multimodal messages, returns Claude-style content array
    """
    # Check if we have multimodal content
    if has_multimodal_content(messages):
        return format_multimodal_messages(messages)

    # Text-only path: return formatted string
    parts = []
    system_prompt = None

    for msg in messages:
        content = "" if msg.content is None else extract_text_from_content(msg.content)

        if msg.role == "system":
            system_prompt = content
        elif msg.role == "user":
            parts.append(f"User: {content}")
        elif msg.role == "assistant":
            # Skip empty assistant messages (e.g., tool call only)
            if content:
                parts.append(f"Assistant: {content}")

    prompt = "\n\n".join(parts)
    if system_prompt:
        prompt = f"System: {system_prompt}\n\n{prompt}"

    return prompt


def format_multimodal_messages(messages: list[Message]) -> list[dict]:
    """Format messages with image content for Claude SDK.

    Builds a flat content array combining text and images from all messages.
    Prefixes user/assistant messages with role labels for context.
    """
    content_blocks = []
    system_prompt = None

    for msg in messages:
        if msg.role == "system":
            system_prompt = extract_text_from_content(msg.content)
        else:
            # Build content for user/assistant messages
            role_prefix = "User" if msg.role == "user" else "Assistant"
            claude_content = openai_content_to_claude(msg.content)

            # Add role prefix to first text block, or prepend new text block
            if claude_content and claude_content[0].get("type") == "text":
                claude_content[0]["text"] = f"{role_prefix}: {claude_content[0]['text']}"
            else:
                content_blocks.append({"type": "text", "text": f"{role_prefix}:"})

            content_blocks.extend(claude_content)

    # Prepend system prompt if present
    if system_prompt:
        content_blocks.insert(0, {"type": "text", "text": f"System: {system_prompt}"})

    return content_blocks


def make_multimodal_prompt(content_blocks: list[dict]):
    """Create an async generator that yields a multimodal user message.

    The Claude SDK's query() method accepts either a string or an async iterable.
    For multimodal content (images), we need to pass an async iterable that yields
    a properly formatted message with content blocks.
    """
    async def _gen():
        yield {
            "type": "user",
            "message": {"role": "user", "content": content_blocks},
        }
    return _gen()


def build_tool_prompt(tools: list[Tool]) -> str:
    """Build a prompt suffix that instructs Claude to respond with JSON matching the tool schema.

    Since the Claude Agent SDK doesn't support custom function calling, we emulate it
    by including the schema in the prompt and asking for JSON output.
    """
    if len(tools) == 1:
        schema = tools[0].function.parameters or {}
        return (
            f"\n\n---\n"
            f"IMPORTANT: You must respond with a JSON object that matches this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Respond ONLY with the JSON object, no other text before or after."
        )

    # Multiple tools - let model choose
    tool_schemas = [
        {
            "name": t.function.name,
            "description": t.function.description,
            "parameters": t.function.parameters,
        }
        for t in tools
    ]
    return (
        f"\n\n---\n"
        f"IMPORTANT: You must call one of these functions by responding with JSON:\n"
        f"```json\n{json.dumps(tool_schemas, indent=2)}\n```\n"
        f"Respond with: {{\"function\": \"<function_name>\", \"arguments\": {{...}}}}\n"
        f"Respond ONLY with the JSON object, no other text before or after."
    )


def parse_tool_response(text: str, tools: list[Tool]) -> tuple[str, list[ToolCall]]:
    """Parse text response to extract tool calls.

    Returns:
        Tuple of (remaining_text, tool_calls)
    """
    # Try to find JSON in the response
    # Look for JSON block or raw JSON
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        r'(\{[\s\S]*\})',                # Raw JSON object
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1).strip()
                data = json.loads(json_str)

                # For multiple tools with explicit function/arguments format,
                # extract the named function call
                if len(tools) > 1 and "function" in data and "arguments" in data:
                    name = data["function"]
                    arguments = json.dumps(data["arguments"])
                else:
                    # Single tool, or multi-tool fallback to first tool
                    name = tools[0].function.name
                    arguments = json.dumps(data)

                tool_call = ToolCall(
                    id=f"call_{uuid4().hex[:12]}",
                    function=FunctionCall(name=name, arguments=arguments),
                )
                return "", [tool_call]
            except json.JSONDecodeError:
                continue

    # No valid JSON found, return text as-is
    return text, []


def apply_tool_prompt(prompt: str | list[dict], tools: list[Tool]) -> str | list[dict]:
    """Append tool prompt suffix to a string or multimodal prompt."""
    tool_suffix = build_tool_prompt(tools)
    if isinstance(prompt, str):
        return prompt + tool_suffix
    # For multimodal, append to the last text block or add new one
    result = prompt.copy()
    if result and result[-1].get("type") == "text":
        result[-1] = {"type": "text", "text": result[-1]["text"] + tool_suffix}
    else:
        result.append({"type": "text", "text": tool_suffix})
    return result


async def send_query(client, prompt: str | list[dict]) -> None:
    """Send a query to a Claude client, handling multimodal vs string dispatch."""
    if isinstance(prompt, list):
        await client.query(make_multimodal_prompt(prompt))
    else:
        await client.query(prompt)


class ClaudeResponse:
    """Container for Claude SDK response with text and/or tool calls."""
    def __init__(self):
        self.text: str = ""
        self.tool_calls: list[ToolCall] = []
        self.usage: dict[str, int] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self.has_tool_calls else "stop"

    def get_usage(self) -> dict[str, int]:
        """Return usage dict with OpenAI-format keys."""
        if not self.usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt = self.usage.get("input_tokens", 0)
        completion = self.usage.get("output_tokens", 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }


async def call_claude_sdk(
    prompt: str | list[dict],
    model: str,
    session_logger: SessionLogger,
    tools: list[Tool] | None = None,
    messages: list[dict] | None = None,
) -> ClaudeResponse:
    """Call Claude Code SDK using pooled client and return response.

    Args:
        prompt: Either a string (text-only) or list of content blocks (multimodal)
        model: Model identifier (OpenRouter slug or simple name)
        session_logger: Session logger for recording the interaction
        tools: Optional list of tool definitions for function calling

    Returns:
        ClaudeResponse containing text and/or tool calls

    Model selection: OpenRouter-style slugs or simple names (opus/sonnet/haiku)
    are resolved to Claude Code model identifiers. Pool replaces clients
    on-demand when a different model is requested.

    Note: Function calling is emulated by prompting for JSON output since the
    Claude Agent SDK doesn't support custom tool definitions.
    """
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    resolved_model = resolve_model(model)
    request_id = session_logger.request_id
    dashboard_state.request_started(request_id, model, api_key=session_logger.api_key, messages=messages)

    # Add tool prompt if tools are provided
    effective_prompt = apply_tool_prompt(prompt, tools) if tools else prompt

    async def _query():
        response = ClaudeResponse()
        acquire_start = time.monotonic()
        async with pool.acquire(resolved_model, request_id=request_id) as client:
            acquire_ms = int((time.monotonic() - acquire_start) * 1000)
            query_start = time.monotonic()
            await send_query(client, effective_prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            response.text += block.text
                            session_logger.log_chunk(block.text)
                elif isinstance(msg, ResultMessage):
                    # Capture usage data from result
                    if msg.usage:
                        response.usage = msg.usage
                        session_logger.log_usage(
                            msg.usage.get("input_tokens", 0),
                            msg.usage.get("output_tokens", 0),
                        )
                    break
            query_ms = int((time.monotonic() - query_start) * 1000)
            session_logger.log_timing(acquire_ms, query_ms)
        return response

    try:
        response = await asyncio.wait_for(_query(), timeout=CLAUDE_TIMEOUT)

        # If tools were provided, try to parse tool calls from the response
        if tools and response.text:
            remaining_text, tool_calls = parse_tool_response(response.text, tools)
            if tool_calls:
                response.text = remaining_text
                response.tool_calls = tool_calls
                session_logger.log_chunk(f"[parsed tool_call: {tool_calls[0].function.name}]")

        total_ms = (session_logger.acquire_ms or 0) + (session_logger.query_ms or 0)
        usage = response.get_usage()
        logging.info(
            f"[{request_id}] Completed | acquire={session_logger.acquire_ms}ms "
            f"query={session_logger.query_ms}ms total={total_ms}ms "
            f"tokens={usage['prompt_tokens']}in/{usage['completion_tokens']}out"
        )
        session_logger.log_finish(response.finish_reason)
        dashboard_state.request_completed(request_id, usage=usage)
    except asyncio.TimeoutError:
        snap = pool.snapshot()
        logging.error(f"[{request_id}] Timeout after {CLAUDE_TIMEOUT}s | pool={snap}")
        session_logger.log_error(
            f"Timeout after {CLAUDE_TIMEOUT}s",
            exception_type="TimeoutError",
            pool_snapshot=snap,
        )
        dashboard_state.request_errored(request_id, f"Timeout after {CLAUDE_TIMEOUT}s")
        raise BridgeHTTPException(
            status_code=504,
            detail=f"Claude SDK timed out after {CLAUDE_TIMEOUT}s. Increase CLAUDE_TIMEOUT env var for longer requests.",
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        snap = pool.snapshot()
        tb = traceback.format_exc()
        logging.error(f"[{request_id}] {type(e).__name__}: {e} | pool={snap}")
        session_logger.log_error(
            str(e),
            exception_type=type(e).__name__,
            traceback_str=tb,
            pool_snapshot=snap,
        )
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
        raise

    return response


async def stream_claude_sdk(
    prompt: str | list[dict],
    model: str,
    request_id: str,
    session_logger: SessionLogger,
    tools: list[Tool] | None = None,
    messages: list[dict] | None = None,
):
    """Stream Claude Code SDK response as SSE chunks using pooled client.

    Args:
        prompt: Either a string (text-only) or list of content blocks (multimodal)
        model: Model identifier (OpenRouter slug or simple name)
        request_id: Unique request identifier for response chunks
        session_logger: Session logger for recording the interaction
        tools: Optional list of tool definitions for function calling

    Model selection: OpenRouter-style slugs or simple names (opus/sonnet/haiku)
    are resolved to Claude Code model identifiers. Pool replaces clients
    on-demand when a different model is requested.

    Note: When tools are provided, we buffer the response to parse JSON at the end
    since we're emulating function calling through prompting.
    """
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    resolved_model = resolve_model(model)
    dashboard_state.request_started(request_id, model, api_key=session_logger.api_key, messages=messages)
    created = int(time.time())
    start_time = time.monotonic()
    finish_reason = "stop"

    # Add tool prompt if tools are provided
    effective_prompt = apply_tool_prompt(prompt, tools) if tools else prompt

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Buffer for tool response parsing
    full_text = ""
    stream_usage: Usage | None = None

    try:
        acquire_start = time.monotonic()
        async with pool.acquire(resolved_model, request_id=request_id) as client:
            acquire_ms = int((time.monotonic() - acquire_start) * 1000)
            query_start = time.monotonic()
            await send_query(client, effective_prompt)
            async for msg in client.receive_response():
                # Check timeout
                if time.monotonic() - start_time > CLAUDE_TIMEOUT:
                    session_logger.log_error(f"Timeout after {CLAUDE_TIMEOUT}s")
                    raise asyncio.TimeoutError(f"Claude SDK timed out after {CLAUDE_TIMEOUT}s")
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            session_logger.log_chunk(block.text)
                            dashboard_state.chunk_received(request_id, block.text)
                            full_text += block.text

                            # If no tools, stream directly; otherwise buffer
                            if not tools:
                                chunk = ChatCompletionChunk(
                                    id=request_id,
                                    created=created,
                                    model=model,
                                    choices=[StreamChoice(delta=DeltaMessage(content=block.text))],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                elif isinstance(msg, ResultMessage):
                    # Capture usage data from result
                    if msg.usage:
                        prompt_tokens = msg.usage.get("input_tokens", 0)
                        completion_tokens = msg.usage.get("output_tokens", 0)
                        stream_usage = Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        session_logger.log_usage(prompt_tokens, completion_tokens)
                    break
            query_ms = int((time.monotonic() - query_start) * 1000)
            session_logger.log_timing(acquire_ms, query_ms)

        # If tools were provided, parse the buffered response
        if tools and full_text:
            remaining_text, tool_calls = parse_tool_response(full_text, tools)
            if tool_calls:
                finish_reason = "tool_calls"
                # Send tool call chunk
                for tool_call in tool_calls:
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[StreamChoice(delta=DeltaMessage(tool_calls=[tool_call]))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                session_logger.log_chunk(f"[parsed tool_call: {tool_calls[0].function.name}]")
            else:
                # No tool calls found, send the text
                chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[StreamChoice(delta=DeltaMessage(content=full_text))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        total_ms = (session_logger.acquire_ms or 0) + (session_logger.query_ms or 0)
        usage_dict = stream_usage.model_dump() if stream_usage else {}
        logging.info(
            f"[{request_id}] Completed | acquire={session_logger.acquire_ms}ms "
            f"query={session_logger.query_ms}ms total={total_ms}ms "
            f"tokens={usage_dict.get('prompt_tokens', 0)}in/{usage_dict.get('completion_tokens', 0)}out"
        )
        session_logger.log_finish(finish_reason)
        dashboard_state.request_completed(request_id, usage=usage_dict)
    except asyncio.TimeoutError:
        snap = pool.snapshot()
        logging.error(f"[{request_id}] Timeout after {CLAUDE_TIMEOUT}s | pool={snap}")
        session_logger.log_error(
            f"Timeout after {CLAUDE_TIMEOUT}s",
            exception_type="TimeoutError",
            pool_snapshot=snap,
        )
        dashboard_state.request_errored(request_id, f"Timeout after {CLAUDE_TIMEOUT}s")
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(content=f"\n\n[Error: Claude SDK timed out after {CLAUDE_TIMEOUT}s. Increase CLAUDE_TIMEOUT env var for longer requests.]"),
                finish_reason="error",
            )],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        snap = pool.snapshot()
        tb = traceback.format_exc()
        logging.error(f"[{request_id}] {type(e).__name__}: {e} | pool={snap}")
        session_logger.log_error(
            str(e),
            exception_type=type(e).__name__,
            traceback_str=tb,
            pool_snapshot=snap,
        )
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(content=f"\n\n[Error: {type(e).__name__}: {str(e)}]"),
                finish_reason="error",
            )],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Send final chunk with usage data
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
        usage=stream_usage,
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """OpenAI-compatible chat completions endpoint.

    Note: Concurrency is managed by the client pool (POOL_SIZE env var).
    Model selection supports OpenRouter-style slugs (e.g., anthropic/claude-sonnet-4)
    or simple names (opus, sonnet, haiku). Model parameter is required.

    Tool calling is supported via the `tools` and `tool_choice` parameters.
    When the model decides to use tools, the response will include `tool_calls`
    with `finish_reason="tool_calls"`.
    """
    # Validate model early to fail fast (UnsupportedModelError handled by exception handler)
    resolve_model(request.model)

    # Warn about unsupported params (once per param)
    _warn_unsupported_params(request)

    request_id = f"chatcmpl-{uuid4().hex[:12]}"
    auth = http_request.headers.get("authorization", "")
    api_key = auth.removeprefix("Bearer ").strip() or None if auth else None
    prompt = format_messages(request.messages)
    session_logger = SessionLogger(request_id, request.model, api_key=api_key)
    # Serialize messages for dashboard display
    dash_messages = [{"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)} for m in request.messages]

    if request.stream:
        async def stream_with_logging():
            try:
                async for chunk in stream_claude_sdk(
                    prompt, request.model, request_id, session_logger, request.tools, messages=dash_messages
                ):
                    yield chunk
            finally:
                session_logger.write(request.messages, request.stream, request.temperature, request.max_tokens)

        return StreamingResponse(
            stream_with_logging(),
            media_type="text/event-stream",
        )

    try:
        response = await call_claude_sdk(prompt, request.model, session_logger, request.tools, messages=dash_messages)
    finally:
        session_logger.write(request.messages, request.stream, request.temperature, request.max_tokens)

    response_message = Message(
        role="assistant",
        content=response.text or None,
        tool_calls=response.tool_calls if response.has_tool_calls else None,
    )

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                message=response_message,
                finish_reason=response.finish_reason,
            )
        ],
        usage=Usage(**response.get_usage()),
    )


@app.get("/api/v1/models")
async def list_models():
    """List available models with OpenRouter-style slugs."""
    return ModelList(
        data=[ModelInfo(id=m["slug"]) for m in AVAILABLE_MODELS]
    )


@app.get("/health")
async def health():
    """Health check endpoint with pool status."""
    result = {"status": "ok", "version": __version__}
    if pool is not None:
        result["pool"] = pool.status()
    return result


def get_version() -> str:
    """Get version string with git hash."""
    try:
        from ._build_info import GIT_HASH
    except ImportError:
        GIT_HASH = "dev"
    return f"{__version__} ({GIT_HASH})"


def _print_banner(port: int, workers: int, model: str, timeout: int) -> None:
    """Print clean startup banner with ASCII art bridge and colors."""
    version = get_version()
    print(f"\n  {_CLAUDE}   ╭───╮       ╭───╮{_RESET}")
    print(f"  {_CLAUDE}═══╯   ╰═══════╯   ╰═══{_RESET}")
    print(f"  {_CLAUDE_DIM}   │   │       │   │{_RESET}")
    print(f"  {_BOLD}{_CLAUDE}claudebridge{_RESET} {_DIM}v{version}{_RESET}\n")
    print(f"  {_DIM}Dashboard{_RESET}  {_CLAUDE}http://127.0.0.1:{port}/dashboard{_RESET}")
    print(f"  {_DIM}API{_RESET}        {_CLAUDE}http://127.0.0.1:{port}/api/v1{_RESET}")
    print(f"  {_DIM}Workers{_RESET}    {_BOLD}{workers}{_RESET} {_DIM}({model}){_RESET}")
    print(f"  {_DIM}Timeout{_RESET}    {timeout}s")
    print()


def main():
    """Entry point for CLI."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Claude Code Bridge - OpenAI-compatible API for Claude")
    parser.add_argument("-v", "--version", action="version", version=f"claudebridge {get_version()}")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of pooled clients (default: 1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8082)), help="Server port (default: 8082)")
    args = parser.parse_args()

    timeout = int(os.environ.get("CLAUDE_TIMEOUT", 120))

    # Set worker count for lifespan initialization
    os.environ["POOL_SIZE"] = str(args.workers)

    _configure_logging()
    _print_banner(args.port, args.workers, "opus", timeout)

    # Suppress uvicorn's default INFO noise
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "loggers": {
                "uvicorn": {"level": "WARNING"},
                "uvicorn.error": {"level": "WARNING"},
                "uvicorn.access": {"level": "WARNING"},
            },
        },
    )


if __name__ == "__main__":
    main()
