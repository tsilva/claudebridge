"""FastAPI server exposing provider adapters as an OpenAI-compatible API."""

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .dashboard import DashboardState, create_dashboard_router
from .models import (
    AVAILABLE_MODELS,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ContentPart,
    DeltaMessage,
    ErrorDetail,
    ErrorResponse,
    FunctionCall,
    ImageUrlContent,
    Message,
    ModelInfo,
    ModelList,
    ReasoningEffort,
    StreamChoice,
    TextContent,
    Tool,
    ToolCall,
    UnsupportedModelError,
    Usage,
    resolve_model,
    resolve_model_request,
)
from .pool import ClientPool

# Claude palette (24-bit true color)
_CLAUDE = "\033[38;2;218;119;86m"  # Terracotta — Claude's signature orange
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


def extract_attachment_metadata(
    messages: list,
) -> list[dict]:
    """Extract attachment metadata from multimodal messages without decoding binary data.

    Returns lightweight dicts with msg_index, att_index, media_type, filename.
    Used for logging — avoids the memory/CPU cost of base64 decoding.
    """
    result = []
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg.content, list):
            continue
        att_idx = 0
        for part in msg.content:
            if not isinstance(part, ImageUrlContent):
                continue
            url = part.image_url.url
            if is_data_url(url):
                media_type, _ = parse_data_url(url)
                ext = EXTENSION_MAP.get(media_type, ".bin")
                filename = f"msg{msg_idx}_att{att_idx}{ext}"
                result.append({
                    "msg_index": msg_idx,
                    "att_index": att_idx,
                    "media_type": media_type,
                    "filename": filename,
                })
            elif is_http_url(url):
                ext = ".png"
                parsed_path = urllib.parse.urlparse(url).path.lower()
                for mt, e in EXTENSION_MAP.items():
                    if parsed_path.endswith(e):
                        ext = e
                        break
                filename = f"msg{msg_idx}_att{att_idx}{ext}"
                result.append({
                    "msg_index": msg_idx,
                    "att_index": att_idx,
                    "media_type": "image/unknown",
                    "filename": filename,
                })
            att_idx += 1
    return result


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
                parsed_path = urllib.parse.urlparse(url).path.lower()
                for mt, e in EXTENSION_MAP.items():
                    if parsed_path.endswith(e):
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
        truncated = url[:50] + ("..." if len(url) > 50 else "")
        raise ValueError(f"Invalid data URL format: {truncated}")
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

    def write(
        self,
        messages: list,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> None:
        """Write the complete session log as JSON."""
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - self.start_time).total_seconds() * 1000)
        full_response = "".join(content for _, content in self.chunks)

        # Format messages for JSON
        msg_list = []
        for msg in messages:
            content = "" if msg.content is None else extract_text_from_content(msg.content)
            msg_list.append({
                "role": msg.role,
                "content": content,
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

        # Build attachments metadata (no base64 decode needed for logging)
        att_meta = []
        try:
            att_meta = extract_attachment_metadata(messages)
        except Exception:
            pass

        data = {
            "request_id": self.request_id,
            "model": self.model,
            "api_key": (
                (self.api_key[:4] + "***")
                if self.api_key and len(self.api_key) > 4
                else ("***" if self.api_key else None)
            ),
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
            "exception_type": self.exception_type,
            "traceback": self.traceback_str,
            "pool_snapshot": self.pool_snapshot,
            "attachments": att_meta,
        }

        def _do_write() -> None:
            with open(self.log_path, "w") as f:
                json.dump(data, f, indent=2)
            self._save_attachments(messages)
            self._cleanup_old_logs()

        try:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, _do_write)

            def _on_write_done(fut: "asyncio.Future[None]") -> None:
                if not fut.cancelled() and fut.exception():
                    logging.error(
                        f"[session_logger] Failed to write log {self.log_path}: "
                        f"{fut.exception()}"
                    )

            future.add_done_callback(_on_write_done)
        except RuntimeError:
            # No running event loop (e.g. tests) — run synchronously
            _do_write()

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

            logging.info(
                f"[session_logger] Saved {len(attachments)} attachment(s) "
                f"for {self.request_id}"
            )
        except Exception as e:
            logging.warning(f"[session_logger] Failed to save attachments: {e}")

    def _cleanup_old_logs(self) -> None:
        """Delete oldest log files if count exceeds MAX_LOG_FILES."""
        try:
            def _mtime(f: Path) -> float:
                try:
                    return f.stat().st_mtime
                except OSError:
                    return 0.0

            log_files = sorted(
                self.log_dir.glob("*.json"),
                key=_mtime,
            )
            if len(log_files) > MAX_LOG_FILES:
                to_delete = log_files[:len(log_files) - MAX_LOG_FILES]
                for f in to_delete:
                    stem = f.stem
                    att_dir = self.log_dir / f"{stem}_attachments"
                    if att_dir.is_dir():
                        shutil.rmtree(att_dir, ignore_errors=True)
                    f.unlink(missing_ok=True)
                logging.info(f"[session_logger] Cleaned up {len(to_delete)} old log files")
        except Exception as e:
            logging.warning(f"[session_logger] Log cleanup failed: {e}")


# Runtime configuration
pool: ClientPool | None = None
_pool_lock: asyncio.Lock | None = None
_pool_size = 1
codex_semaphore: asyncio.Semaphore | None = None
dashboard_state = DashboardState()

# Track which unsupported parameter warnings have been shown (log once per param)
_warned_params: set[str] = set()

# Parameters accepted for compatibility but not supported by provider adapters
_UNSUPPORTED_PARAMS = {
    "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "stop", "n", "seed", "response_format", "logit_bias", "logprobs",
    "top_logprobs", "parallel_tool_calls", "stream_options", "user",
    "max_tokens",  # Accepted but not supported by Claude SDK client options here
}

CODEX_DEFAULT_REASONING_EFFORT_BY_MODEL: dict[str, ReasoningEffort] = {
    "gpt-5.5": "high",
}


def _warn_unsupported_params(
    request: "ChatCompletionRequest",
    provider: str,
) -> None:
    """Log a warning for unsupported parameters that have a non-None value."""
    if provider == "openrouter":
        return
    for param in _UNSUPPORTED_PARAMS:
        if param not in _warned_params:
            value = getattr(request, param, None)
            if value is not None:
                _warned_params.add(param)
                logging.warning(
                    f"Parameter '{param}' is accepted but not supported by AgentBridge "
                    "- value ignored"
                )


def _resolve_codex_reasoning_effort(
    request: ChatCompletionRequest,
    resolution: Any,
) -> ReasoningEffort | None:
    """Resolve Codex reasoning effort from OpenAI/OpenRouter-compatible fields."""
    if resolution.provider != "codex":
        return None

    effort = request.reasoning_effort
    if effort is None and request.reasoning:
        raw_effort = request.reasoning.get("effort")
        if raw_effort is not None:
            valid_efforts = {"minimal", "low", "medium", "high", "xhigh"}
            if raw_effort not in valid_efforts:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unsupported reasoning effort. Use one of: "
                        "minimal, low, medium, high, xhigh."
                    ),
                )
            effort = raw_effort

    model = (resolution.model or "").lower()
    return effort or CODEX_DEFAULT_REASONING_EFFORT_BY_MODEL.get(model)


async def ensure_claude_pool() -> ClientPool:
    """Create the Claude client pool on demand."""
    global pool, _pool_lock
    if pool is not None:
        return pool
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    async with _pool_lock:
        if pool is not None:
            return pool
        new_pool = ClientPool(
            size=_pool_size,
            default_model="opus",
            on_change=dashboard_state.notify_pool_change,
        )
        await new_pool.initialize()
        pool = new_pool
        return new_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and shutdown provider resources."""
    global pool, _pool_lock, _pool_size, codex_semaphore
    _pool_size = int(os.environ.get("POOL_SIZE", 1))
    _pool_lock = asyncio.Lock()
    codex_semaphore = asyncio.Semaphore(_pool_size)

    yield
    if pool is not None:
        await pool.shutdown()
        pool = None


app = FastAPI(title="AgentBridge", version=__version__, lifespan=lifespan)

app.include_router(
    create_dashboard_router(
        dashboard_state,
        pool_status_fn=lambda: pool.status()
        if pool
        else {"size": _pool_size, "available": 0, "in_use": 0, "models": []},
    )
)

# Timeout for provider calls (in seconds)
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", 120))
CODEX_TIMEOUT = int(
    os.environ.get("CODEX_TIMEOUT", os.environ.get("CLAUDE_TIMEOUT", 120))
)
OPENROUTER_TIMEOUT = int(
    os.environ.get("OPENROUTER_TIMEOUT", os.environ.get("CLAUDE_TIMEOUT", 120))
)
OPENROUTER_API_URL = os.environ.get(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)


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

    # Extract request_id from BridgeHTTPException for response header
    request_id = getattr(exc, "request_id", None)

    response = JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc.detail),
                type=error_type,
                code=error_type,
            )
        ).model_dump(),
    )
    if request_id:
        response.headers["X-Request-Id"] = request_id
    return response


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic validation errors to OpenAI error format."""
    first_error = exc.errors()[0] if exc.errors() else {}
    message = first_error.get("msg", "Invalid request")
    param = ".".join(str(p) for p in first_error.get("loc", [])) or None
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorDetail(
                message=f"Invalid request: {message}",
                type="invalid_request_error",
                param=param,
                code="invalid_request_error",
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
                message="Internal server error",
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
        elif msg.role == "tool":
            name = getattr(msg, "name", None) or "tool"
            parts.append(f"Tool ({name}): {content}")

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
        if msg.content is None:
            continue
        if msg.role == "system":
            system_prompt = extract_text_from_content(msg.content)
        elif msg.role == "tool":
            name = getattr(msg, "name", None) or "tool"
            text = extract_text_from_content(msg.content)
            content_blocks.append({"type": "text", "text": f"Tool ({name}): {text}"})
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
    """Build a prompt suffix that asks the model for JSON matching the tool schema.

    Provider adapters do not expose custom OpenAI-style function calling, so this
    emulates it by including the schema in the prompt and asking for JSON output.
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


async def send_query(
    client, prompt: str | list[dict], session_id: str = "default"
) -> None:
    """Send a query to a Claude client, handling multimodal vs string dispatch."""
    if isinstance(prompt, list):
        await client.query(make_multimodal_prompt(prompt), session_id=session_id)
    else:
        await client.query(prompt, session_id=session_id)


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


def _get_codex_semaphore() -> asyncio.Semaphore:
    """Return the process limit semaphore, creating it for tests without lifespan."""
    global codex_semaphore
    if codex_semaphore is None:
        codex_semaphore = asyncio.Semaphore(_pool_size)
    return codex_semaphore


def _codex_binary() -> str:
    """Return configured Codex executable path."""
    codex_bin = os.environ.get("CODEX_BIN") or shutil.which("codex")
    if not codex_bin:
        raise RuntimeError("Codex CLI not found. Install Codex or set CODEX_BIN.")
    return codex_bin


def _codex_text_from_content(content: Any) -> str:
    """Extract assistant text from flexible Codex JSON event content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            block_type = str(block.get("type", ""))
            if block_type in {"text", "output_text", "assistant_message", "agent_message"}:
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif "text" in block and isinstance(block["text"], str):
                parts.append(block["text"])
    return "".join(parts)


def _codex_container_is_assistant(container: dict[str, Any]) -> bool:
    """Return whether a JSON event/container looks like assistant output."""
    role = container.get("role")
    item_type = str(container.get("type", ""))
    return (
        role == "assistant"
        or "assistant" in item_type
        or item_type in {"agent_message", "output_text"}
    )


def _codex_event_error(event: dict[str, Any]) -> str | None:
    """Extract an error string from a Codex JSON event if present."""
    event_type = str(event.get("type", ""))
    if event_type in {"error", "turn.failed"} or event_type.endswith(".failed"):
        error = event.get("error") or event.get("message") or event.get("reason")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    error = event.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if isinstance(error, str):
        return error
    return None


def _codex_event_usage(event: dict[str, Any]) -> dict[str, int] | None:
    """Extract token usage from known Codex event shapes."""
    candidates = [event, event.get("usage"), event.get("turn"), event.get("response")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        usage = candidate.get("usage") if isinstance(candidate.get("usage"), dict) else candidate
        if not isinstance(usage, dict):
            continue
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if input_tokens is not None or output_tokens is not None:
            return {
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
            }
    return None


def _codex_event_text_delta(
    event: dict[str, Any],
    current_text: str,
) -> tuple[str, str]:
    """Return newly available assistant text and the updated full text."""
    containers = [event]
    for key in ("item", "message", "response", "delta"):
        value = event.get(key)
        if isinstance(value, dict):
            containers.append(value)

    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("text_delta", "content_delta", "output_text_delta", "delta"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value, current_text + value

    full_candidates: list[str] = []
    for container in containers:
        if not isinstance(container, dict) or not _codex_container_is_assistant(container):
            continue
        for key in ("text", "output_text", "content"):
            text = _codex_text_from_content(container.get(key))
            if text:
                full_candidates.append(text)

    if not full_candidates:
        return "", current_text

    full_text = max(full_candidates, key=len)
    if full_text.startswith(current_text):
        return full_text[len(current_text):], full_text
    if full_text and full_text not in current_text:
        return full_text, current_text + full_text
    return "", current_text


def _prepare_codex_prompt(
    prompt: str | list[dict],
    work_dir: Path,
) -> tuple[str, list[Path]]:
    """Convert formatted prompt blocks into Codex stdin text and image files."""
    if isinstance(prompt, str):
        return prompt, []

    text_parts: list[str] = []
    image_paths: list[Path] = []
    attachment_index = 0

    for block in prompt:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
            continue

        source = block.get("source") if isinstance(block.get("source"), dict) else {}
        media_type = source.get("media_type", "application/octet-stream")
        ext = EXTENSION_MAP.get(media_type, ".bin")
        filename = f"attachment_{attachment_index}{ext}"
        path = work_dir / filename
        attachment_index += 1

        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            path.write_bytes(base64.b64decode(source["data"]))
            if str(media_type).startswith("image/"):
                image_paths.append(path)
            else:
                text_parts.append(f"[Attached document saved at: {path}]")
        elif source.get("type") == "url" and source.get("url"):
            text_parts.append(f"[Image URL: {source['url']}]")

    return "\n\n".join(part for part in text_parts if part), image_paths


def _build_codex_command(
    backend_model: str,
    work_dir: Path,
    output_file: Path,
    image_paths: list[Path],
    reasoning_effort: ReasoningEffort | None = None,
) -> list[str]:
    """Build a non-interactive Codex CLI command."""
    cmd = [
        _codex_binary(),
        "-a",
        "never",
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-C",
        str(work_dir),
        "-o",
        str(output_file),
        "-m",
        backend_model,
    ]
    if reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    for image_path in image_paths:
        cmd.extend(["--image", str(image_path)])
    cmd.append("-")
    return cmd


def _parse_codex_json_lines(output: str) -> tuple[str, dict[str, int] | None]:
    """Parse Codex JSONL output into final assistant text and usage."""
    full_text = ""
    usage: dict[str, int] | None = None
    for line in output.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_usage = _codex_event_usage(event)
        if event_usage:
            usage = event_usage
        _, full_text = _codex_event_text_delta(event, full_text)
    return full_text, usage


async def call_codex_cli(
    prompt: str | list[dict],
    model: str,
    session_logger: SessionLogger,
    tools: list[Tool] | None = None,
    messages: list[dict] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> ClaudeResponse:
    """Call Codex CLI and return an OpenAI-compatible response container."""
    resolution = resolve_model_request(model)
    request_id = session_logger.request_id
    dashboard_state.request_started(
        request_id,
        model,
        api_key=session_logger.api_key,
        messages=messages,
    )
    effective_prompt = apply_tool_prompt(prompt, tools) if tools else prompt
    response = ClaudeResponse()
    semaphore = _get_codex_semaphore()
    acquire_start = time.monotonic()
    acquired = False

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=CODEX_TIMEOUT)
        acquired = True
        acquire_ms = int((time.monotonic() - acquire_start) * 1000)
        query_start = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="agentbridge-codex-") as tmp:
            work_dir = Path(tmp)
            output_file = work_dir / "last-message.txt"
            codex_prompt, image_paths = _prepare_codex_prompt(effective_prompt, work_dir)
            proc = await asyncio.create_subprocess_exec(
                *_build_codex_command(
                    resolution.model or model,
                    work_dir,
                    output_file,
                    image_paths,
                    reasoning_effort,
                ),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(codex_prompt.encode()),
                timeout=CODEX_TIMEOUT,
            )
            query_ms = int((time.monotonic() - query_start) * 1000)
            session_logger.log_timing(acquire_ms, query_ms)

            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            parsed_text, usage = _parse_codex_json_lines(stdout)
            if output_file.exists():
                parsed_text = output_file.read_text(errors="replace")

            if proc.returncode != 0:
                raise RuntimeError((stderr or stdout or "Codex CLI failed").strip())

            response.text = parsed_text
            if response.text:
                session_logger.log_chunk(response.text)
            if usage:
                response.usage = usage
                session_logger.log_usage(
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )

        if tools and response.text:
            remaining_text, tool_calls = parse_tool_response(response.text, tools)
            if tool_calls:
                response.text = remaining_text
                response.tool_calls = tool_calls
                session_logger.log_chunk(
                    f"[parsed tool_call: {tool_calls[0].function.name}]"
                )

        usage_dict = response.get_usage()
        logging.info(
            f"[{request_id}] Completed codex | acquire={session_logger.acquire_ms}ms "
            f"query={session_logger.query_ms}ms "
            f"tokens={usage_dict['prompt_tokens']}in/{usage_dict['completion_tokens']}out"
        )
        session_logger.log_finish(response.finish_reason)
        dashboard_state.request_completed(request_id, usage=usage_dict)
        return response
    except asyncio.TimeoutError:
        session_logger.log_error(
            f"Timeout after {CODEX_TIMEOUT}s",
            exception_type="TimeoutError",
        )
        dashboard_state.request_errored(request_id, f"Timeout after {CODEX_TIMEOUT}s")
        raise BridgeHTTPException(
            status_code=504,
            detail=(
                f"Codex CLI timed out after {CODEX_TIMEOUT}s. "
                "Increase CODEX_TIMEOUT env var for longer requests."
            ),
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"[{request_id}] Codex {type(e).__name__}: {e}")
        session_logger.log_error(
            str(e),
            exception_type=type(e).__name__,
            traceback_str=tb,
        )
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
        raise
    finally:
        if acquired:
            semaphore.release()


def _openrouter_api_key() -> str:
    """Return the configured OpenRouter API key."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for openrouter/<model> requests.")
    return api_key


def _openrouter_headers() -> dict[str, str]:
    """Build OpenRouter HTTP headers."""
    headers = {
        "Authorization": f"Bearer {_openrouter_api_key()}",
        "Content-Type": "application/json",
        "User-Agent": f"agentbridge/{__version__}",
    }
    referer = os.environ.get("OPENROUTER_SITE_URL")
    title = os.environ.get("OPENROUTER_APP_NAME", "agentbridge")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _openrouter_payload(
    request: ChatCompletionRequest,
    backend_model: str,
    *,
    stream: bool,
) -> dict[str, Any]:
    """Convert an AgentBridge request into an OpenRouter chat completions payload."""
    payload = request.model_dump(exclude_none=True)
    payload["model"] = backend_model
    payload["stream"] = stream
    return payload


def _openrouter_request(payload: dict[str, Any]) -> urllib.request.Request:
    """Create an OpenRouter urllib request."""
    return urllib.request.Request(
        OPENROUTER_API_URL,
        data=json.dumps(payload).encode(),
        headers=_openrouter_headers(),
        method="POST",
    )


def _openrouter_error_message(exc: urllib.error.HTTPError) -> str:
    """Read a useful message from an OpenRouter HTTP error."""
    body = exc.read().decode(errors="replace")
    if not body:
        return f"OpenRouter HTTP {exc.code}"
    try:
        data = json.loads(body)
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    except json.JSONDecodeError:
        pass
    return body.strip()


def _openrouter_post_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Blocking OpenRouter JSON request for use in asyncio.to_thread()."""
    try:
        with urllib.request.urlopen(
            _openrouter_request(payload),
            timeout=OPENROUTER_TIMEOUT,
        ) as response:
            return json.loads(response.read().decode(errors="replace"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_openrouter_error_message(exc)) from exc


def _message_from_openrouter(data: dict[str, Any]) -> Message:
    """Convert an OpenRouter assistant message into our response model."""
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    first = choices[0] if choices else {}
    raw_message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(raw_message, dict):
        raw_message = {}

    tool_calls = raw_message.get("tool_calls")
    parsed_tool_calls = None
    if isinstance(tool_calls, list):
        parsed_tool_calls = [ToolCall.model_validate(call) for call in tool_calls]

    content = raw_message.get("content")
    if content is not None and not isinstance(content, str):
        content = _codex_text_from_content(content)

    return Message(
        role="assistant",
        content=content,
        tool_calls=parsed_tool_calls,
    )


def _usage_from_openrouter(data: dict[str, Any]) -> dict[str, int] | None:
    """Extract OpenAI-format usage from OpenRouter response JSON."""
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(
        usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def call_openrouter_api(
    request: ChatCompletionRequest,
    backend_model: str,
    session_logger: SessionLogger,
    messages: list[dict] | None = None,
) -> ClaudeResponse:
    """Call OpenRouter's Chat Completions API and return a local response container."""
    request_id = session_logger.request_id
    dashboard_state.request_started(
        request_id,
        request.model,
        api_key=session_logger.api_key,
        messages=messages,
    )
    response = ClaudeResponse()
    query_start = time.monotonic()

    try:
        data = await asyncio.to_thread(
            _openrouter_post_json,
            _openrouter_payload(request, backend_model, stream=False),
        )
        query_ms = int((time.monotonic() - query_start) * 1000)
        session_logger.log_timing(0, query_ms)

        message = _message_from_openrouter(data)
        response.text = message.content or ""
        response.tool_calls = message.tool_calls or []
        if response.text:
            session_logger.log_chunk(response.text)
        usage = _usage_from_openrouter(data)
        if usage:
            response.usage = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
            }
            session_logger.log_usage(
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )

        logging.info(f"[{request_id}] Completed openrouter | query={query_ms}ms")
        session_logger.log_finish(response.finish_reason)
        dashboard_state.request_completed(request_id, usage=response.get_usage())
        return response
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"[{request_id}] OpenRouter {type(e).__name__}: {e}")
        session_logger.log_error(
            str(e),
            exception_type=type(e).__name__,
            traceback_str=tb,
        )
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
        raise


async def _openrouter_stream_lines(payload: dict[str, Any]):
    """Yield blocking OpenRouter SSE lines without blocking the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = object()

    def _worker() -> None:
        try:
            with urllib.request.urlopen(
                _openrouter_request(payload),
                timeout=OPENROUTER_TIMEOUT,
            ) as response:
                for raw_line in response:
                    line = raw_line.decode(errors="replace").strip()
                    if not line:
                        continue
                    asyncio.run_coroutine_threadsafe(queue.put(line), loop).result()
        except urllib.error.HTTPError as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(RuntimeError(_openrouter_error_message(exc))),
                loop,
            ).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


def _openrouter_delta_text(chunk: dict[str, Any]) -> str:
    """Extract text deltas from an OpenRouter stream chunk."""
    choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            text = delta.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


async def stream_openrouter_api(
    request: ChatCompletionRequest,
    backend_model: str,
    request_id: str,
    session_logger: SessionLogger,
    messages: list[dict] | None = None,
):
    """Stream OpenRouter SSE as OpenAI-compatible chunks."""
    dashboard_state.request_started(
        request_id,
        request.model,
        api_key=session_logger.api_key,
        messages=messages,
    )
    created = int(time.time())
    finish_reason = "stop"
    stream_usage: Usage | None = None
    _dashboard_handled = False
    query_start = time.monotonic()

    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=request.model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    try:
        payload = _openrouter_payload(request, backend_model, stream=True)
        async for line in _openrouter_stream_lines(payload):
            if line.startswith("data:"):
                line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                break
            if not line:
                continue
            chunk = json.loads(line)
            if not isinstance(chunk, dict):
                continue
            chunk["model"] = request.model
            chunk.setdefault("id", request_id)
            chunk.setdefault("created", created)
            chunk.setdefault("object", "chat.completion.chunk")

            usage = _usage_from_openrouter(chunk)
            if usage:
                stream_usage = Usage(**usage)
                session_logger.log_usage(
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                )

            text = _openrouter_delta_text(chunk)
            if text:
                session_logger.log_chunk(text)
                dashboard_state.chunk_received(request_id, text)

            for choice in chunk.get("choices", []):
                if isinstance(choice, dict) and choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

            yield f"data: {json.dumps(chunk)}\n\n"

        query_ms = int((time.monotonic() - query_start) * 1000)
        session_logger.log_timing(0, query_ms)
        usage_dict = stream_usage.model_dump() if stream_usage else {}
        logging.info(f"[{request_id}] Completed openrouter | query={query_ms}ms")
        session_logger.log_finish(finish_reason)
        dashboard_state.request_completed(request_id, usage=usage_dict)
        _dashboard_handled = True
        yield "data: [DONE]\n\n"
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"[{request_id}] OpenRouter {type(e).__name__}: {e}")
        session_logger.log_error(
            str(e),
            exception_type=type(e).__name__,
            traceback_str=tb,
        )
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
        _dashboard_handled = True
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                StreamChoice(
                    delta=DeltaMessage(content="\n\n[Error: OpenRouter request failed.]"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if not _dashboard_handled:
            dashboard_state.request_errored(request_id, "Request cancelled")


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
    are resolved to Claude Code model identifiers. The pool lazily creates and
    reuses clients by model, evicting idle clients only when the max size is full.

    Note: Function calling is emulated by prompting for JSON output since the
    Claude Agent SDK doesn't support custom tool definitions.
    """
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    resolved_model = resolve_model(model)
    request_id = session_logger.request_id
    dashboard_state.request_started(
        request_id,
        model,
        api_key=session_logger.api_key,
        messages=messages,
    )

    # Add tool prompt if tools are provided
    effective_prompt = apply_tool_prompt(prompt, tools) if tools else prompt

    # Capture timing in outer scope so timeout handler can record partial data
    _acquire_ms: int | None = None
    _query_start: float | None = None

    async def _query():
        nonlocal _acquire_ms, _query_start
        response = ClaudeResponse()
        acquire_start = time.monotonic()
        claude_pool = await ensure_claude_pool()
        async with claude_pool.acquire(resolved_model, request_id=request_id) as client:
            _acquire_ms = int((time.monotonic() - acquire_start) * 1000)
            _query_start = time.monotonic()
            await send_query(client, effective_prompt, session_id=request_id)
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
            query_ms = int((time.monotonic() - _query_start) * 1000)
            session_logger.log_timing(_acquire_ms, query_ms)
        return response

    try:
        response = await asyncio.wait_for(_query(), timeout=CLAUDE_TIMEOUT)

        # If tools were provided, try to parse tool calls from the response
        if tools and response.text:
            remaining_text, tool_calls = parse_tool_response(response.text, tools)
            if tool_calls:
                response.text = remaining_text
                response.tool_calls = tool_calls
                session_logger.log_chunk(
                    f"[parsed tool_call: {tool_calls[0].function.name}]"
                )

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
        snap = pool.snapshot() if pool is not None else {}
        logging.error(f"[{request_id}] Timeout after {CLAUDE_TIMEOUT}s | pool={snap}")
        # Record partial timing if acquire completed before timeout fired
        if (
            _acquire_ms is not None
            and _query_start is not None
            and session_logger.acquire_ms is None
        ):
            session_logger.log_timing(
                _acquire_ms,
                int((time.monotonic() - _query_start) * 1000),
            )
        session_logger.log_error(
            f"Timeout after {CLAUDE_TIMEOUT}s",
            exception_type="TimeoutError",
            pool_snapshot=snap,
        )
        dashboard_state.request_errored(request_id, f"Timeout after {CLAUDE_TIMEOUT}s")
        raise BridgeHTTPException(
            status_code=504,
            detail=(
                f"Claude SDK timed out after {CLAUDE_TIMEOUT}s. "
                "Increase CLAUDE_TIMEOUT env var for longer requests."
            ),
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        snap = pool.snapshot() if pool is not None else {}
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
    except BaseException:
        # Handles asyncio.CancelledError and other non-Exception BaseExceptions
        session_logger.log_error("Request cancelled", exception_type="CancelledError")
        dashboard_state.request_errored(request_id, "Request cancelled")
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
    are resolved to Claude Code model identifiers. The pool lazily creates and
    reuses clients by model, evicting idle clients only when the max size is full.

    Note: When tools are provided, we buffer the response to parse JSON at the end
    since we're emulating function calling through prompting.
    """
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    resolved_model = resolve_model(model)
    dashboard_state.request_started(
        request_id,
        model,
        api_key=session_logger.api_key,
        messages=messages,
    )
    created = int(time.time())
    finish_reason = "stop"
    _dashboard_handled = False  # True once request_completed or request_errored has been called

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
    # Pre-declare timing vars so timeout handler can record partial data
    acquire_ms: int | None = None
    query_start: float | None = None

    try:
        acquire_start = time.monotonic()
        claude_pool = await ensure_claude_pool()
        async with claude_pool.acquire(resolved_model, request_id=request_id) as client:
            acquire_ms = int((time.monotonic() - acquire_start) * 1000)
            query_start = time.monotonic()
            async with asyncio.timeout(CLAUDE_TIMEOUT):
                await send_query(client, effective_prompt, session_id=request_id)
                async for msg in client.receive_response():
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
                                        choices=[
                                            StreamChoice(
                                                delta=DeltaMessage(content=block.text)
                                            )
                                        ],
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
                        choices=[
                            StreamChoice(delta=DeltaMessage(tool_calls=[tool_call]))
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                session_logger.log_chunk(
                    f"[parsed tool_call: {tool_calls[0].function.name}]"
                )
            else:
                # No tool calls found, stream the buffered text in chunks
                _CHUNK_SIZE = 100
                for i in range(0, len(full_text), _CHUNK_SIZE):
                    text_chunk = full_text[i : i + _CHUNK_SIZE]
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[StreamChoice(delta=DeltaMessage(content=text_chunk))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

        total_ms = (session_logger.acquire_ms or 0) + (session_logger.query_ms or 0)
        usage_dict = stream_usage.model_dump() if stream_usage else {}
        logging.info(
            f"[{request_id}] Completed | acquire={session_logger.acquire_ms}ms "
            f"query={session_logger.query_ms}ms total={total_ms}ms "
            f"tokens={usage_dict.get('prompt_tokens', 0)}in/"
            f"{usage_dict.get('completion_tokens', 0)}out"
        )
        session_logger.log_finish(finish_reason)
        dashboard_state.request_completed(request_id, usage=usage_dict)
        _dashboard_handled = True

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
    except asyncio.TimeoutError:
        snap = pool.snapshot() if pool is not None else {}
        logging.error(f"[{request_id}] Timeout after {CLAUDE_TIMEOUT}s | pool={snap}")
        # Record partial timing if acquire completed before timeout fired
        if (
            session_logger.acquire_ms is None
            and acquire_ms is not None
            and query_start is not None
        ):
            session_logger.log_timing(
                acquire_ms,
                int((time.monotonic() - query_start) * 1000),
            )
        session_logger.log_error(
            f"Timeout after {CLAUDE_TIMEOUT}s",
            exception_type="TimeoutError",
            pool_snapshot=snap,
        )
        dashboard_state.request_errored(request_id, f"Timeout after {CLAUDE_TIMEOUT}s")
        _dashboard_handled = True
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    delta=DeltaMessage(
                        content=(
                            "\n\n[Error: Request timed out. Increase CLAUDE_TIMEOUT "
                            "env var for longer requests.]"
                        )
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return
    except HTTPException as he:
        logging.error(f"[{request_id}] HTTPException {he.status_code}: {he.detail}")
        session_logger.log_error(
            str(he.detail),
            exception_type="HTTPException",
        )
        dashboard_state.request_errored(request_id, f"HTTP {he.status_code}: {he.detail}")
        _dashboard_handled = True
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(content=f"\n\n[Error: {he.detail}]"),
                finish_reason=None,
            )],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        snap = pool.snapshot() if pool is not None else {}
        tb = traceback.format_exc()
        logging.error(f"[{request_id}] {type(e).__name__}: {e} | pool={snap}")
        session_logger.log_error(
            str(e),
            exception_type=type(e).__name__,
            traceback_str=tb,
            pool_snapshot=snap,
        )
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
        _dashboard_handled = True
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(content="\n\n[Error: An internal error occurred.]"),
                finish_reason=None,
            )],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return
    finally:
        # Handles GeneratorExit/CancelledError — only fires if not already handled
        if not _dashboard_handled:
            dashboard_state.request_errored(request_id, "Request cancelled")


async def stream_codex_cli(
    prompt: str | list[dict],
    model: str,
    request_id: str,
    session_logger: SessionLogger,
    tools: list[Tool] | None = None,
    messages: list[dict] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
):
    """Stream Codex CLI output as OpenAI-compatible SSE chunks."""
    resolution = resolve_model_request(model)
    dashboard_state.request_started(
        request_id,
        model,
        api_key=session_logger.api_key,
        messages=messages,
    )
    created = int(time.time())
    finish_reason = "stop"
    _dashboard_handled = False
    effective_prompt = apply_tool_prompt(prompt, tools) if tools else prompt
    semaphore = _get_codex_semaphore()
    acquired = False
    proc: asyncio.subprocess.Process | None = None
    stderr_task: asyncio.Task[bytes] | None = None
    full_text = ""
    stream_usage: Usage | None = None
    acquire_ms: int | None = None
    query_start: float | None = None

    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    try:
        acquire_start = time.monotonic()
        await asyncio.wait_for(semaphore.acquire(), timeout=CODEX_TIMEOUT)
        acquired = True
        acquire_ms = int((time.monotonic() - acquire_start) * 1000)
        query_start = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="agentbridge-codex-") as tmp:
            work_dir = Path(tmp)
            output_file = work_dir / "last-message.txt"
            codex_prompt, image_paths = _prepare_codex_prompt(effective_prompt, work_dir)
            proc = await asyncio.create_subprocess_exec(
                *_build_codex_command(
                    resolution.model or model,
                    work_dir,
                    output_file,
                    image_paths,
                    reasoning_effort,
                ),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stderr_task = asyncio.create_task(proc.stderr.read())
            assert proc.stdin is not None
            proc.stdin.write(codex_prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()

            async with asyncio.timeout(CODEX_TIMEOUT):
                assert proc.stdout is not None
                async for raw_line in proc.stdout:
                    line = raw_line.decode(errors="replace").strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_error = _codex_event_error(event)
                    if event_error:
                        raise RuntimeError(event_error)

                    event_usage = _codex_event_usage(event)
                    if event_usage:
                        prompt_tokens = event_usage.get("input_tokens", 0)
                        completion_tokens = event_usage.get("output_tokens", 0)
                        stream_usage = Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        session_logger.log_usage(prompt_tokens, completion_tokens)

                    delta, full_text = _codex_event_text_delta(event, full_text)
                    if not delta:
                        continue
                    session_logger.log_chunk(delta)
                    dashboard_state.chunk_received(request_id, delta)
                    if not tools:
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created,
                            model=model,
                            choices=[StreamChoice(delta=DeltaMessage(content=delta))],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                await proc.wait()

            stderr = ""
            if stderr_task is not None:
                stderr = (await stderr_task).decode(errors="replace")
            if proc.returncode != 0:
                raise RuntimeError((stderr or "Codex CLI failed").strip())

            if output_file.exists():
                final_text = output_file.read_text(errors="replace")
                if final_text.startswith(full_text):
                    final_delta = final_text[len(full_text):]
                elif final_text != full_text:
                    final_delta = final_text
                else:
                    final_delta = ""
                if final_delta:
                    session_logger.log_chunk(final_delta)
                    dashboard_state.chunk_received(request_id, final_delta)
                    if not tools:
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created,
                            model=model,
                            choices=[StreamChoice(delta=DeltaMessage(content=final_delta))],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                full_text = final_text or full_text

        query_ms = int((time.monotonic() - query_start) * 1000)
        session_logger.log_timing(acquire_ms, query_ms)

        if tools and full_text:
            remaining_text, tool_calls = parse_tool_response(full_text, tools)
            if tool_calls:
                finish_reason = "tool_calls"
                for tool_call in tool_calls:
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            StreamChoice(delta=DeltaMessage(tool_calls=[tool_call]))
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                session_logger.log_chunk(
                    f"[parsed tool_call: {tool_calls[0].function.name}]"
                )
            else:
                _CHUNK_SIZE = 100
                for i in range(0, len(full_text), _CHUNK_SIZE):
                    text_chunk = full_text[i : i + _CHUNK_SIZE]
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[StreamChoice(delta=DeltaMessage(content=text_chunk))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

        usage_dict = stream_usage.model_dump() if stream_usage else {}
        logging.info(
            f"[{request_id}] Completed codex | acquire={session_logger.acquire_ms}ms "
            f"query={session_logger.query_ms}ms "
            f"tokens={usage_dict.get('prompt_tokens', 0)}in/"
            f"{usage_dict.get('completion_tokens', 0)}out"
        )
        session_logger.log_finish(finish_reason)
        dashboard_state.request_completed(request_id, usage=usage_dict)
        _dashboard_handled = True

        final_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
            usage=stream_usage,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except asyncio.TimeoutError:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        if (
            session_logger.acquire_ms is None
            and acquire_ms is not None
            and query_start is not None
        ):
            session_logger.log_timing(
                acquire_ms,
                int((time.monotonic() - query_start) * 1000),
            )
        session_logger.log_error(
            f"Timeout after {CODEX_TIMEOUT}s",
            exception_type="TimeoutError",
        )
        dashboard_state.request_errored(request_id, f"Timeout after {CODEX_TIMEOUT}s")
        _dashboard_handled = True
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    delta=DeltaMessage(
                        content=(
                            "\n\n[Error: Request timed out. Increase CODEX_TIMEOUT "
                            "env var for longer requests.]"
                        )
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        tb = traceback.format_exc()
        logging.error(f"[{request_id}] Codex {type(e).__name__}: {e}")
        session_logger.log_error(
            str(e),
            exception_type=type(e).__name__,
            traceback_str=tb,
        )
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
        _dashboard_handled = True
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[StreamChoice(
                delta=DeltaMessage(content="\n\n[Error: An internal error occurred.]"),
                finish_reason=None,
            )],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return
    finally:
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
        if acquired:
            semaphore.release()
        if not _dashboard_handled:
            dashboard_state.request_errored(request_id, "Request cancelled")


@app.post("/api/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """OpenAI-compatible chat completions endpoint.

    Note: Concurrency is managed by the client pool (POOL_SIZE env var).
    Model selection requires provider namespaces:
    claudecode/<model>, codex/<model>, or openrouter/<provider>/<model>.
    Model parameter is required.

    Tool calling is supported via the `tools` and `tool_choice` parameters.
    When the model decides to use tools, the response will include `tool_calls`
    with `finish_reason="tool_calls"`.
    """
    # Validate model early to fail fast (UnsupportedModelError handled by exception handler)
    model_resolution = resolve_model_request(request.model)
    codex_reasoning_effort = _resolve_codex_reasoning_effort(request, model_resolution)

    # Warn about unsupported params (once per param)
    _warn_unsupported_params(request, model_resolution.provider)

    request_id = f"chatcmpl-{uuid4().hex[:12]}"
    auth = http_request.headers.get("authorization", "")
    api_key = (auth.removeprefix("Bearer ").strip() or None) if auth else None
    prompt = format_messages(request.messages)
    session_logger = SessionLogger(request_id, request.model, api_key=api_key)
    # Serialize messages for dashboard display
    dash_messages = [
        {
            "role": m.role,
            "content": m.content if isinstance(m.content, str) else str(m.content),
        }
        for m in request.messages
    ]

    if request.stream:

        async def stream_with_logging():
            try:
                if model_resolution.provider == "codex":
                    async for chunk in stream_codex_cli(
                        prompt,
                        request.model,
                        request_id,
                        session_logger,
                        request.tools,
                        messages=dash_messages,
                        reasoning_effort=codex_reasoning_effort,
                    ):
                        yield chunk
                elif model_resolution.provider == "claudecode":
                    async for chunk in stream_claude_sdk(
                        prompt,
                        request.model,
                        request_id,
                        session_logger,
                        request.tools,
                        messages=dash_messages,
                    ):
                        yield chunk
                else:
                    async for chunk in stream_openrouter_api(
                        request,
                        model_resolution.model,
                        request_id,
                        session_logger,
                        messages=dash_messages,
                    ):
                        yield chunk
            finally:
                session_logger.write(
                    request.messages,
                    request.stream,
                    request.temperature,
                    request.max_tokens,
                )

        return StreamingResponse(
            stream_with_logging(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-Id": request_id,
            },
        )

    try:
        if model_resolution.provider == "claudecode":
            response = await call_claude_sdk(
                prompt,
                request.model,
                session_logger,
                request.tools,
                messages=dash_messages,
            )
        elif model_resolution.provider == "codex":
            response = await call_codex_cli(
                prompt,
                request.model,
                session_logger,
                request.tools,
                messages=dash_messages,
                reasoning_effort=codex_reasoning_effort,
            )
        else:
            response = await call_openrouter_api(
                request,
                model_resolution.model,
                session_logger,
                messages=dash_messages,
            )
    finally:
        session_logger.write(
            request.messages,
            request.stream,
            request.temperature,
            request.max_tokens,
        )

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
        data=[
            ModelInfo(id=m["slug"], owned_by=m.get("owned_by", "agentbridge"))
            for m in AVAILABLE_MODELS
        ]
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


def _print_banner(port: int, workers: int, timeout: int) -> None:
    """Print clean startup banner with ASCII art bridge and colors."""
    version = get_version()
    print(f"\n  {_CLAUDE}   ╭───╮       ╭───╮{_RESET}")
    print(f"  {_CLAUDE}═══╯   ╰═══════╯   ╰═══{_RESET}")
    print(f"  {_CLAUDE_DIM}   │   │       │   │{_RESET}")
    print(f"  {_BOLD}{_CLAUDE}agentbridge{_RESET} {_DIM}v{version}{_RESET}\n")
    print(f"  {_DIM}Dashboard{_RESET}  {_CLAUDE}http://127.0.0.1:{port}/dashboard{_RESET}")
    print(f"  {_DIM}API{_RESET}        {_CLAUDE}http://127.0.0.1:{port}/api/v1{_RESET}")
    print(f"  {_DIM}Workers{_RESET}    {_BOLD}{workers}{_RESET}")
    print(f"  {_DIM}Timeout{_RESET}    {timeout}s")
    print()


def main():
    """Entry point for CLI."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="AgentBridge - OpenAI-compatible API for Claude and Codex"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"agentbridge {get_version()}",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of pooled clients (default: 1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8082)),
        help="Server port (default: 8082)",
    )
    args = parser.parse_args()

    timeout = int(os.environ.get("CLAUDE_TIMEOUT", 120))

    # Set worker count for lifespan initialization
    os.environ["POOL_SIZE"] = str(args.workers)

    _configure_logging()
    _print_banner(args.port, args.workers, timeout)

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
