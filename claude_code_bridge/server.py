"""FastAPI server exposing Claude Code SDK as OpenAI-compatible API."""

import asyncio
import os
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

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
)
from .session_logger import SessionLogger

app = FastAPI(title="Claude Code Bridge", version="0.1.0")

# Limit concurrent Claude SDK calls
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 3))
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Timeout for Claude SDK calls (in seconds)
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", 120))

# Model name mapping
MODEL_MAP = {
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
    "claude-opus": "opus",
    "claude-sonnet": "sonnet",
    "claude-haiku": "haiku",
    "claude-3-opus": "opus",
    "claude-3-sonnet": "sonnet",
    "claude-3-haiku": "haiku",
    "claude-3.5-sonnet": "sonnet",
    "claude-3.5-haiku": "haiku",
}

AVAILABLE_MODELS = ["opus", "sonnet", "haiku"]


def map_model(model: str | None) -> str | None:
    """Map incoming model name to Claude Code SDK model, or None for local default."""
    if not model:
        return None  # Use local Claude Code settings
    model_lower = model.lower()
    if model_lower in MODEL_MAP:
        return MODEL_MAP[model_lower]
    return None  # Unknown model → use local settings


def format_messages(messages: list[Message]) -> str:
    """Convert OpenAI-style messages to a single prompt string."""
    parts = []
    system_prompt = None

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")

    prompt = "\n\n".join(parts)
    if system_prompt:
        prompt = f"System: {system_prompt}\n\n{prompt}"

    return prompt


async def call_claude_sdk(prompt: str, model: str | None, logger: SessionLogger) -> str:
    """Call Claude Code SDK and return response text."""
    mapped_model = map_model(model)
    options = ClaudeAgentOptions(
        max_turns=1,
        setting_sources=["user"],  # Load user settings (including default model)
        system_prompt={"type": "preset", "preset": "claude_code"},  # Use default Claude Code system prompt
        **({"model": mapped_model} if mapped_model else {}),
    )

    async def _query():
        response_text = ""
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
                        logger.log_chunk(block.text)
        return response_text

    try:
        response_text = await asyncio.wait_for(_query(), timeout=CLAUDE_TIMEOUT)
        logger.log_finish("stop")
    except asyncio.TimeoutError:
        logger.log_error(f"Timeout after {CLAUDE_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"Claude SDK timed out after {CLAUDE_TIMEOUT}s")
    except Exception as e:
        logger.log_error(str(e))
        raise

    return response_text


async def stream_claude_sdk(prompt: str, model: str | None, request_id: str, logger: SessionLogger):
    """Stream Claude Code SDK response as SSE chunks."""
    mapped_model = map_model(model)
    options = ClaudeAgentOptions(
        max_turns=1,
        setting_sources=["user"],  # Load user settings (including default model)
        system_prompt={"type": "preset", "preset": "claude_code"},  # Use default Claude Code system prompt
        **({"model": mapped_model} if mapped_model else {}),
    )

    created = int(time.time())
    start_time = time.monotonic()

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    try:
        async for msg in query(prompt=prompt, options=options):
            # Check timeout
            if time.monotonic() - start_time > CLAUDE_TIMEOUT:
                logger.log_error(f"Timeout after {CLAUDE_TIMEOUT}s")
                raise asyncio.TimeoutError(f"Claude SDK timed out after {CLAUDE_TIMEOUT}s")
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        logger.log_chunk(block.text)
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created,
                            model=model,
                            choices=[StreamChoice(delta=DeltaMessage(content=block.text))],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
        logger.log_finish("stop")
    except asyncio.TimeoutError:
        logger.log_error(f"Timeout after {CLAUDE_TIMEOUT}s")
        raise
    except Exception as e:
        logger.log_error(str(e))
        raise

    # Send final chunk
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    request_id = f"chatcmpl-{uuid4().hex[:12]}"
    prompt = format_messages(request.messages)
    logger = SessionLogger(request_id, request.model)

    async with semaphore:
        if request.stream:
            async def stream_with_logging():
                try:
                    async for chunk in stream_claude_sdk(prompt, request.model, request_id, logger):
                        yield chunk
                finally:
                    logger.write(request.messages, request.stream, request.temperature, request.max_tokens)

            return StreamingResponse(
                stream_with_logging(),
                media_type="text/event-stream",
            )

        response_text = await call_claude_sdk(prompt, request.model, logger)
        logger.write(request.messages, request.stream, request.temperature, request.max_tokens)

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    message=Message(role="assistant", content=response_text),
                )
            ],
        )


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return ModelList(
        data=[ModelInfo(id=model) for model in AVAILABLE_MODELS]
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    """Entry point for CLI."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
