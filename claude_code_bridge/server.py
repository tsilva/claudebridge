"""FastAPI server exposing Claude Code SDK as OpenAI-compatible API."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
from fastapi.responses import StreamingResponse
from claude_agent_sdk import ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock

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
from .model_mapping import resolve_model, AVAILABLE_MODELS, UnsupportedModelError
from .pool import ClientPool
from .session_logger import SessionLogger

# Pool configuration
POOL_SIZE = int(os.environ.get("POOL_SIZE", 3))
pool: ClientPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and shutdown pool."""
    global pool
    options = ClaudeAgentOptions(
        max_turns=1,
        setting_sources=["user"],
        system_prompt={"type": "preset", "preset": "claude_code"},
    )
    pool = ClientPool(size=POOL_SIZE, options=options)
    await pool.initialize()
    yield
    await pool.shutdown()


app = FastAPI(title="Claude Code Bridge", version="0.1.0", lifespan=lifespan)

# Timeout for Claude SDK calls (in seconds)
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", 120))


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


async def call_claude_sdk(prompt: str, model: str, logger: SessionLogger) -> str:
    """Call Claude Code SDK using pooled client and return response text.

    Model selection: OpenRouter-style slugs or simple names (opus/sonnet/haiku)
    are resolved to Claude Code model identifiers.
    """
    resolved_model = resolve_model(model)

    async def _query():
        response_text = ""
        async with pool.acquire() as client:
            # Set model if specified
            if resolved_model:
                await client.query(f"/model {resolved_model}")
                async for msg in client.receive_response():
                    if isinstance(msg, ResultMessage):
                        break

            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                            logger.log_chunk(block.text)
                elif isinstance(msg, ResultMessage):
                    break
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


async def stream_claude_sdk(prompt: str, model: str, request_id: str, logger: SessionLogger):
    """Stream Claude Code SDK response as SSE chunks using pooled client.

    Model selection: OpenRouter-style slugs or simple names (opus/sonnet/haiku)
    are resolved to Claude Code model identifiers.
    """
    resolved_model = resolve_model(model)
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
        async with pool.acquire() as client:
            # Set model if specified
            if resolved_model:
                await client.query(f"/model {resolved_model}")
                async for msg in client.receive_response():
                    if isinstance(msg, ResultMessage):
                        break

            await client.query(prompt)
            async for msg in client.receive_response():
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
                elif isinstance(msg, ResultMessage):
                    break
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
    """OpenAI-compatible chat completions endpoint.

    Note: Concurrency is managed by the client pool (POOL_SIZE env var).
    Model selection supports OpenRouter-style slugs (e.g., anthropic/claude-sonnet-4)
    or simple names (opus, sonnet, haiku). Model parameter is required.
    """
    # Validate model early to fail fast
    try:
        resolve_model(request.model)
    except UnsupportedModelError as e:
        raise HTTPException(status_code=400, detail=str(e))

    request_id = f"chatcmpl-{uuid4().hex[:12]}"
    prompt = format_messages(request.messages)
    logger = SessionLogger(request_id, request.model)

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
    """List available models with OpenRouter-style slugs."""
    return ModelList(
        data=[ModelInfo(id=m["slug"]) for m in AVAILABLE_MODELS]
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
