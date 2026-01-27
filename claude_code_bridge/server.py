"""FastAPI server exposing Claude Code SDK as OpenAI-compatible API."""

import asyncio
import json
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
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

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
)
from .model_mapping import resolve_model, AVAILABLE_MODELS, UnsupportedModelError
from .pool import ClientPool
from .session_logger import SessionLogger
from .image_utils import has_multimodal_content, openai_content_to_claude

# Pool configuration
POOL_SIZE = int(os.environ.get("POOL_SIZE", 3))
pool: ClientPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and shutdown pool."""
    global pool
    pool = ClientPool(size=POOL_SIZE, default_model="opus")
    await pool.initialize()
    yield
    await pool.shutdown()


app = FastAPI(title="Claude Code Bridge", version="0.1.0", lifespan=lifespan)

# Timeout for Claude SDK calls (in seconds)
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", 120))


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
        # Handle None content (e.g., in tool call messages)
        if msg.content is None:
            content = ""
        elif isinstance(msg.content, str):
            content = msg.content
        else:
            content = extract_text_content(msg.content)

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


def extract_text_content(content_parts: list) -> str:
    """Extract text from content parts list."""
    texts = []
    for part in content_parts:
        if hasattr(part, "type") and part.type == "text":
            texts.append(part.text)
    return " ".join(texts)


def format_multimodal_messages(messages: list[Message]) -> list[dict]:
    """Format messages with image content for Claude SDK.

    Builds a flat content array combining text and images from all messages.
    Prefixes user/assistant messages with role labels for context.
    """
    content_blocks = []
    system_prompt = None

    for msg in messages:
        if msg.role == "system":
            # Extract system prompt text
            if isinstance(msg.content, str):
                system_prompt = msg.content
            else:
                system_prompt = extract_text_content(msg.content)
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
        tool = tools[0]
        schema = tool.function.parameters or {}
        return (
            f"\n\n---\n"
            f"IMPORTANT: You must respond with a JSON object that matches this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Respond ONLY with the JSON object, no other text before or after."
        )
    else:
        # Multiple tools - let model choose
        tool_schemas = []
        for tool in tools:
            tool_schemas.append({
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            })
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
    import re
    from uuid import uuid4

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

                # Determine which tool was called
                if len(tools) == 1:
                    # Single tool - the JSON is the arguments
                    tool_call = ToolCall(
                        id=f"call_{uuid4().hex[:12]}",
                        function=FunctionCall(
                            name=tools[0].function.name,
                            arguments=json.dumps(data),
                        ),
                    )
                    return "", [tool_call]
                else:
                    # Multiple tools - look for function name in response
                    if "function" in data and "arguments" in data:
                        tool_call = ToolCall(
                            id=f"call_{uuid4().hex[:12]}",
                            function=FunctionCall(
                                name=data["function"],
                                arguments=json.dumps(data["arguments"]),
                            ),
                        )
                        return "", [tool_call]
                    else:
                        # Assume first tool
                        tool_call = ToolCall(
                            id=f"call_{uuid4().hex[:12]}",
                            function=FunctionCall(
                                name=tools[0].function.name,
                                arguments=json.dumps(data),
                            ),
                        )
                        return "", [tool_call]
            except json.JSONDecodeError:
                continue

    # No valid JSON found, return text as-is
    return text, []


class ClaudeResponse:
    """Container for Claude SDK response with text and/or tool calls."""
    def __init__(self):
        self.text: str = ""
        self.tool_calls: list[ToolCall] = []

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self.has_tool_calls else "stop"


async def call_claude_sdk(
    prompt: str | list[dict],
    model: str,
    logger: SessionLogger,
    tools: list[Tool] | None = None,
) -> ClaudeResponse:
    """Call Claude Code SDK using pooled client and return response.

    Args:
        prompt: Either a string (text-only) or list of content blocks (multimodal)
        model: Model identifier (OpenRouter slug or simple name)
        logger: Session logger for recording the interaction
        tools: Optional list of tool definitions for function calling

    Returns:
        ClaudeResponse containing text and/or tool calls

    Model selection: OpenRouter-style slugs or simple names (opus/sonnet/haiku)
    are resolved to Claude Code model identifiers. Pool replaces clients
    on-demand when a different model is requested.

    Note: Function calling is emulated by prompting for JSON output since the
    Claude Agent SDK doesn't support custom tool definitions.
    """
    resolved_model = resolve_model(model)

    # Add tool prompt if tools are provided
    effective_prompt = prompt
    if tools:
        tool_suffix = build_tool_prompt(tools)
        if isinstance(prompt, str):
            effective_prompt = prompt + tool_suffix
        else:
            # For multimodal, append to the last text block or add new one
            effective_prompt = prompt.copy()
            if effective_prompt and effective_prompt[-1].get("type") == "text":
                effective_prompt[-1] = {
                    "type": "text",
                    "text": effective_prompt[-1]["text"] + tool_suffix,
                }
            else:
                effective_prompt.append({"type": "text", "text": tool_suffix})

    async def _query():
        response = ClaudeResponse()
        async with pool.acquire(resolved_model) as client:
            # For multimodal content, create an async generator
            if isinstance(effective_prompt, list):
                await client.query(make_multimodal_prompt(effective_prompt))
            else:
                await client.query(effective_prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            response.text += block.text
                            logger.log_chunk(block.text)
                elif isinstance(msg, ResultMessage):
                    break
        return response

    try:
        response = await asyncio.wait_for(_query(), timeout=CLAUDE_TIMEOUT)

        # If tools were provided, try to parse tool calls from the response
        if tools and response.text:
            remaining_text, tool_calls = parse_tool_response(response.text, tools)
            if tool_calls:
                response.text = remaining_text
                response.tool_calls = tool_calls
                logger.log_chunk(f"[parsed tool_call: {tool_calls[0].function.name}]")

        logger.log_finish(response.finish_reason)
    except asyncio.TimeoutError:
        logger.log_error(f"Timeout after {CLAUDE_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"Claude SDK timed out after {CLAUDE_TIMEOUT}s")
    except Exception as e:
        logger.log_error(str(e))
        raise

    return response


async def stream_claude_sdk(
    prompt: str | list[dict],
    model: str,
    request_id: str,
    logger: SessionLogger,
    tools: list[Tool] | None = None,
):
    """Stream Claude Code SDK response as SSE chunks using pooled client.

    Args:
        prompt: Either a string (text-only) or list of content blocks (multimodal)
        model: Model identifier (OpenRouter slug or simple name)
        request_id: Unique request identifier for response chunks
        logger: Session logger for recording the interaction
        tools: Optional list of tool definitions for function calling

    Model selection: OpenRouter-style slugs or simple names (opus/sonnet/haiku)
    are resolved to Claude Code model identifiers. Pool replaces clients
    on-demand when a different model is requested.

    Note: When tools are provided, we buffer the response to parse JSON at the end
    since we're emulating function calling through prompting.
    """
    resolved_model = resolve_model(model)
    created = int(time.time())
    start_time = time.monotonic()
    finish_reason = "stop"

    # Add tool prompt if tools are provided
    effective_prompt = prompt
    if tools:
        tool_suffix = build_tool_prompt(tools)
        if isinstance(prompt, str):
            effective_prompt = prompt + tool_suffix
        else:
            effective_prompt = prompt.copy()
            if effective_prompt and effective_prompt[-1].get("type") == "text":
                effective_prompt[-1] = {
                    "type": "text",
                    "text": effective_prompt[-1]["text"] + tool_suffix,
                }
            else:
                effective_prompt.append({"type": "text", "text": tool_suffix})

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

    try:
        async with pool.acquire(resolved_model) as client:
            # For multimodal content, create an async generator
            if isinstance(effective_prompt, list):
                await client.query(make_multimodal_prompt(effective_prompt))
            else:
                await client.query(effective_prompt)
            async for msg in client.receive_response():
                # Check timeout
                if time.monotonic() - start_time > CLAUDE_TIMEOUT:
                    logger.log_error(f"Timeout after {CLAUDE_TIMEOUT}s")
                    raise asyncio.TimeoutError(f"Claude SDK timed out after {CLAUDE_TIMEOUT}s")
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            logger.log_chunk(block.text)
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
                    break

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
                logger.log_chunk(f"[parsed tool_call: {tool_calls[0].function.name}]")
            else:
                # No tool calls found, send the text
                chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[StreamChoice(delta=DeltaMessage(content=full_text))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        logger.log_finish(finish_reason)
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
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Note: Concurrency is managed by the client pool (POOL_SIZE env var).
    Model selection supports OpenRouter-style slugs (e.g., anthropic/claude-sonnet-4)
    or simple names (opus, sonnet, haiku). Model parameter is required.

    Tool calling is supported via the `tools` and `tool_choice` parameters.
    When the model decides to use tools, the response will include `tool_calls`
    with `finish_reason="tool_calls"`.
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
                async for chunk in stream_claude_sdk(
                    prompt, request.model, request_id, logger, request.tools
                ):
                    yield chunk
            finally:
                logger.write(request.messages, request.stream, request.temperature, request.max_tokens)

        return StreamingResponse(
            stream_with_logging(),
            media_type="text/event-stream",
        )

    response = await call_claude_sdk(prompt, request.model, logger, request.tools)
    logger.write(request.messages, request.stream, request.temperature, request.max_tokens)

    # Build response message based on whether we have tool calls
    if response.has_tool_calls:
        response_message = Message(
            role="assistant",
            content=response.text if response.text else None,
            tool_calls=response.tool_calls,
        )
    else:
        response_message = Message(role="assistant", content=response.text)

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
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
