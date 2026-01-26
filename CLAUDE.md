# claude-code-bridge

Bridge OpenAI tools to Claude Code SDK. Uses your active Claude subscription.

## Quick Start

```bash
uv pip install -e .
claude-code-bridge
```

## Endpoints

- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `GET /v1/models` - List available models
- `GET /health` - Health check

## Available Models

- `opus`, `sonnet`, `haiku` (also accepts `claude-3-sonnet`, `claude-sonnet`, etc.)

## Architecture

```
claude_code_bridge/
├── server.py         # FastAPI app, endpoints, Claude SDK integration
├── models.py         # Pydantic models for OpenAI request/response format
├── client.py         # CLI client
└── session_logger.py # Request/response logging
```

## Key Implementation Details

- **Concurrency**: Uses `asyncio.Semaphore(3)` to limit concurrent Claude SDK calls
- **Streaming**: SSE format matching OpenAI's streaming response
- **Model mapping**: Flexible model name mapping (e.g., `claude-3-sonnet` → `sonnet`)
- **User settings**: Uses `setting_sources=["user"]` to load user's Claude Code settings (including default model)
- **System prompt**: Uses `system_prompt={"type": "preset", "preset": "claude_code"}` to preserve the default Claude Code system prompt

## Testing

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Usage with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `claude-agent-sdk` - Claude Code SDK for Python

## README.md

Keep README.md updated with any significant project changes.
