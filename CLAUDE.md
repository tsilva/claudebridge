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

## Model Selection

Supports OpenRouter-style model slugs for unified model naming across providers:

**Supported formats:**
- OpenRouter slugs: `anthropic/claude-sonnet-4`, `anthropic/claude-opus-4.5`, etc.
- Simple names: `opus`, `sonnet`, `haiku`
- No model: Uses your Claude Code user settings default

**Examples:**
```python
# OpenRouter-style slug
client.chat.completions.create(model="anthropic/claude-sonnet-4", ...)

# Simple name
client.chat.completions.create(model="sonnet", ...)

# Use default from Claude Code settings
client.chat.completions.create(model=None, ...)
```

Unsupported model IDs return HTTP 400 with an error message listing valid options.

## Architecture

```
claude_code_bridge/
├── server.py         # FastAPI app, endpoints, Claude SDK integration
├── pool.py           # Client pool for connection reuse
├── models.py         # Pydantic models for OpenAI request/response format
├── model_mapping.py  # OpenRouter slug to Claude Code model resolution
├── client.py         # CLI client
└── session_logger.py # Request/response logging
```

## Key Implementation Details

- **Client Pool**: Pre-spawns `ClaudeSDKClient` instances for reduced latency. Uses `/clear` command between requests to reset conversation state while keeping subprocesses warm.
- **Concurrency**: Pool size controls concurrent requests (default: 3, configurable via `POOL_SIZE` env var)
- **Streaming**: SSE format matching OpenAI's streaming response
- **Model selection**: Resolves OpenRouter slugs to Claude Code models via `/model` command. Uses user defaults when no model specified.
- **User settings**: Uses `setting_sources=["user"]` to load user's Claude Code settings (including default model)
- **System prompt**: Uses `system_prompt={"type": "preset", "preset": "claude_code"}` to preserve the default Claude Code system prompt

## Environment Variables

- `POOL_SIZE` - Number of pooled clients (default: 3)
- `CLAUDE_TIMEOUT` - Request timeout in seconds (default: 120)
- `PORT` - Server port (default: 8000)

## Testing

```bash
# Using OpenRouter-style slug
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}]}'

# Using simple name
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Usage with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",  # or "sonnet"
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
