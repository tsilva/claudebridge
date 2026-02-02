# claude-code-bridge

Bridge OpenAI tools to Claude Code SDK. Uses your active Claude subscription.

## Quick Start

```bash
uv pip install -e .
claude-code-bridge
```

## Endpoints

- `POST /api/v1/chat/completions` - Chat completions (streaming supported)
- `GET /api/v1/models` - List available models
- `GET /health` - Health check

## Model Selection

Supports OpenRouter-style model slugs for unified model naming across providers:

**Supported formats:**
- OpenRouter slugs: `anthropic/claude-sonnet-4`, `anthropic/claude-opus-4.5`, etc.
- Simple names: `opus`, `sonnet`, `haiku`

**Examples:**
```python
# OpenRouter-style slug
client.chat.completions.create(model="anthropic/claude-sonnet-4", ...)

# Simple name
client.chat.completions.create(model="sonnet", ...)
```

Unsupported model IDs return HTTP 400 with an error message listing valid options.

## Architecture

```
claude_code_bridge/
├── server.py         # FastAPI app, endpoints, Claude SDK integration
├── pool.py           # Dynamic client pool with model replacement
├── models.py         # Pydantic models for OpenAI request/response format
├── model_mapping.py  # OpenRouter slug to Claude Code model resolution
├── client.py         # CLI client
└── session_logger.py # Request/response logging
```

## Key Implementation Details

- **Client Pool**: Single dynamic pool that tracks model per client. Initializes with opus clients by default. Replaces clients on-demand when a different model is requested. Prefers reusing clients with matching models. Uses `/clear` command between requests to reset conversation state.
- **Concurrency**: Pool size controls concurrent requests (default: 3, configurable via `POOL_SIZE` env var)
- **Streaming**: SSE format matching OpenAI's streaming response
- **Model selection**: Resolves OpenRouter slugs to Claude Code models. Model parameter is required.
- **Pure chat mode**: Tools are disabled (`tools=[]`) - Claude operates as a conversational assistant without file access, bash commands, or web access
- **Isolated settings**: Uses `setting_sources=None` to not load user filesystem settings, providing isolation
- **System prompt**: Uses `system_prompt={"type": "preset", "preset": "claude_code"}` to preserve the default Claude Code system prompt

## Environment Variables

- `POOL_SIZE` - Number of pooled clients (default: 3)
- `CLAUDE_TIMEOUT` - Request timeout in seconds (default: 120)
- `PORT` - Server port (default: 8082)

## Testing

```bash
# Using OpenRouter-style slug
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}]}'

# Using simple name
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Usage with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8082/api/v1", api_key="not-needed")
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
