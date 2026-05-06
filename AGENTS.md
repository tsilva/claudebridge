# agentbridge

Bridge OpenAI tools to Claude Code SDK, Codex CLI, and OpenRouter.

## Quick Start

```bash
uv pip install -e .
agentbridge
```

## Endpoints

- `POST /api/v1/chat/completions` - Chat completions (streaming supported)
- `GET /api/v1/models` - List available models
- `GET /health` - Health check

## Model Selection

Model IDs must start with an AgentBridge provider namespace:

**Supported formats:**
- Claude Code: `claudecode/opus`, `claudecode/sonnet`, `claudecode/haiku`, or slugs containing those names, such as `claudecode/anthropic/claude-sonnet-4`
- Codex CLI: `codex/<model>`, such as `codex/gpt-5.5`
- OpenRouter: `openrouter/<provider>/<model>`, such as `openrouter/anthropic/claude-sonnet-4`

**Examples:**
```python
# Claude Code
client.chat.completions.create(model="claudecode/sonnet", ...)

# Codex CLI
client.chat.completions.create(model="codex/gpt-5.5", ...)

# OpenRouter
client.chat.completions.create(model="openrouter/anthropic/claude-sonnet-4", ...)
```

Unsupported model IDs return HTTP 400 with an error message listing valid options.

## Architecture

```
agentbridge/
├── server.py         # FastAPI app, endpoints, provider adapters, session logging
├── pool.py           # Dynamic client pool with model replacement
├── models.py         # Pydantic models for OpenAI request/response format and model mapping
├── dashboard.py      # Real-time dashboard with SSE for pool/request monitoring
└── __init__.py       # Package version
```

## Key Implementation Details

- **Client Pool**: Single-use clients — each request gets a fresh (or pre-warmed) client, destroyed after use. Background pre-warming hides creation latency. Never reuses clients across requests to prevent cross-contamination.
- **Concurrency**: Worker count controls concurrent requests (default: 3, configurable via `--workers`/`-w` flag or `POOL_SIZE` env var)
- **Streaming**: SSE format matching OpenAI's streaming response
- **Model selection**: Resolves required provider prefixes (`claudecode/`, `codex/`, `openrouter/`) before dispatch. Model parameter is required.
- **Pure chat mode**: Tools are disabled (`tools=[]`) - Claude operates as a conversational assistant without file access, bash commands, or web access
- **Isolated settings**: Uses `setting_sources=None` to not load user filesystem settings, providing isolation
- **System prompt**: Uses `system_prompt={"type": "preset", "preset": "claude_code"}` to preserve the default Claude Code system prompt

## Environment Variables

- `POOL_SIZE` - Number of pooled clients (default: 1, configurable via `--workers`)
- `CLAUDE_TIMEOUT` - Request timeout in seconds (default: 120)
- `CODEX_TIMEOUT` - Codex request timeout in seconds (defaults to `CLAUDE_TIMEOUT`)
- `OPENROUTER_TIMEOUT` - OpenRouter request timeout in seconds (defaults to `CLAUDE_TIMEOUT`)
- `OPENROUTER_API_KEY` - API key required for OpenRouter requests
- `PORT` - Server port (default: 8082)

## Testing

```bash
# Using Claude Code
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claudecode/sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Using Codex
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "codex/gpt-5.5", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openrouter/anthropic/claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Usage with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8082/api/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="claudecode/sonnet",
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
