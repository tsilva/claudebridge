<div align="center">
  <img src="logo.png" alt="claude-code-bridge" width="512"/>

  # claude-code-bridge

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

  **Bridge OpenAI-compatible tools to Claude Code SDK — use your Claude subscription with any OpenAI client**

</div>

## Features

- **Lightweight** — ~200 lines of Python, minimal dependencies
- **OpenAI-compatible** — Drop-in replacement for `/v1/chat/completions`
- **Uses your subscription** — No API keys needed, uses Claude Code OAuth
- **Streaming support** — Real-time SSE responses matching OpenAI format
- **Connection pooling** — Pre-spawned clients for reduced latency
- **Session logging** — Full request/response logging for debugging

## Quick Start

```bash
# Install globally
uv tool install git+https://github.com/tsilva/claude-code-bridge

# Or install from source
git clone https://github.com/tsilva/claude-code-bridge
cd claude-code-bridge
uv pip install -e .

# Run the server
claude-code-bridge
```

Server starts at `http://localhost:8000`

## Usage

### With curl

```bash
# Non-streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

### With OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### CLI Client

The CLI client can be used for ad-hoc testing:

```bash
# Direct prompt
python -m claude_code_bridge.client "What is Python?"

# Pipe from stdin
echo "Hello" | python -m claude_code_bridge.client

# Use different model
python -m claude_code_bridge.client --model opus "Explain decorators"

# Non-streaming mode
python -m claude_code_bridge.client --no-stream "Quick answer"

# Multiple parallel requests
python -m claude_code_bridge.client -n 3 "Hello"
```

### BridgeClient Library

Use `BridgeClient` programmatically for testing or integration:

```python
from claude_code_bridge.client import BridgeClient

# Sync usage
with BridgeClient() as client:
    if client.health_check():
        models = client.list_models()
        response = client.complete_sync("Hello!", stream=False)
        print(response)

# Async usage
import asyncio

async def main():
    async with BridgeClient() as client:
        response = await client.complete("Hello!")
        print(response)

        # Or stream chunks
        async for chunk in client.stream("Tell me a story"):
            print(chunk, end="")

asyncio.run(main())
```

## Testing

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run the test suite (requires server running)
uv run pytest tests/test_client.py -v
```

## Available Models

| Model ID | Description |
|----------|-------------|
| `opus` | Claude Opus (most capable) |
| `sonnet` | Claude Sonnet (balanced) |
| `haiku` | Claude Haiku (fastest) |

Also accepts: `claude-opus`, `claude-sonnet`, `claude-haiku`, `claude-3-sonnet`, `claude-3.5-sonnet`, etc.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (OpenAI format) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## Configuration

The bridge uses your existing Claude Code authentication:

```bash
claude login
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `POOL_SIZE` | `3` | Number of pooled clients |
| `CLAUDE_TIMEOUT` | `120` | Request timeout in seconds |

### Client Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BRIDGE_URL` | `http://localhost:8000` | API base URL |
| `OPENROUTER_API_KEY` | - | API key (if needed) |
| `OPENROUTER_MODEL` | `default` | Default model |

## Architecture

```
claude_code_bridge/
├── server.py         # FastAPI app, endpoints, Claude SDK integration
├── pool.py           # Client pool for connection reuse
├── models.py         # Pydantic models for OpenAI request/response format
├── client.py         # BridgeClient library + CLI
└── session_logger.py # Request/response logging
```

## Requirements

- Python 3.10+
- Active Claude Code subscription
- Claude Code CLI authenticated (`claude login`)

## License

MIT
