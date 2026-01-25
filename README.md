<div align="center">

# claude-code-bridge

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**🌉 Bridge OpenAI-compatible tools to Claude Code SDK — use your Claude subscription with any OpenAI client**

</div>

## Features

- ⚡ **Lightweight** — ~200 lines of Python, minimal dependencies
- 🔄 **OpenAI-compatible** — Drop-in replacement for `/v1/chat/completions`
- 💰 **Uses your subscription** — No API keys needed, uses Claude Code OAuth
- 🌊 **Streaming support** — Real-time SSE responses matching OpenAI format
- 🔀 **Concurrent requests** — Handles multiple requests with semaphore limiting
- 📝 **Session logging** — Full request/response logging for debugging

## Quick Start

```bash
# Clone and install
git clone https://github.com/tsilva/claude-code-bridge
cd claude-code-bridge
uv venv && uv pip install -e .

# Run the server
source .venv/bin/activate
claude-bridge
```

Server starts at `http://localhost:8000`

## Usage

### With curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sonnet",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
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

```bash
# Direct prompt
claude-client "What is Python?"

# Pipe from stdin
echo "Hello" | claude-client

# Use different model
claude-client --model opus "Explain decorators"

# Non-streaming mode
claude-client --no-stream "Quick answer"
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

The proxy uses your existing Claude Code authentication:

```bash
claude login
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `MAX_CONCURRENT` | `3` | Max concurrent Claude SDK calls |

## Architecture

```
claude_proxy/
├── server.py         # FastAPI app, endpoints, Claude SDK integration
├── models.py         # Pydantic models for OpenAI request/response format
├── client.py         # CLI client for the proxy
└── session_logger.py # Request/response logging
```

## Requirements

- Python 3.10+
- Active Claude Code subscription
- Claude Code CLI authenticated (`claude login`)

## License

MIT
