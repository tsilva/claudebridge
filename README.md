<div align="center">

# claude-code-bridge

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)

**Bridge OpenAI tools to Claude Code SDK - use your Claude subscription with any OpenAI client**

</div>

## Features

- ⚡ **Lightweight** - ~150 lines of Python, minimal dependencies
- 🔄 **OpenAI-compatible** - Drop-in replacement for `/v1/chat/completions`
- 💰 **Uses your subscription** - No API keys needed, uses Claude Code OAuth
- 🌊 **Streaming support** - Real-time SSE responses
- 🔀 **Concurrent requests** - Handles multiple requests with semaphore limiting

## Quick Start

```bash
# Install
git clone https://github.com/tsilva/claude-code-bridge
cd claude-code-bridge
uv venv && uv pip install -e .

# Run
source .venv/bin/activate
claude-proxy
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

### With OpenAI Python client

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

The proxy uses your existing Claude Code authentication. Make sure you're logged in:

```bash
claude login
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `MAX_CONCURRENT` | `3` | Max concurrent Claude SDK calls |

## Requirements

- Python 3.10+
- Active Claude Code subscription
- Claude Code CLI authenticated (`claude login`)

## License

MIT
