<div align="center">
  <img src="logo.png" alt="claudebridge" width="512"/>

  # claudebridge

  [![CI](https://github.com/tsilva/claudebridge/actions/workflows/ci.yml/badge.svg)](https://github.com/tsilva/claudebridge/actions/workflows/ci.yml)
  [![PyPI](https://img.shields.io/pypi/v/claudebridge.svg)](https://pypi.org/project/claudebridge/)
  [![Python 3.12+](https://img.shields.io/pypi/pyversions/claudebridge.svg)](https://pypi.org/project/claudebridge/)
  [![License: MIT](https://img.shields.io/github/license/tsilva/claudebridge.svg)](LICENSE)
  [![GitHub stars](https://img.shields.io/github/stars/tsilva/claudebridge?style=social)](https://github.com/tsilva/claudebridge)

  **🌉 Bridge OpenAI tools to Claude Code SDK — use your subscription anywhere 🔌**

  > ⚠️ **Legal Notice**: This tool bridges OpenAI-compatible clients to Claude using the Claude Code SDK. The permissibility of this usage under Anthropic's Terms of Service is unclear. **Use at your own peril.** Please review the [Legal Disclaimer](#legal-disclaimer) section before use.

</div>

## Why claudebridge?

You have a Claude subscription. You have tools that speak OpenAI's API.
claudebridge connects them — no API keys, no extra costs.

| | claudebridge | LiteLLM | Direct API |
|---|:---:|:---:|:---:|
| Uses your Claude subscription | Yes | No | No |
| No API key needed | Yes | No | No |
| One command to start | Yes | ~Yes | No |
| OpenAI-compatible | Yes | Yes | No |

## Features

- **Lightweight** — Minimal dependencies, easy to understand
- **OpenAI-compatible** — Drop-in replacement for `/api/v1/chat/completions`
- **Uses your subscription** — No API keys needed, uses Claude Code OAuth
- **Streaming support** — Real-time SSE responses matching OpenAI format
- **Connection pooling** — Pre-spawned clients for reduced latency
- **Session logging** — Full request/response logging for debugging

## Quick Start

```bash
# Install globally
uv tool install git+https://github.com/tsilva/claudebridge

# Or install from source
git clone https://github.com/tsilva/claudebridge
cd claudebridge
uv pip install -e .

# Run the server
claudebridge

# Verify installed version
claudebridge --version
```

### Local Development

When reinstalling from source, use `--no-cache` to ensure you get the latest code:

```bash
uv tool install . --force --no-cache
```

Server starts at `http://localhost:8082`

## Usage

### With curl

```bash
# Non-streaming
curl -X POST http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl -X POST http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

### With OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/api/v1",
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
python -m claudebridge.client "What is Python?"

# Pipe from stdin
echo "Hello" | python -m claudebridge.client

# Use different model
python -m claudebridge.client --model opus "Explain decorators"

# Non-streaming mode
python -m claudebridge.client --no-stream "Quick answer"

# Multiple parallel requests
python -m claudebridge.client -n 3 "Hello"
```

### BridgeClient Library

Use `BridgeClient` programmatically for testing or integration:

```python
from claudebridge.client import BridgeClient

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

<details>
<summary><strong>Compatible Tools</strong></summary>

- **Cursor** — Use Claude through Cursor's OpenAI-compatible backend
- **Continue.dev** — VS Code extension with OpenAI endpoint support
- **Open WebUI** — Self-hosted ChatGPT-like interface
- **LangChain / LlamaIndex** — Via OpenAI provider
- **Any OpenAI SDK client** — Python, TypeScript, Go, etc.
</details>

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
| `/api/v1/chat/completions` | POST | Chat completions (OpenAI format) |
| `/api/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## Configuration

The bridge uses your existing Claude Code authentication:

```bash
claude login
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8082` | Server port |
| `POOL_SIZE` | `3` | Number of workers (also settable via `--workers`/`-w` flag) |
| `CLAUDE_TIMEOUT` | `120` | Request timeout in seconds |

### Client Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BRIDGE_URL` | `http://localhost:8082` | API base URL |
| `OPENROUTER_API_KEY` | - | API key (if needed) |
| `OPENROUTER_MODEL` | `default` | Default model |

## Architecture

```
claudebridge/
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

## Legal Disclaimer

**⚠️ Use at Your Own Peril**

This tool (`claudebridge`) creates a bridge between OpenAI-compatible clients and Claude using the Claude Code SDK. The permissibility of this usage under [Anthropic's Terms of Service](https://www.anthropic.com/legal/commercial-terms) is **not clearly defined** and subject to interpretation.

### The Ambiguity

Anthropic's terms contain provisions that may affect this usage:

- **Commercial Terms**: Prohibit building competing products, reverse engineering, or reselling services
- **Consumer Terms**: Restrict automated/non-human access except via approved APIs
- **Both**: Prohibit developing competing products or training competing AI models

This tool:
- Uses the Claude Code SDK (not the official Anthropic API)
- Enables programmatic access to Claude through your existing subscription
- Could be interpreted as "automated access" or a "competing product"

### Our Interpretation

We believe that for **lightweight personal local usage**, this should fall within acceptable use since:
- It requires an active Claude subscription
- It uses the official Claude Code SDK
- It's for personal/local development purposes only
- It doesn't compete with or replace Anthropic's services

**However, this interpretation is disputable and may not align with Anthropic's view.**

### Guidelines for Conservative Use

Even if you conclude this usage is legitimate, **act conservatively**:

- **For yourself only** — Do not share access, create multi-user services, or allow others to use your instance
- **Stay local** — Run only on your personal machine, not on servers or cloud infrastructure
- **Minimal usage** — Use sparingly and only for genuine personal development needs
- **No automation** — Do not build automated pipelines, bots, or services on top of this tool
- **No redistribution** — Do not package or distribute this as part of other products
- **Stay within boundaries** — Respect rate limits and avoid any behavior that could be seen as abuse

### Your Responsibility

By using this tool, you acknowledge that:

1. You have read and understand [Anthropic's Terms of Service](https://www.anthropic.com/legal/commercial-terms)
2. You accept that this usage may violate those terms
3. You use this tool entirely at your own peril
4. You are solely responsible for any consequences of use
5. We assume no liability for your use of this tool

**We strongly encourage you to:**
- Review Anthropic's terms yourself
- Make your own determination about permissibility
- Err on the side of caution
- Contact Anthropic if you need clarification
- Discontinue use if you have any concerns

This project is provided as-is for educational purposes only.

## License

MIT
