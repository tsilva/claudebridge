<div align="center">
  <img src="./logo.png" alt="agentbridge" width="512" />

  **🌉 Bridge OpenAI tools to Claude Code SDK or Codex CLI — use your subscriptions anywhere 🔌**
</div>

agentbridge is a local OpenAI-compatible API server for Claude Code SDK and Codex CLI. It lets tools that already speak the OpenAI Chat Completions API use your active Claude Code or Codex login through `http://localhost:8082/api/v1`.

It supports non-streaming and streaming chat completions, multimodal image/PDF inputs for Claude, image inputs for Codex, OpenRouter-style model slugs, a small live dashboard, and JSON session logs for debugging.

> **Legal notice:** agentbridge uses Claude Code SDK and Codex CLI access through your local subscriptions, not the Anthropic or OpenAI APIs directly. Whether this is allowed under the relevant service terms is your responsibility to evaluate. Use it conservatively and at your own risk.

## Install

```bash
uv tool install agentbridge-py
claude login
codex login
agentbridge
```

Open [http://localhost:8082/dashboard](http://localhost:8082/dashboard), or point an OpenAI-compatible client at `http://localhost:8082/api/v1`.

Install from source when working on the repo:

```bash
git clone https://github.com/tsilva/agentbridge.git
cd agentbridge
uv pip install -e .
agentbridge
```

## Usage

```bash
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```bash
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-5.5", "reasoning_effort": "high", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/api/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="openai/gpt-5.5",
    reasoning_effort="high",
    messages=[{"role": "user", "content": "Hello from Codex!"}],
)
print(response.choices[0].message.content)
```

## Commands

```bash
agentbridge                         # start on http://127.0.0.1:8082
agentbridge --port 8083             # choose another port
agentbridge -w 3                    # run with three pooled workers
agentbridge --version               # print package and git version
uv tool install . --force --no-cache # reinstall local source build
uv run --extra test pytest -m unit   # run unit tests
uv run --extra test pytest           # run full tests; integration tests expect a server
```

## Notes

- Requires Python 3.12+ and at least one authenticated backend: `claude login` for Claude models or `codex login` for Codex models.
- Endpoints: `POST /api/v1/chat/completions`, `GET /api/v1/models`, `GET /health`, and `/dashboard`.
- Claude model inputs are `opus`, `sonnet`, `haiku`, or slugs containing those names, such as `anthropic/claude-sonnet-4`.
- Codex model inputs are `openai/<model>`, `codex/<model>`, or bare `gpt-5...` model IDs. The requested model is passed directly to Codex CLI; `gpt-5.5` defaults to `reasoning_effort="high"` unless the request sets another effort.
- `PORT`, `POOL_SIZE`, `CLAUDE_TIMEOUT`, `CODEX_TIMEOUT`, `LOG_DIR`, and `MAX_LOG_FILES` control local runtime behavior.
- Each Claude request gets a fresh or pre-warmed Claude SDK client and the client is destroyed after use. Each Codex request runs an ephemeral `codex exec` process in an isolated temporary directory.
- Claude Code tools are disabled for SDK sessions. Codex runs with read-only sandboxing, no approvals, ephemeral sessions, and ignored project rules. OpenAI-style function calling is emulated by prompting for JSON tool-call output.
- Session logs are written as JSON under `logs/sessions` by default; base64 image and PDF attachments are saved beside their request logs.

## Architecture

![agentbridge architecture diagram](./architecture.png)

## License

[MIT](LICENSE)
