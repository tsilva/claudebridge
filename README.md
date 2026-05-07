<div align="center">
  <img src="./logo.png" alt="agentbridge" width="512" />

  **🌉 Bridge OpenAI tools to Claude Code SDK, Codex CLI, or OpenRouter — use your subscriptions anywhere 🔌**
</div>

agentbridge is a local OpenAI-compatible API server with provider adapters for Claude Code SDK, Codex CLI, and OpenRouter. It lets tools that already speak the OpenAI Chat Completions API use your active local logins or OpenRouter API key through `http://localhost:8082/api/v1`.

It supports non-streaming and streaming chat completions, multimodal image/PDF inputs for Claude, image inputs for Codex, namespaced provider model IDs, a small live dashboard, and JSON session logs for debugging.

> **Legal notice:** agentbridge can use Claude Code SDK and Codex CLI access through your local subscriptions, and can also forward requests to OpenRouter when configured. Whether this is allowed under the relevant service terms is your responsibility to evaluate. Use it conservatively and at your own risk.

## Install

```bash
uv tool install agentbridge-py
claude login
codex login
agentbridge
```

Open [http://localhost:8082/dashboard](http://localhost:8082/dashboard), use Chat at [http://localhost:8082/dashboard/chat](http://localhost:8082/dashboard/chat), or point an OpenAI-compatible client at `http://localhost:8082/api/v1`.

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
  -d '{"model": "claudecode/sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```bash
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "codex/gpt-5.5", "reasoning_effort": "high", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```bash
# On first start, agentbridge creates ~/.config/agentbridge/.env.
# Put OPENROUTER_API_KEY=sk-or-... there, or set it in the process environment.
agentbridge
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openrouter/anthropic/claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/api/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="claudecode/anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="codex/gpt-5.5",
    reasoning_effort="high",
    messages=[{"role": "user", "content": "Hello from Codex!"}],
)
print(response.choices[0].message.content)

response = client.chat.completions.create(
    model="openrouter/anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello from OpenRouter!"}],
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

## Publishing

Publishing to PyPI runs through the `Release` GitHub Actions workflow and targets the `agentbridge-py` project. Configure PyPI Trusted Publishing for:

- Owner: `tsilva`
- Repository: `agentbridge`
- Workflow: `release.yml`
- Environment: `pypi`

No PyPI API token is required. The workflow builds the package, verifies the built project metadata is `agentbridge-py`, creates the GitHub release, then publishes `dist/` to PyPI.

## Notes

- Requires Python 3.12+ and at least one authenticated backend: `claude login` for Claude Code models, `codex login` for Codex models, or `OPENROUTER_API_KEY` for OpenRouter models.
- Endpoints: `POST /api/v1/chat/completions`, `GET /api/v1/models`, `GET /health`, `/dashboard`, and `/dashboard/chat`.
- Model IDs must be prefixed with an AgentBridge provider namespace: `claudecode/<model>`, `codex/<model>`, or `openrouter/<provider>/<model>`.
- Claude Code model inputs are `claudecode/opus`, `claudecode/sonnet`, `claudecode/haiku`, or namespaced slugs containing those names, such as `claudecode/anthropic/claude-sonnet-4`.
- Codex model inputs are `codex/<model>`. The requested model is passed directly to Codex CLI; `codex/gpt-5.5` defaults to `reasoning_effort="high"` unless the request sets another effort.
- OpenRouter model inputs are `openrouter/<provider>/<model>`, for example `openrouter/anthropic/claude-sonnet-4`. The upstream model ID after `openrouter/` is passed through the official OpenRouter Python SDK.
- User configuration lives in `~/.config/agentbridge/`. AgentBridge creates `~/.config/agentbridge/.env` on startup for local keys such as `OPENROUTER_API_KEY`; process environment variables still take precedence. Set `AGENTBRIDGE_CONFIG_DIR` to move this directory.
- `PORT`, `POOL_SIZE`, `CLAUDE_TIMEOUT`, `CODEX_TIMEOUT`, `OPENROUTER_TIMEOUT`, `OPENROUTER_API_KEY`, `LOG_DIR`, and `MAX_LOG_FILES` control local runtime behavior.
- Claude SDK clients are created lazily for the requested model and reused from an idle pool capped by `POOL_SIZE`; no Claude clients are warmed at server boot. Each Codex request runs an ephemeral `codex exec` process in an isolated temporary directory.
- Claude Code tools are disabled for SDK sessions. Codex runs with read-only sandboxing, no approvals, ephemeral sessions, and ignored project rules. OpenAI-style function calling is emulated by prompting for JSON tool-call output.
- Session logs are written as JSON under `~/.config/agentbridge/logs/sessions` by default; base64 image and PDF attachments are saved beside their request logs. `LOG_DIR` overrides this.

## Architecture

![agentbridge architecture diagram](./architecture.png)

## License

[MIT](LICENSE)
