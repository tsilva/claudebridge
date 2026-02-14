# Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an htmx + Jinja2 dashboard at `/dashboard` for viewing active requests, live token streams, pool status, and completed request logs.

**Architecture:** New `dashboard_state.py` tracks active requests in-memory with asyncio.Queue fan-out for live token streaming. New `dashboard_routes.py` serves Jinja2 templates and SSE endpoints. Minimal hooks added to existing `server.py` to emit state changes. Templates use htmx for polling and SSE for real-time updates. Pico CSS for styling from CDN.

**Tech Stack:** FastAPI, Jinja2, htmx (CDN), Pico CSS (CDN), SSE (via `sse-starlette` or raw `StreamingResponse`)

---

### Task 1: Add jinja2 dependency

**Files:**
- Modify: `pyproject.toml:19-26`

**Step 1: Add jinja2 to dependencies**

In `pyproject.toml`, add `jinja2` to the dependencies list:

```python
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "claude-agent-sdk>=0.1.0",
    "python-dotenv>=1.0.0",
    "openai>=1.0.0",
    "fpdf>=1.7.2",
    "jinja2>=3.1.0",
]
```

**Step 2: Install updated dependencies**

Run: `uv pip install -e .`
Expected: Installs jinja2, all other deps remain unchanged.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(dashboard): add jinja2 dependency"
```

---

### Task 2: Create dashboard_state.py with tests

**Files:**
- Create: `claudebridge/dashboard_state.py`
- Create: `tests/test_dashboard_state.py`

**Step 1: Write the tests**

```python
"""Tests for dashboard state tracking."""

import asyncio
import pytest
from claudebridge.dashboard_state import DashboardState


@pytest.fixture
def state():
    return DashboardState()


class TestRequestTracking:
    """Test active request lifecycle."""

    @pytest.mark.unit
    def test_request_started_adds_to_active(self, state):
        state.request_started("req-1", "sonnet")
        active = state.get_active_requests()
        assert len(active) == 1
        assert active[0]["request_id"] == "req-1"
        assert active[0]["model"] == "sonnet"
        assert active[0]["status"] == "active"

    @pytest.mark.unit
    def test_request_completed_removes_from_active(self, state):
        state.request_started("req-1", "sonnet")
        state.request_completed("req-1")
        assert len(state.get_active_requests()) == 0

    @pytest.mark.unit
    def test_request_completed_unknown_id_no_error(self, state):
        state.request_completed("nonexistent")  # Should not raise

    @pytest.mark.unit
    def test_multiple_active_requests(self, state):
        state.request_started("req-1", "sonnet")
        state.request_started("req-2", "opus")
        active = state.get_active_requests()
        assert len(active) == 2

    @pytest.mark.unit
    def test_request_error_removes_from_active(self, state):
        state.request_started("req-1", "sonnet")
        state.request_errored("req-1", "Timeout")
        assert len(state.get_active_requests()) == 0


class TestChunkTracking:
    """Test chunk reception and fan-out."""

    @pytest.mark.unit
    def test_chunk_received_increments_count(self, state):
        state.request_started("req-1", "sonnet")
        state.chunk_received("req-1", "Hello")
        state.chunk_received("req-1", " world")
        active = state.get_active_requests()
        assert active[0]["chunks_received"] == 2

    @pytest.mark.unit
    def test_chunk_received_unknown_id_no_error(self, state):
        state.chunk_received("nonexistent", "text")  # Should not raise


class TestSubscription:
    """Test SSE subscriber fan-out for live token streaming."""

    @pytest.mark.unit
    async def test_subscribe_receives_chunks(self, state):
        state.request_started("req-1", "sonnet")
        queue = state.subscribe("req-1")

        state.chunk_received("req-1", "Hello")
        state.chunk_received("req-1", " world")

        assert await asyncio.wait_for(queue.get(), timeout=1) == {"type": "chunk", "text": "Hello"}
        assert await asyncio.wait_for(queue.get(), timeout=1) == {"type": "chunk", "text": " world"}

    @pytest.mark.unit
    async def test_subscribe_receives_done_on_completion(self, state):
        state.request_started("req-1", "sonnet")
        queue = state.subscribe("req-1")

        state.request_completed("req-1")

        msg = await asyncio.wait_for(queue.get(), timeout=1)
        assert msg["type"] == "done"

    @pytest.mark.unit
    async def test_subscribe_receives_error(self, state):
        state.request_started("req-1", "sonnet")
        queue = state.subscribe("req-1")

        state.request_errored("req-1", "Timeout after 120s")

        msg = await asyncio.wait_for(queue.get(), timeout=1)
        assert msg["type"] == "error"
        assert msg["error"] == "Timeout after 120s"

    @pytest.mark.unit
    async def test_multiple_subscribers_fan_out(self, state):
        state.request_started("req-1", "sonnet")
        q1 = state.subscribe("req-1")
        q2 = state.subscribe("req-1")

        state.chunk_received("req-1", "Hello")

        msg1 = await asyncio.wait_for(q1.get(), timeout=1)
        msg2 = await asyncio.wait_for(q2.get(), timeout=1)
        assert msg1 == msg2 == {"type": "chunk", "text": "Hello"}

    @pytest.mark.unit
    def test_subscribe_unknown_request_returns_none(self, state):
        queue = state.subscribe("nonexistent")
        assert queue is None

    @pytest.mark.unit
    async def test_unsubscribe_removes_queue(self, state):
        state.request_started("req-1", "sonnet")
        queue = state.subscribe("req-1")
        state.unsubscribe("req-1", queue)

        # After unsubscribe, new chunks should not go to this queue
        state.chunk_received("req-1", "Hello")
        assert queue.empty()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_state.py -v -m unit`
Expected: All tests FAIL (module not found)

**Step 3: Implement dashboard_state.py**

```python
"""In-memory state tracking for the dashboard."""

import asyncio
import time


class _ActiveRequest:
    """Tracks a single in-flight request."""

    __slots__ = ("request_id", "model", "start_time", "status", "chunks_received", "_subscribers")

    def __init__(self, request_id: str, model: str):
        self.request_id = request_id
        self.model = model
        self.start_time = time.monotonic()
        self.status = "active"
        self.chunks_received = 0
        self._subscribers: list[asyncio.Queue] = []

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "elapsed_s": round(time.monotonic() - self.start_time, 1),
            "status": self.status,
            "chunks_received": self.chunks_received,
        }


class DashboardState:
    """Tracks active requests for dashboard display.

    Not thread-safe — designed for single async event loop use.
    """

    def __init__(self):
        self._active: dict[str, _ActiveRequest] = {}

    def request_started(self, request_id: str, model: str) -> None:
        self._active[request_id] = _ActiveRequest(request_id, model)

    def chunk_received(self, request_id: str, text: str) -> None:
        req = self._active.get(request_id)
        if req is None:
            return
        req.chunks_received += 1
        msg = {"type": "chunk", "text": text}
        for queue in req._subscribers:
            queue.put_nowait(msg)

    def request_completed(self, request_id: str) -> None:
        req = self._active.pop(request_id, None)
        if req is None:
            return
        for queue in req._subscribers:
            queue.put_nowait({"type": "done"})

    def request_errored(self, request_id: str, error: str) -> None:
        req = self._active.pop(request_id, None)
        if req is None:
            return
        for queue in req._subscribers:
            queue.put_nowait({"type": "error", "error": error})

    def get_active_requests(self) -> list[dict]:
        return [r.to_dict() for r in self._active.values()]

    def subscribe(self, request_id: str) -> asyncio.Queue | None:
        req = self._active.get(request_id)
        if req is None:
            return None
        queue: asyncio.Queue = asyncio.Queue()
        req._subscribers.append(queue)
        return queue

    def unsubscribe(self, request_id: str, queue: asyncio.Queue) -> None:
        req = self._active.get(request_id)
        if req is None:
            return
        try:
            req._subscribers.remove(queue)
        except ValueError:
            pass
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_state.py -v -m unit`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add claudebridge/dashboard_state.py tests/test_dashboard_state.py
git commit -m "feat(dashboard): add in-memory state tracking with fan-out subscriptions"
```

---

### Task 3: Wire dashboard_state into server.py

**Files:**
- Modify: `claudebridge/server.py:73-93` (lifespan), `server.py:378-474` (call_claude_sdk), `server.py:477-644` (stream_claude_sdk)

**Step 1: Write integration tests**

Create `tests/test_dashboard_integration.py`:

```python
"""Tests for dashboard state integration with server functions."""

import pytest
from claudebridge.dashboard_state import DashboardState
from claudebridge.server import dashboard_state


@pytest.mark.unit
class TestDashboardStateIntegration:
    """Verify dashboard_state module-level singleton exists."""

    def test_dashboard_state_exists(self):
        assert dashboard_state is not None
        assert isinstance(dashboard_state, DashboardState)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_integration.py -v -m unit`
Expected: FAIL (cannot import dashboard_state from server)

**Step 3: Add state hooks to server.py**

At the top of `server.py`, after existing imports, add:

```python
from .dashboard_state import DashboardState

# Dashboard state (singleton)
dashboard_state = DashboardState()
```

In `call_claude_sdk()`, add hooks around the existing flow. After `request_id = session_logger.request_id` (line ~403):

```python
    dashboard_state.request_started(request_id, model)
```

After `session_logger.log_finish(response.finish_reason)` (line ~446), add:

```python
        dashboard_state.request_completed(request_id)
```

In the `except asyncio.TimeoutError` block (after `session_logger.log_error(...)`, before `raise`):

```python
        dashboard_state.request_errored(request_id, f"Timeout after {CLAUDE_TIMEOUT}s")
```

In the general `except Exception as e` block (after `session_logger.log_error(...)`, before `raise`):

```python
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
```

In `stream_claude_sdk()`, add similar hooks. After `resolved_model = resolve_model(model)` (line ~500):

```python
    dashboard_state.request_started(request_id, model)
```

Inside the streaming loop, after `session_logger.log_chunk(block.text)` (line ~535):

```python
                            dashboard_state.chunk_received(request_id, block.text)
```

After `session_logger.log_finish(finish_reason)` (line ~591):

```python
        dashboard_state.request_completed(request_id)
```

In the `except asyncio.TimeoutError` block (after `session_logger.log_error(...)`, before the error_chunk):

```python
        dashboard_state.request_errored(request_id, f"Timeout after {CLAUDE_TIMEOUT}s")
```

In the general `except Exception as e` block (after `session_logger.log_error(...)`, before error_chunk):

```python
        dashboard_state.request_errored(request_id, f"{type(e).__name__}: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard_integration.py -v -m unit`
Expected: PASS

**Step 5: Run existing tests to verify no regressions**

Run: `pytest tests/ -v -m unit`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add claudebridge/server.py tests/test_dashboard_integration.py
git commit -m "feat(dashboard): wire state tracking hooks into request lifecycle"
```

---

### Task 4: Create dashboard templates

**Files:**
- Create: `claudebridge/templates/dashboard/page.html`
- Create: `claudebridge/templates/dashboard/active.html`
- Create: `claudebridge/templates/dashboard/recent.html`
- Create: `claudebridge/templates/dashboard/detail.html`
- Create: `claudebridge/templates/dashboard/pool.html`
- Create: `claudebridge/templates/dashboard/stream.html`

**Step 1: Create `page.html` — base layout**

```html
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>claudebridge dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <script src="https://unpkg.com/htmx-ext-sse@2.3.0/sse.js"></script>
    <style>
        :root { --pico-font-size: 14px; }
        body { padding: 1rem; }
        .dashboard { display: grid; grid-template-columns: 1fr 2fr; gap: 1rem; min-height: 80vh; }
        .sidebar { display: flex; flex-direction: column; gap: 1rem; }
        .panel { border: 1px solid var(--pico-muted-border-color); border-radius: var(--pico-border-radius); padding: 1rem; }
        .panel h3 { margin-top: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pico-muted-color); }
        .request-row { display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0.6rem; border-radius: 4px; cursor: pointer; font-size: 0.85rem; font-family: monospace; }
        .request-row:hover { background: var(--pico-muted-border-color); }
        .request-row.active { background: var(--pico-primary-focus); }
        .badge { font-size: 0.75rem; padding: 0.15rem 0.4rem; border-radius: 3px; }
        .badge-ok { background: #2d6a4f; color: #b7e4c7; }
        .badge-err { background: #9d0208; color: #ffd7d7; }
        .badge-active { background: #1d3557; color: #a8dadc; }
        .detail-panel { overflow-y: auto; }
        .detail-panel pre { white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; max-height: 60vh; overflow-y: auto; }
        .pool-bar { display: flex; gap: 0.3rem; align-items: center; }
        .pool-dot { width: 12px; height: 12px; border-radius: 50%; }
        .pool-dot.available { background: #2d6a4f; }
        .pool-dot.in-use { background: #e9c46a; }
        .pool-dot.empty { background: var(--pico-muted-border-color); }
        .elapsed { color: var(--pico-muted-color); }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        header h1 { margin: 0; font-size: 1.2rem; }
        .empty-state { color: var(--pico-muted-color); font-style: italic; font-size: 0.85rem; }
    </style>
</head>
<body>
    <header>
        <h1>claudebridge</h1>
        <div hx-get="/dashboard/pool" hx-trigger="load, every 3s" hx-swap="innerHTML"></div>
    </header>
    <div class="dashboard">
        <div class="sidebar">
            <div class="panel" hx-ext="sse" sse-connect="/dashboard/active" sse-swap="message" hx-swap="innerHTML">
                <h3>Active Requests</h3>
                <p class="empty-state">No active requests</p>
            </div>
            <div class="panel" hx-get="/dashboard/recent" hx-trigger="load, every 5s" hx-swap="innerHTML">
                <h3>Recent Requests</h3>
                <p class="empty-state">Loading...</p>
            </div>
        </div>
        <div class="panel detail-panel" id="detail">
            <h3>Request Detail</h3>
            <p class="empty-state">Click a request to view details</p>
        </div>
    </div>
</body>
</html>
```

**Step 2: Create `active.html` — active requests partial (SSE payload)**

```html
<h3>Active Requests</h3>
{% if requests %}
{% for req in requests %}
<div class="request-row" hx-get="/dashboard/request/{{ req.request_id }}" hx-target="#detail" hx-swap="innerHTML">
    <span>{{ req.request_id[-8:] }} <span class="badge badge-active">{{ req.model }}</span></span>
    <span class="elapsed">{{ req.elapsed_s }}s</span>
</div>
{% endfor %}
{% else %}
<p class="empty-state">No active requests</p>
{% endif %}
```

**Step 3: Create `recent.html` — recent completed requests partial**

```html
<h3>Recent Requests</h3>
{% if logs %}
{% for log in logs %}
<div class="request-row" hx-get="/dashboard/request/{{ log.request_id }}" hx-target="#detail" hx-swap="innerHTML">
    <span>{{ log.request_id[-8:] }}
        {% if log.error %}
        <span class="badge badge-err">err</span>
        {% else %}
        <span class="badge badge-ok">ok</span>
        {% endif %}
        <span class="badge">{{ log.model }}</span>
    </span>
    <span class="elapsed">{{ log.duration_ms }}ms</span>
</div>
{% endfor %}
{% else %}
<p class="empty-state">No recent requests</p>
{% endif %}
```

**Step 4: Create `detail.html` — request detail view**

```html
<h3>{{ request_id }}</h3>
<table>
    <tr><td><strong>Model</strong></td><td>{{ model }}</td></tr>
    <tr><td><strong>Time</strong></td><td>{{ timestamp }}</td></tr>
    <tr><td><strong>Duration</strong></td><td>{{ duration_ms }}ms</td></tr>
    {% if acquire_ms %}<tr><td><strong>Acquire</strong></td><td>{{ acquire_ms }}ms</td></tr>{% endif %}
    {% if query_ms %}<tr><td><strong>Query</strong></td><td>{{ query_ms }}ms</td></tr>{% endif %}
    {% if error %}<tr><td><strong>Error</strong></td><td style="color: #ff6b6b;">{{ error }}</td></tr>{% endif %}
</table>
{% if is_active %}
<h4>Live Stream</h4>
<div hx-ext="sse" sse-connect="/dashboard/stream/{{ request_id }}" sse-swap="chunk" hx-swap="beforeend">
    <pre id="stream-output">{{ buffered_text }}</pre>
</div>
{% else %}
<h4>Messages</h4>
{% for msg in messages %}
<details {% if loop.last %}open{% endif %}>
    <summary>[{{ msg.role }}]</summary>
    <pre>{{ msg.content }}</pre>
</details>
{% endfor %}
{% if response %}
<h4>Response</h4>
<pre>{{ response }}</pre>
{% endif %}
{% endif %}
```

**Step 5: Create `pool.html` — pool status partial**

```html
<div class="pool-bar">
    {% for i in range(size) %}
        {% if i < in_use %}
        <div class="pool-dot in-use" title="in use"></div>
        {% elif i < in_use + available %}
        <div class="pool-dot available" title="available"></div>
        {% else %}
        <div class="pool-dot empty" title="empty"></div>
        {% endif %}
    {% endfor %}
    <small>{{ available }}/{{ size }} available</small>
</div>
```

**Step 6: Create `stream.html` — live token SSE fragment**

This is a minimal SSE data fragment. The actual SSE event format is `event: chunk\ndata: <html>\n\n`. The template renders the HTML that gets appended to the stream output:

```html
{{ text }}
```

**Step 7: Commit**

```bash
git add claudebridge/templates/
git commit -m "feat(dashboard): add htmx + Pico CSS templates"
```

---

### Task 5: Create dashboard_routes.py with tests

**Files:**
- Create: `claudebridge/dashboard_routes.py`
- Create: `tests/test_dashboard_routes.py`

**Step 1: Write the tests**

```python
"""Tests for dashboard routes."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from claudebridge.dashboard_routes import create_dashboard_router
from claudebridge.dashboard_state import DashboardState


@pytest.fixture
def state():
    return DashboardState()


@pytest.fixture
def app(state):
    app = FastAPI()
    router = create_dashboard_router(state, pool_status_fn=lambda: {"size": 3, "available": 2, "in_use": 1, "models": ["opus"]})
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.unit
class TestDashboardPage:

    def test_dashboard_returns_html(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "claudebridge" in resp.text

    def test_pool_endpoint_returns_html(self, client):
        resp = client.get("/dashboard/pool")
        assert resp.status_code == 200
        assert "pool-dot" in resp.text

    def test_recent_endpoint_returns_html(self, client):
        resp = client.get("/dashboard/recent")
        assert resp.status_code == 200


@pytest.mark.unit
class TestActiveSSE:

    def test_active_endpoint_returns_sse(self, client, state):
        state.request_started("req-1", "sonnet")
        # SSE endpoint returns event-stream
        with client.stream("GET", "/dashboard/active") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.unit
class TestRequestDetail:

    def test_active_request_detail(self, client, state):
        state.request_started("req-1", "sonnet")
        resp = client.get("/dashboard/request/req-1")
        assert resp.status_code == 200
        assert "req-1" in resp.text
        assert "Live Stream" in resp.text

    def test_unknown_request_404(self, client):
        resp = client.get("/dashboard/request/nonexistent")
        assert resp.status_code == 404
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_routes.py -v -m unit`
Expected: FAIL (module not found)

**Step 3: Implement dashboard_routes.py**

```python
"""Dashboard routes for htmx-based UI."""

import asyncio
import os
import re
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from .dashboard_state import DashboardState

# Templates directory
_TEMPLATE_DIR = Path(__file__).parent / "templates" / "dashboard"
_templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


def _parse_log_file(path: Path) -> dict | None:
    """Parse a session log file into a summary dict."""
    try:
        text = path.read_text()
    except Exception:
        return None

    request_id = path.stem
    model = ""
    duration_ms = ""
    error = None
    timestamp = ""
    acquire_ms = None
    query_ms = None
    messages = []
    response = ""

    for line in text.splitlines():
        if line.startswith("MODEL: "):
            model = line[7:]
        elif line.startswith("TIMESTAMP: "):
            timestamp = line[11:]
        elif line.startswith("Duration: "):
            duration_ms = line[10:].rstrip("ms")
        elif line.startswith("Acquire: "):
            acquire_ms = line[9:].rstrip("ms")
        elif line.startswith("Query: "):
            query_ms = line[7:].rstrip("ms")
        elif line.startswith("[") and "] ERROR: " in line:
            error = line.split("] ERROR: ", 1)[1]

    # Extract messages
    in_messages = False
    in_response = False
    response_lines = []
    for line in text.splitlines():
        if line == "Messages:":
            in_messages = True
            continue
        if line == "" and in_messages:
            in_messages = False
            continue
        if in_messages and line.startswith("["):
            match = re.match(r"\[(\w+)\] (.*)", line)
            if match:
                messages.append({"role": match.group(1), "content": match.group(2)})
        if line == "Full response:":
            in_response = True
            continue
        if in_response:
            if line.startswith("=" * 10):
                in_response = False
                continue
            response_lines.append(line)

    response = "\n".join(response_lines).strip()

    return {
        "request_id": request_id,
        "model": model,
        "timestamp": timestamp,
        "duration_ms": duration_ms,
        "acquire_ms": acquire_ms,
        "query_ms": query_ms,
        "error": error,
        "messages": messages,
        "response": response,
    }


def _get_recent_logs(limit: int = 20) -> list[dict]:
    """Read recent session log files, newest first."""
    log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
    if not log_dir.exists():
        return []
    files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files[:limit]:
        parsed = _parse_log_file(f)
        if parsed:
            results.append(parsed)
    return results


def create_dashboard_router(
    state: DashboardState,
    pool_status_fn: Callable[[], dict],
) -> APIRouter:
    """Create dashboard router with injected dependencies."""

    router = APIRouter()

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page(request: Request):
        return _templates.TemplateResponse(request, "page.html")

    @router.get("/dashboard/pool", response_class=HTMLResponse)
    async def dashboard_pool(request: Request):
        status = pool_status_fn()
        return _templates.TemplateResponse(request, "pool.html", {
            "size": status["size"],
            "available": status["available"],
            "in_use": status["in_use"],
        })

    @router.get("/dashboard/recent", response_class=HTMLResponse)
    async def dashboard_recent(request: Request):
        logs = _get_recent_logs()
        return _templates.TemplateResponse(request, "recent.html", {"logs": logs})

    @router.get("/dashboard/active")
    async def dashboard_active(request: Request):
        """SSE endpoint that pushes active request list updates."""
        async def event_stream():
            while True:
                active = state.get_active_requests()
                html = _templates.get_template("active.html").render(requests=active)
                yield f"data: {html}\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.get("/dashboard/request/{request_id}", response_class=HTMLResponse)
    async def dashboard_request_detail(request: Request, request_id: str):
        # Check if it's an active request
        active = state.get_active_requests()
        active_match = next((r for r in active if r["request_id"] == request_id), None)
        if active_match:
            return _templates.TemplateResponse(request, "detail.html", {
                "request_id": request_id,
                "model": active_match["model"],
                "timestamp": "",
                "duration_ms": f"{active_match['elapsed_s']}s (running)",
                "acquire_ms": None,
                "query_ms": None,
                "error": None,
                "is_active": True,
                "buffered_text": "",
                "messages": [],
                "response": "",
            })

        # Check log files
        log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
        log_path = log_dir / f"{request_id}.log"
        if not log_path.exists():
            raise HTTPException(status_code=404, detail="Request not found")

        parsed = _parse_log_file(log_path)
        if not parsed:
            raise HTTPException(status_code=404, detail="Could not parse log")

        return _templates.TemplateResponse(request, "detail.html", {
            **parsed,
            "is_active": False,
            "buffered_text": "",
        })

    @router.get("/dashboard/stream/{request_id}")
    async def dashboard_stream(request: Request, request_id: str):
        """SSE endpoint for live token streaming of an active request."""
        queue = state.subscribe(request_id)
        if queue is None:
            raise HTTPException(status_code=404, detail="Request not active")

        async def event_stream():
            try:
                while True:
                    msg = await queue.get()
                    if msg["type"] == "chunk":
                        text = msg["text"]
                        # Escape HTML
                        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        yield f"event: chunk\ndata: {text}\n\n"
                    elif msg["type"] == "done":
                        yield f"event: done\ndata: complete\n\n"
                        break
                    elif msg["type"] == "error":
                        yield f"event: error\ndata: {msg['error']}\n\n"
                        break
            finally:
                state.unsubscribe(request_id, queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_routes.py -v -m unit`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add claudebridge/dashboard_routes.py tests/test_dashboard_routes.py
git commit -m "feat(dashboard): add FastAPI routes with SSE streaming and log parsing"
```

---

### Task 6: Mount dashboard router in server.py

**Files:**
- Modify: `claudebridge/server.py:73-95` (lifespan and app setup)

**Step 1: Write the test**

Add to `tests/test_dashboard_integration.py`:

```python
@pytest.mark.unit
class TestDashboardMounted:
    """Verify dashboard routes are mounted on the app."""

    def test_dashboard_route_exists(self):
        from claudebridge.server import app
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/dashboard" in routes
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_integration.py::TestDashboardMounted -v -m unit`
Expected: FAIL

**Step 3: Mount the router**

In `server.py`, after the `dashboard_state = DashboardState()` line, add:

```python
from .dashboard_routes import create_dashboard_router
```

After `app = FastAPI(...)` (line ~95), add:

```python
# Mount dashboard
app.include_router(create_dashboard_router(
    dashboard_state,
    pool_status_fn=lambda: pool.status() if pool else {"size": 0, "available": 0, "in_use": 0, "models": []},
))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard_integration.py -v -m unit`
Expected: All PASS

**Step 5: Run all unit tests for regressions**

Run: `pytest tests/ -v -m unit`
Expected: All PASS

**Step 6: Commit**

```bash
git add claudebridge/server.py tests/test_dashboard_integration.py
git commit -m "feat(dashboard): mount dashboard router on main app"
```

---

### Task 7: Include templates in package build

**Files:**
- Modify: `pyproject.toml` (if needed)
- Modify: `MANIFEST.in` (if needed)

**Step 1: Verify templates are included**

Since we're using hatchling and the templates are inside the `claudebridge/` package directory, they should be included automatically. Verify:

Run: `python -c "from pathlib import Path; p = Path('claudebridge/templates/dashboard'); print(f'Templates exist: {p.exists()}'); print(list(p.glob('*.html')))"`
Expected: Shows all 6 template files

**Step 2: Ensure hatchling includes non-Python files**

Check if `pyproject.toml` needs a `[tool.hatch.build.targets.wheel]` `artifacts` entry. Hatchling includes all files in the package directory by default, including non-Python files. No change needed unless templates are excluded.

Run: `pip install -e . && python -c "import claudebridge; from pathlib import Path; p = Path(claudebridge.__file__).parent / 'templates' / 'dashboard'; print(list(p.glob('*.html')))"`
Expected: Lists all template files

**Step 3: Commit (if any changes needed)**

```bash
git add pyproject.toml
git commit -m "build: ensure dashboard templates included in wheel"
```

---

### Task 8: End-to-end verification

**Step 1: Start the server**

Run: `claudebridge`
Expected: Server starts, pool initializes

**Step 2: Open dashboard in browser**

Open: `http://localhost:8082/dashboard`
Expected: Dashboard page loads with Pico CSS styling, empty active/recent panels, pool status dots

**Step 3: Send a streaming request**

In another terminal:
```bash
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4", "messages": [{"role": "user", "content": "Count from 1 to 10 slowly"}], "stream": true}'
```

Expected in dashboard:
- Active Requests panel shows the request with model badge and ticking elapsed time
- Click on the request to see live tokens streaming in the detail panel
- After completion, request disappears from Active and appears in Recent
- Click on the completed request to see full conversation and response

**Step 4: Send a non-streaming request**

```bash
curl http://localhost:8082/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Expected: Request appears briefly in Active, then moves to Recent with timing info

**Step 5: Verify pool status**

Watch the pool status dots change color when a request is in flight (in-use vs available).

**Step 6: Final commit**

If any tweaks were needed during verification:
```bash
git add -A
git commit -m "fix(dashboard): polish from end-to-end testing"
```
