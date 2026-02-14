# Dashboard Design

## Context

claudebridge users need visibility into what the server is doing - active requests, pool state, streaming progress, and request history with full conversation logs. This is a single-user local debugging tool, not production monitoring.

## Approach

htmx + Jinja2 server-rendered dashboard at `/dashboard`. No JS to write, no build step. SSE for live token streaming. Pico CSS from CDN for styling.

## Data Layer

**New file: `dashboard_state.py` (~40 lines)**

- Tracks active requests in-memory: `request_id -> {model, start_time, status, chunks_received, token_queue}`
- Entries added on request start, removed shortly after completion
- Each active streaming request gets an `asyncio.Queue` for token fan-out to dashboard viewers
- Pool status: reuse existing `pool.status()` - no new code
- Completed request history: read from existing session log files on disk

**Integration (minimal changes to `server.py`):**
- `stream_claude_sdk()`: call `state.request_started()`, `state.chunk_received()`, `state.request_completed()`
- `call_claude_sdk()`: same start/complete hooks

## Layout

```
+---------------------------------------------+
|  claudebridge dashboard          [pool: 2/3]|
+----------------------+----------------------+
|  Active Requests     |  Request Detail      |
|  - req-abc  sonnet   |                      |
|  - req-def  opus     |  (click a request    |
|                      |   to see full        |
|  Recent Requests     |   conversation +     |
|  - req-123 ok  2.3s  |   live tokens)       |
|  - req-456 err       |                      |
|  - req-789 ok  1.1s  |                      |
+----------------------+----------------------+
|  Pool: **o  (2 available, 1 in use)         |
+---------------------------------------------+
```

- **Header:** server name + pool badge (htmx poll)
- **Left column:** active requests (SSE) + recent requests (htmx poll every 5s from log files)
- **Right column:** request detail - full conversation for completed, live token stream for active
- **Footer:** pool utilization bar

## File Structure

```
claudebridge/
  dashboard_state.py      # ~40 lines - active request tracking + token queues
  dashboard_routes.py     # ~80 lines - FastAPI routes
  templates/dashboard/
    page.html             # Base page layout, loads htmx + Pico CSS from CDN
    active.html           # Partial: active requests list
    recent.html           # Partial: recent completed requests
    detail.html           # Partial: full request detail view
    pool.html             # Partial: pool status bar
    stream.html           # Partial: live token stream fragment
```

2 Python files + 6 small templates.

## Routes

| Route | Purpose | Update mechanism |
|---|---|---|
| `GET /dashboard` | Full page | Initial load |
| `GET /dashboard/active` | Active requests list | SSE |
| `GET /dashboard/recent` | Recent completed requests | htmx poll 5s |
| `GET /dashboard/request/{id}` | Request detail (from log or live state) | htmx click |
| `GET /dashboard/stream/{id}` | Live token stream for active request | SSE |
| `GET /dashboard/pool` | Pool status | htmx poll 3s |

## Live Token Streaming

1. `stream_claude_sdk()` pushes chunks to `dashboard_state.chunk_received(request_id, text)`
2. Chunk goes into `asyncio.Queue` on the active request
3. `GET /dashboard/stream/{id}` reads from queue, sends as SSE events
4. htmx `sse-connect` appends each chunk to a `<pre>` block in real-time
5. On completion, final SSE event triggers swap to full completed detail view

**Edge cases:**
- Request finishes before dashboard connects: show completed view from log file
- Multiple dashboard tabs: each gets own queue copy (fan-out)
- Mid-stream error: SSE sends error event, detail shows error info

## Dependencies

- `jinja2` - template engine (FastAPI optional dep, ~1MB)
- Pico CSS - loaded from CDN, no install
- htmx - loaded from CDN, no install

## Verification

1. Start server: `claudebridge`
2. Open `http://localhost:8082/dashboard`
3. Send a request via curl/client - verify it appears in active requests
4. Watch tokens stream live in detail panel
5. After completion, verify request moves to recent list
6. Click completed request - verify full conversation is shown
7. Verify pool status updates when requests are in flight
