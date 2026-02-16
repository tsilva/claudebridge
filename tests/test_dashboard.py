"""
Unit tests for dashboard state tracking and routes.

Usage:
- pytest tests/test_dashboard.py -v -m unit
"""

import asyncio
import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claudebridge.dashboard import (
    DashboardState,
    _ActiveRequest,
    _parse_log_file,
    _get_recent_logs,
    create_dashboard_router,
)


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestActiveRequest:
    """Tests for _ActiveRequest dataclass."""

    def test_to_dict_fields(self):
        """to_dict returns all expected keys."""
        req = _ActiveRequest("req-1", "sonnet")
        d = req.to_dict()
        assert d["request_id"] == "req-1"
        assert d["model"] == "sonnet"
        assert d["status"] == "active"
        assert d["chunks_received"] == 0
        assert isinstance(d["elapsed_s"], float)

    def test_elapsed_increases(self):
        """Elapsed time increases between calls."""
        import time
        req = _ActiveRequest("req-1", "sonnet")
        t1 = req.to_dict()["elapsed_s"]
        time.sleep(0.02)
        t2 = req.to_dict()["elapsed_s"]
        assert t2 > t1


@pytest.mark.unit
class TestRequestLifecycle:
    """Tests for start, complete, error lifecycle."""

    def test_start_adds_request(self):
        """request_started adds to active list."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        active = state.get_active_requests()
        assert len(active) == 1
        assert active[0]["request_id"] == "req-1"
        assert active[0]["model"] == "opus"

    def test_complete_removes_request(self):
        """request_completed removes from active list."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        state.request_completed("req-1")
        assert state.get_active_requests() == []

    def test_error_removes_request(self):
        """request_errored removes from active list."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        state.request_errored("req-1", "timeout")
        assert state.get_active_requests() == []

    def test_complete_unknown_request_no_error(self):
        """Completing an unknown request does not raise."""
        state = DashboardState()
        state.request_completed("nonexistent")  # should not raise

    def test_error_unknown_request_no_error(self):
        """Erroring an unknown request does not raise."""
        state = DashboardState()
        state.request_errored("nonexistent", "boom")  # should not raise


@pytest.mark.unit
class TestMultipleActiveRequests:
    """Tests for concurrent request tracking."""

    def test_multiple_active(self):
        """Multiple requests can be active simultaneously."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        state.request_started("req-2", "sonnet")
        state.request_started("req-3", "haiku")
        active = state.get_active_requests()
        assert len(active) == 3
        ids = {r["request_id"] for r in active}
        assert ids == {"req-1", "req-2", "req-3"}

    def test_completing_one_preserves_others(self):
        """Completing one request leaves others active."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        state.request_started("req-2", "sonnet")
        state.request_completed("req-1")
        active = state.get_active_requests()
        assert len(active) == 1
        assert active[0]["request_id"] == "req-2"


@pytest.mark.unit
class TestChunkCounting:
    """Tests for chunk counting."""

    def test_chunks_increment(self):
        """chunk_received increments the counter."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        state.chunk_received("req-1", "hello")
        state.chunk_received("req-1", " world")
        active = state.get_active_requests()
        assert active[0]["chunks_received"] == 2

    def test_chunk_unknown_request_no_error(self):
        """Sending a chunk to an unknown request does not raise."""
        state = DashboardState()
        state.chunk_received("nonexistent", "data")  # should not raise


@pytest.mark.unit
@pytest.mark.asyncio
class TestSubscriptionFanOut:
    """Tests for subscriber queue fan-out."""

    async def test_single_subscriber_gets_chunk(self):
        """A subscriber receives chunk messages."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        q = state.subscribe("req-1")
        assert q is not None

        state.chunk_received("req-1", "token")
        msg = q.get_nowait()
        assert msg == {"type": "chunk", "text": "token"}

    async def test_multiple_subscribers_get_same_chunks(self):
        """Multiple subscribers all receive the same chunk."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        q1 = state.subscribe("req-1")
        q2 = state.subscribe("req-1")

        state.chunk_received("req-1", "hello")

        msg1 = q1.get_nowait()
        msg2 = q2.get_nowait()
        assert msg1 == {"type": "chunk", "text": "hello"}
        assert msg2 == {"type": "chunk", "text": "hello"}

    async def test_done_signal_to_subscribers(self):
        """Subscribers receive done signal on completion."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        q = state.subscribe("req-1")

        state.request_completed("req-1")
        msg = q.get_nowait()
        assert msg == {"type": "done"}

    async def test_error_signal_to_subscribers(self):
        """Subscribers receive error signal on error."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        q = state.subscribe("req-1")

        state.request_errored("req-1", "connection lost")
        msg = q.get_nowait()
        assert msg == {"type": "error", "error": "connection lost"}

    async def test_subscribe_unknown_request_returns_none(self):
        """Subscribing to an unknown request returns None."""
        state = DashboardState()
        result = state.subscribe("nonexistent")
        assert result is None


@pytest.mark.unit
class TestUnsubscribe:
    """Tests for unsubscribe."""

    def test_unsubscribe_removes_queue(self):
        """Unsubscribed queue no longer receives chunks."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        q = state.subscribe("req-1")

        state.unsubscribe("req-1", q)
        state.chunk_received("req-1", "after unsub")

        assert q.empty()

    def test_unsubscribe_unknown_request_no_error(self):
        """Unsubscribing from an unknown request does not raise."""
        state = DashboardState()
        q = asyncio.Queue()
        state.unsubscribe("nonexistent", q)  # should not raise

    def test_unsubscribe_unknown_queue_no_error(self):
        """Unsubscribing a queue that was never subscribed does not raise."""
        state = DashboardState()
        state.request_started("req-1", "opus")
        q = asyncio.Queue()
        state.unsubscribe("req-1", q)  # should not raise


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

def _make_app(state=None, pool_status_fn=None):
    """Create a minimal FastAPI app with the dashboard router mounted."""
    if state is None:
        state = DashboardState()
    if pool_status_fn is None:
        pool_status_fn = lambda: {"size": 3, "available": 2, "in_use": 1}
    app = FastAPI()
    router = create_dashboard_router(state, pool_status_fn)
    app.include_router(router)
    return app


SAMPLE_LOG = {
    "request_id": "chatcmpl-abc123",
    "model": "sonnet",
    "api_key": None,
    "timestamp": "2026-02-14T10:00:00.000Z",
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ],
    "parameters": {"stream": False, "temperature": None, "max_tokens": None},
    "response": "Hi there",
    "finish_reason": "stop",
    "timing": {"acquire_ms": 45, "query_ms": 1050, "duration_ms": 1100},
    "usage": {"input_tokens": 100, "output_tokens": 50},
    "error": None,
    "attachments": [],
}

SAMPLE_LOG_WITH_ERROR = {
    "request_id": "chatcmpl-err456",
    "model": "opus",
    "api_key": None,
    "timestamp": "2026-02-14T10:05:00.000Z",
    "messages": [{"role": "user", "content": "Do something"}],
    "parameters": {"stream": True, "temperature": None, "max_tokens": None},
    "response": "",
    "finish_reason": None,
    "timing": {"duration_ms": 2000},
    "usage": {},
    "error": "Timeout after 120s",
    "attachments": [],
}


@pytest.mark.unit
class TestParseLogFile:
    """Tests for _parse_log_file."""

    def test_parses_valid_log(self, tmp_path):
        """Parses JSON log file with all expected fields."""
        log_path = tmp_path / "chatcmpl-abc123.json"
        log_path.write_text(json.dumps(SAMPLE_LOG))

        result = _parse_log_file(log_path)
        assert result is not None
        assert result["request_id"] == "chatcmpl-abc123"
        assert result["model"] == "sonnet"
        assert result["timestamp"] == "2026-02-14T10:00:00.000Z"
        assert result["timing"]["duration_ms"] == 1100
        assert result["timing"]["acquire_ms"] == 45
        assert result["timing"]["query_ms"] == 1050
        assert result["error"] is None
        assert len(result["messages"]) == 2
        assert result["response"] == "Hi there"

    def test_parses_error_log(self, tmp_path):
        """Parses log file with error."""
        log_path = tmp_path / "chatcmpl-err456.json"
        log_path.write_text(json.dumps(SAMPLE_LOG_WITH_ERROR))

        result = _parse_log_file(log_path)
        assert result is not None
        assert result["request_id"] == "chatcmpl-err456"
        assert result["error"] == "Timeout after 120s"

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None if file does not exist."""
        result = _parse_log_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_for_garbage(self, tmp_path):
        """Returns None for a file with invalid JSON."""
        bad = tmp_path / "bad.json"
        bad.write_text("just some random text\nnothing useful here\n")
        result = _parse_log_file(bad)
        assert result is None


@pytest.mark.unit
class TestGetRecentLogs:
    """Tests for _get_recent_logs."""

    def test_returns_empty_for_missing_dir(self, tmp_path, monkeypatch):
        """Returns empty list if log directory does not exist."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "no_such_dir"))
        result = _get_recent_logs()
        assert result == []

    def test_returns_parsed_logs_newest_first(self, tmp_path, monkeypatch):
        """Returns logs sorted by mtime, newest first."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path))

        older = {**SAMPLE_LOG, "request_id": "chatcmpl-older"}
        newer = {**SAMPLE_LOG, "request_id": "chatcmpl-newer"}

        p1 = tmp_path / "chatcmpl-older.json"
        p1.write_text(json.dumps(older))

        p2 = tmp_path / "chatcmpl-newer.json"
        p2.write_text(json.dumps(newer))

        # Ensure different mtimes
        os.utime(p1, (1000000, 1000000))
        os.utime(p2, (2000000, 2000000))

        result = _get_recent_logs(limit=10)
        assert len(result) == 2
        assert result[0]["request_id"] == "chatcmpl-newer"
        assert result[1]["request_id"] == "chatcmpl-older"

    def test_respects_limit(self, tmp_path, monkeypatch):
        """Respects the limit parameter."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path))

        for i in range(5):
            log = {**SAMPLE_LOG, "request_id": f"chatcmpl-{i:03d}"}
            p = tmp_path / f"chatcmpl-{i:03d}.json"
            p.write_text(json.dumps(log))

        result = _get_recent_logs(limit=2)
        assert len(result) == 2


@pytest.mark.unit
class TestDashboardPage:
    """Tests for GET /dashboard."""

    def test_returns_html_with_claudebridge(self):
        """Dashboard page returns HTML containing 'claudebridge'."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "claudebridge" in resp.text


@pytest.mark.unit
class TestDashboardPool:
    """Tests for GET /dashboard/pool."""

    def test_returns_html_with_pool_dot(self):
        """Pool endpoint returns HTML with 'pool-dot'."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/dashboard/pool")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "pool-dot" in resp.text


@pytest.mark.unit
class TestDashboardRequests:
    """Tests for GET /dashboard/requests."""

    def test_returns_sse_content_type(self):
        """Requests endpoint returns SSE content-type."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from fastapi.responses import StreamingResponse

        state = DashboardState()
        pool_fn = lambda: {"size": 3, "available": 2, "in_use": 1}
        router = create_dashboard_router(state, pool_fn)

        handler = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/dashboard/requests":
                handler = route.endpoint
                break

        assert handler is not None, "Could not find /dashboard/requests route"

        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(handler(mock_request))
        finally:
            loop.close()
        assert isinstance(resp, StreamingResponse)
        assert resp.media_type == "text/event-stream"

    def test_sse_contains_active_and_completed(self, tmp_path, monkeypatch):
        """SSE stream renders both active and completed requests."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from fastapi.responses import StreamingResponse

        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        log = {**SAMPLE_LOG, "request_id": "chatcmpl-done1"}
        log_path = tmp_path / "chatcmpl-done1.json"
        log_path.write_text(json.dumps(log))

        state = DashboardState()
        state.request_started("chatcmpl-live1", "sonnet")
        pool_fn = lambda: {"size": 3, "available": 2, "in_use": 1}
        router = create_dashboard_router(state, pool_fn)

        handler = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/dashboard/requests":
                handler = route.endpoint
                break

        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(side_effect=[False, True])

        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(handler(mock_request))
            chunks = []
            async def collect():
                async for chunk in resp.body_iterator:
                    chunks.append(chunk)
                    break
            loop.run_until_complete(collect())
        finally:
            loop.close()

        content = "".join(chunks)
        assert "chatcmpl-live1" in content or "live1" in content
        assert "chatcmpl-done1" in content or "done1" in content
        assert "request-row-active" in content


@pytest.mark.unit
class TestDashboardRequestDetail:
    """Tests for GET /dashboard/request/{request_id}."""

    def test_active_request_shows_live_stream(self):
        """Active request detail shows 'Live Stream'."""
        state = DashboardState()
        state.request_started("chatcmpl-live1", "sonnet")
        app = _make_app(state=state)
        client = TestClient(app)
        resp = client.get("/dashboard/request/chatcmpl-live1")
        assert resp.status_code == 200
        assert "Live Stream" in resp.text

    def test_log_file_request_shows_detail(self, tmp_path, monkeypatch):
        """Completed request detail is rendered from log file."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        log_path = tmp_path / "chatcmpl-abc123.json"
        log_path.write_text(json.dumps(SAMPLE_LOG))

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/dashboard/request/chatcmpl-abc123")
        assert resp.status_code == 200
        assert "chatcmpl-abc123" in resp.text

    def test_unknown_request_returns_404(self, tmp_path, monkeypatch):
        """Unknown request returns 404."""
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/dashboard/request/chatcmpl-nonexistent")
        assert resp.status_code == 404


@pytest.mark.unit
class TestDashboardStream:
    """Tests for GET /dashboard/stream/{request_id}."""

    def test_stream_inactive_request_returns_404(self):
        """Streaming an inactive request returns 404."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/dashboard/stream/chatcmpl-nope")
        assert resp.status_code == 404
