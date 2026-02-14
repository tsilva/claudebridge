"""
Unit tests for dashboard routes.

Usage:
- pytest tests/test_dashboard_routes.py -v -m unit
"""

import os
import textwrap
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claudebridge.dashboard_routes import (
    _parse_log_file,
    _get_recent_logs,
    create_dashboard_router,
)
from claudebridge.dashboard_state import DashboardState


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


SAMPLE_LOG = textwrap.dedent("""\
    ================================================================================
    SESSION: chatcmpl-abc123
    TIMESTAMP: 2026-02-14T10:00:00.000Z
    MODEL: sonnet
    ================================================================================

    --- REQUEST ---
    Messages:
    [user] Hello
    [assistant] Hi there

    Parameters:
      stream: False
      temperature: None
      max_tokens: None

    --- RESPONSE ---
    [10:00:01.000] RESPONSE: 8 chars
    [10:00:01.100] FINISH: stop

    --- TIMING ---
    Start: 10:00:00.000
    End: 10:00:01.100
    Duration: 1100ms
    Acquire: 45ms
    Query: 1050ms

    --- COMPLETE ---
    Full response:
    Hi there
    ================================================================================
""")


SAMPLE_LOG_WITH_ERROR = textwrap.dedent("""\
    ================================================================================
    SESSION: chatcmpl-err456
    TIMESTAMP: 2026-02-14T10:05:00.000Z
    MODEL: opus
    ================================================================================

    --- REQUEST ---
    Messages:
    [user] Do something

    Parameters:
      stream: True
      temperature: None
      max_tokens: None

    --- RESPONSE ---
    [10:05:02.000] ERROR: Timeout after 120s

    --- TIMING ---
    Start: 10:05:00.000
    End: 10:05:02.000
    Duration: 2000ms

    --- COMPLETE ---
    Full response:

    ================================================================================
""")


@pytest.mark.unit
class TestParseLogFile:
    """Tests for _parse_log_file."""

    def test_parses_valid_log(self, tmp_path):
        """Parses session, model, timestamp, duration, acquire, query, messages, response."""
        log_path = tmp_path / "chatcmpl-abc123.log"
        log_path.write_text(SAMPLE_LOG)

        result = _parse_log_file(log_path)
        assert result is not None
        assert result["request_id"] == "chatcmpl-abc123"
        assert result["model"] == "sonnet"
        assert result["timestamp"] == "2026-02-14T10:00:00.000Z"
        assert result["duration_ms"] == 1100
        assert result["acquire_ms"] == 45
        assert result["query_ms"] == 1050
        assert result["error"] is None
        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "user", "content": "Hello"}
        assert result["messages"][1] == {"role": "assistant", "content": "Hi there"}
        assert result["response"] == "Hi there"

    def test_parses_error_log(self, tmp_path):
        """Parses log file with error."""
        log_path = tmp_path / "chatcmpl-err456.log"
        log_path.write_text(SAMPLE_LOG_WITH_ERROR)

        result = _parse_log_file(log_path)
        assert result is not None
        assert result["request_id"] == "chatcmpl-err456"
        assert result["error"] == "Timeout after 120s"
        assert result["duration_ms"] == 2000
        assert result["acquire_ms"] is None
        assert result["query_ms"] is None

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None if file does not exist."""
        result = _parse_log_file(tmp_path / "nonexistent.log")
        assert result is None

    def test_returns_none_for_garbage(self, tmp_path):
        """Returns None for a file with no SESSION line."""
        bad = tmp_path / "bad.log"
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

        # Write two log files with different mtimes
        p1 = tmp_path / "chatcmpl-older.log"
        p1.write_text(SAMPLE_LOG.replace("chatcmpl-abc123", "chatcmpl-older"))

        p2 = tmp_path / "chatcmpl-newer.log"
        p2.write_text(SAMPLE_LOG.replace("chatcmpl-abc123", "chatcmpl-newer"))

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
            p = tmp_path / f"chatcmpl-{i:03d}.log"
            p.write_text(SAMPLE_LOG.replace("chatcmpl-abc123", f"chatcmpl-{i:03d}"))

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
        """Requests endpoint returns SSE content-type.

        The SSE endpoint is an infinite streaming generator so we cannot consume it
        with TestClient.  Instead we invoke the route handler directly and verify
        the returned StreamingResponse has the correct media_type.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from fastapi.responses import StreamingResponse

        state = DashboardState()
        pool_fn = lambda: {"size": 3, "available": 2, "in_use": 1}
        router = create_dashboard_router(state, pool_fn)

        # Find the route handler for /dashboard/requests
        handler = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/dashboard/requests":
                handler = route.endpoint
                break

        assert handler is not None, "Could not find /dashboard/requests route"

        # Create a mock Request with is_disconnected
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        # Call the handler and verify it returns a StreamingResponse
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
        log_path = tmp_path / "chatcmpl-done1.log"
        log_path.write_text(SAMPLE_LOG.replace("chatcmpl-abc123", "chatcmpl-done1"))

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
        # Return False first (render one frame), then True (disconnect)
        mock_request.is_disconnected = AsyncMock(side_effect=[False, True])

        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(handler(mock_request))
            # Consume the first SSE event
            chunks = []
            async def collect():
                async for chunk in resp.body_iterator:
                    chunks.append(chunk)
                    break  # just first event
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
        log_path = tmp_path / "chatcmpl-abc123.log"
        log_path.write_text(SAMPLE_LOG)

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
