"""
Unit tests for dashboard in-memory state tracking.

Usage:
- pytest tests/test_dashboard_state.py -v -m unit
"""

import asyncio

import pytest

from claudebridge.dashboard_state import DashboardState, _ActiveRequest


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
