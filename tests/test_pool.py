"""
Unit tests for single-use client pool.

These tests mock the Claude SDK to test pool logic in isolation.

Usage:
- pytest tests/test_pool.py -v
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from claudebridge.pool import ClientPool, make_options


def _make_mock_client():
    """Create a mock ClaudeSDKClient."""
    return AsyncMock()


@pytest.mark.unit
class TestMakeOptions:
    """Tests for make_options function."""

    def test_make_options_opus(self):
        """Options for opus model."""
        opts = make_options("opus")
        assert opts.model == "opus"
        assert opts.max_turns == 1
        assert opts.setting_sources is None
        assert opts.tools == []

    def test_make_options_sonnet(self):
        """Options for sonnet model."""
        opts = make_options("sonnet")
        assert opts.model == "sonnet"

    def test_make_options_haiku(self):
        """Options for haiku model."""
        opts = make_options("haiku")
        assert opts.model == "haiku"

    def test_make_options_system_prompt(self):
        """Options include Claude Code preset system prompt."""
        opts = make_options("opus")
        assert opts.system_prompt["type"] == "preset"
        assert opts.system_prompt["preset"] == "claude_code"

    def test_make_options_env_variable(self):
        """Options include bridge environment variable."""
        opts = make_options("opus")
        assert opts.env["CLAUDE_CODE_BRIDGE"] == "1"

    def test_make_options_with_max_tokens(self):
        """Options with max_tokens set."""
        opts = make_options("opus", max_tokens=1024)
        assert opts.max_tokens == 1024

    def test_make_options_without_max_tokens(self):
        """Options without max_tokens leaves it unset."""
        opts = make_options("opus")
        # Should not have max_tokens set (or default)
        assert not hasattr(opts, 'max_tokens') or opts.max_tokens is None


@pytest.mark.unit
class TestClientPoolInit:
    """Tests for ClientPool initialization."""

    def test_default_pool_size(self):
        """Default pool size is 3."""
        pool = ClientPool()
        assert pool.size == 3

    def test_custom_pool_size(self):
        """Custom pool size."""
        pool = ClientPool(size=5)
        assert pool.size == 5

    def test_default_model(self):
        """Default model is opus."""
        pool = ClientPool()
        assert pool.default_model == "opus"

    def test_custom_default_model(self):
        """Custom default model."""
        pool = ClientPool(default_model="sonnet")
        assert pool.default_model == "sonnet"

    def test_initial_state(self):
        """Initial state is empty."""
        pool = ClientPool(size=2)
        assert len(pool._available) == 0
        assert len(pool._client_models) == 0
        assert pool._initialized is False
        assert pool._in_use == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolInitialize:
    """Tests for pool initialization."""

    async def test_initialize_creates_clients(self):
        """Initialize creates correct number of clients."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await pool.initialize()

            assert MockClient.call_count == 2
            assert mock_instance.connect.call_count == 2
            assert len(pool._available) == 2
            assert pool._initialized is True

            # Cleanup health check task
            await pool.shutdown()

    async def test_initialize_sets_model(self):
        """Initialize sets model for all clients."""
        pool = ClientPool(size=2, default_model="sonnet")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await pool.initialize()

            for client in pool._client_models:
                assert pool._client_models[client] == "sonnet"

            await pool.shutdown()

    async def test_double_initialize_no_op(self):
        """Double initialization is a no-op."""
        pool = ClientPool(size=2)

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await pool.initialize()
            await pool.initialize()  # Second call

            # Should only create clients once
            assert MockClient.call_count == 2

            await pool.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolAcquire:
    """Tests for pool acquire functionality."""

    async def test_acquire_matching_model(self):
        """Acquire returns a pre-warmed client for matching model."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            # init client, then prewarm client
            mock_prewarm = _make_mock_client()
            MockClient.side_effect = [mock_client, mock_prewarm]

            await pool.initialize()

            async with pool.acquire("opus") as client:
                assert client is mock_client

            # Allow background tasks to complete
            await asyncio.sleep(0.01)

            await pool.shutdown()

    async def test_acquire_different_model_discards_prewarmed(self):
        """Acquire with different model discards pre-warmed client."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_opus = _make_mock_client()
            mock_sonnet = _make_mock_client()
            mock_prewarm = _make_mock_client()
            MockClient.side_effect = [mock_opus, mock_sonnet, mock_prewarm]

            await pool.initialize()

            async with pool.acquire("sonnet") as client:
                assert client is mock_sonnet

            # Allow background tasks (disconnect old, prewarm new)
            await asyncio.sleep(0.01)
            mock_opus.disconnect.assert_called_once()

            await pool.shutdown()

    async def test_acquire_destroys_after_use(self):
        """Client is destroyed (not returned) after use."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            mock_prewarm = _make_mock_client()
            MockClient.side_effect = [mock_client, mock_prewarm]

            await pool.initialize()

            async with pool.acquire("opus") as client:
                assert client is mock_client

            # Allow background tasks to complete
            await asyncio.sleep(0.01)

            # Client should have been disconnected, NOT returned to pool
            mock_client.disconnect.assert_called_once()

            await pool.shutdown()

    async def test_acquire_triggers_prewarm(self):
        """Acquire triggers background pre-warm after use."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_clients = [_make_mock_client() for _ in range(4)]
            MockClient.side_effect = mock_clients

            await pool.initialize()

            async with pool.acquire("opus"):
                pass

            # Allow background tasks to complete
            await asyncio.sleep(0.01)

            # A pre-warm client should have been created (3rd client after the 2 init ones)
            assert MockClient.call_count >= 3

            await pool.shutdown()

    async def test_sequential_requests_get_different_clients(self):
        """Sequential requests get different client instances (no reuse)."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            clients = [_make_mock_client() for _ in range(4)]
            MockClient.side_effect = clients

            await pool.initialize()

            acquired_clients = []

            async with pool.acquire("opus") as c1:
                acquired_clients.append(c1)

            # Allow prewarm to complete
            await asyncio.sleep(0.01)

            async with pool.acquire("opus") as c2:
                acquired_clients.append(c2)

            # The two clients must be different objects
            assert acquired_clients[0] is not acquired_clients[1]
            # First client must have been disconnected
            acquired_clients[0].disconnect.assert_called_once()

            await pool.shutdown()

    async def test_acquire_concurrent_limit(self):
        """Concurrent acquisitions limited by pool size."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            MockClient.return_value = _make_mock_client()

            await pool.initialize()

            concurrent = 0
            max_concurrent = 0
            lock = asyncio.Lock()

            async def use_client(delay: float):
                nonlocal concurrent, max_concurrent
                async with pool.acquire("opus"):
                    async with lock:
                        concurrent += 1
                        max_concurrent = max(max_concurrent, concurrent)
                    await asyncio.sleep(delay)
                    async with lock:
                        concurrent -= 1

            tasks = [
                asyncio.create_task(use_client(0.05)),
                asyncio.create_task(use_client(0.05)),
                asyncio.create_task(use_client(0.05)),
            ]
            await asyncio.gather(*tasks)

            assert max_concurrent <= 2

            await pool.shutdown()

    async def test_acquire_semaphore_timeout(self):
        """Acquire raises 503 when semaphore times out."""
        pool = ClientPool(size=1, default_model="opus")
        pool._acquire_timeout = 0.1  # Very short timeout

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            # Hold the only slot
            async with pool.acquire("opus"):
                from fastapi import HTTPException
                with pytest.raises(HTTPException) as exc_info:
                    async with pool.acquire("opus"):
                        pass
                assert exc_info.value.status_code == 503
                assert "busy" in str(exc_info.value.detail).lower()

            await pool.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolPrewarm:
    """Tests for pre-warming logic."""

    async def test_prewarm_fills_pool(self):
        """Pre-warm adds client to available list."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            # Pool has 2 available. Use one (leaves 1), which triggers prewarm.
            async with pool.acquire("opus"):
                pass

            # Allow prewarm to complete
            await asyncio.sleep(0.01)

            # Pool should be restored (prewarm added a client)
            # Note: the used client is destroyed, so available count
            # depends on prewarm filling the slot
            assert len(pool._available) >= 1

            await pool.shutdown()

    async def test_prewarm_respects_pool_size(self):
        """Pre-warm doesn't grow pool beyond size."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            MockClient.return_value = _make_mock_client()

            # Manually add a client
            client = await pool._create_client("opus")
            pool._available.append(client)

            # Now prewarm — should not add because pool is full
            await pool._prewarm_client("opus")

            assert len(pool._available) == 1

            await pool.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolShutdown:
    """Tests for pool shutdown."""

    async def test_shutdown_disconnects_all(self):
        """Shutdown disconnects all clients."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_clients = [_make_mock_client() for _ in range(2)]
            MockClient.side_effect = mock_clients

            await pool.initialize()
            await pool.shutdown()

            for mock_client in mock_clients:
                mock_client.disconnect.assert_called_once()

            assert len(pool._available) == 0
            assert len(pool._client_models) == 0
            assert pool._initialized is False

    async def test_shutdown_clears_state(self):
        """Shutdown clears all pool state."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            mock_prewarm = _make_mock_client()
            MockClient.side_effect = [mock_client, mock_prewarm]

            await pool.initialize()

            async with pool.acquire("opus"):
                pass

            await asyncio.sleep(0.01)
            await pool.shutdown()

            assert pool._in_use == 0
            assert pool._initialized is False

    async def test_shutdown_cancels_health_check(self):
        """Shutdown cancels the periodic health check task."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()
            assert pool._health_check_task is not None

            await pool.shutdown()
            assert pool._health_check_task is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolModelTracking:
    """Tests for model tracking in pool."""

    async def test_model_tracked_per_client(self):
        """Each client tracks its model."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_clients = [_make_mock_client() for _ in range(2)]
            MockClient.side_effect = mock_clients

            await pool.initialize()

            for client in pool._available:
                assert pool._client_models[client] == "opus"

            await pool.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolStatus:
    """Tests for pool status method."""

    async def test_status_returns_metrics(self):
        """Status returns pool size, available, in_use, and models."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_clients = [_make_mock_client() for _ in range(2)]
            MockClient.side_effect = mock_clients

            await pool.initialize()

            status = pool.status()
            assert status["size"] == 2
            assert status["available"] == 2
            assert status["in_use"] == 0
            assert len(status["models"]) == 2

            await pool.shutdown()

    async def test_status_during_acquire(self):
        """Status reflects in-use clients during acquire."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_clients = [_make_mock_client() for _ in range(3)]
            MockClient.side_effect = mock_clients

            await pool.initialize()

            async with pool.acquire("opus"):
                status = pool.status()
                assert status["in_use"] == 1
                assert status["available"] == 1

            await pool.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolSnapshot:
    """Tests for pool snapshot method."""

    async def test_snapshot_returns_state(self):
        """Snapshot returns size, in_use, available, available_models, all_models."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_clients = [_make_mock_client() for _ in range(3)]
            MockClient.side_effect = mock_clients

            await pool.initialize()

            async with pool.acquire("opus"):
                snap = pool.snapshot()
                assert snap["size"] == 2
                assert snap["in_use"] == 1
                assert snap["available"] == 1
                assert isinstance(snap["available_models"], list)
                assert isinstance(snap["all_models"], list)
                assert len(snap["all_models"]) >= 1

            await pool.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolRequestIdLogging:
    """Tests for request ID in pool logs."""

    async def test_acquire_logs_request_id(self, caplog):
        """Acquire logs include request ID when provided."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            mock_prewarm = _make_mock_client()
            MockClient.side_effect = [mock_client, mock_prewarm]

            await pool.initialize()

            with caplog.at_level("INFO", logger="claudebridge.pool"):
                async with pool.acquire("opus", request_id="chatcmpl-test123"):
                    pass

            # Allow background tasks to complete
            await asyncio.sleep(0.01)

            # Check that request ID appears in log messages
            request_id_logs = [r for r in caplog.records if "chatcmpl-test123" in r.message]
            assert len(request_id_logs) >= 1

            await pool.shutdown()
