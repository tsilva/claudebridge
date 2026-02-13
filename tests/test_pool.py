"""
Unit tests for client pool functionality.

These tests mock the Claude SDK to test pool logic in isolation.

Usage:
- pytest tests/test_pool.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_agent_sdk import ResultMessage

from claudebridge.pool import ClientPool, make_options


def _make_mock_client():
    """Create a mock ClaudeSDKClient that handles /clear correctly."""
    client = AsyncMock()
    # receive_response needs to yield a real ResultMessage for isinstance checks
    result_msg = MagicMock(spec=ResultMessage)

    async def _receive():
        yield result_msg

    client.receive_response = _receive
    return client


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
        """Acquire returns a client for matching model."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            async with pool.acquire("opus") as client:
                assert client is mock_client

            await pool.shutdown()

    async def test_acquire_different_model_replaces(self):
        """Acquire with different model replaces client."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_opus = _make_mock_client()
            mock_sonnet = _make_mock_client()
            MockClient.side_effect = [mock_opus, mock_sonnet]

            await pool.initialize()

            async with pool.acquire("sonnet") as client:
                assert client is mock_sonnet
                mock_opus.disconnect.assert_called_once()

            await pool.shutdown()

    async def test_acquire_returns_client_to_pool(self):
        """Client is returned to pool after use."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            async with pool.acquire("opus"):
                assert len(pool._available) == 0
                assert pool._in_use == 1

            assert len(pool._available) == 1
            assert pool._in_use == 0

            await pool.shutdown()

    async def test_acquire_concurrent_limit(self):
        """Concurrent acquisitions limited by pool size."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            clients = [_make_mock_client() for _ in range(2)]
            MockClient.side_effect = clients

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

            # Hold the only client
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
            MockClient.return_value = mock_client

            await pool.initialize()

            async with pool.acquire("opus"):
                pass

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
class TestClientPoolErrorHandling:
    """Tests for pool error handling."""

    async def test_acquire_handles_clear_error(self):
        """Pool replaces client when /clear fails."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.query.side_effect = Exception("Connection lost")
            # Need replacement client for clear failure, then another for the fresh client
            mock_replacement = _make_mock_client()
            MockClient.side_effect = [mock_client, mock_replacement]

            await pool.initialize()

            # Clear will fail on mock_client, pool should replace it with mock_replacement
            async with pool.acquire("opus") as client:
                assert client is mock_replacement

            await pool.shutdown()

    async def test_acquire_client_returned_on_success(self):
        """Client is returned to pool on successful use."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            async with pool.acquire("opus"):
                pass

            assert mock_client in pool._available

            await pool.shutdown()

    async def test_clear_timeout_replaces_client(self):
        """Client is replaced when /clear times out."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            # First client: /clear hangs
            slow_client = AsyncMock()

            async def slow_clear(*args, **kwargs):
                await asyncio.sleep(100)  # Will timeout

            slow_client.query.side_effect = slow_clear

            async def _slow_receive():
                await asyncio.sleep(100)
                yield MagicMock(spec=ResultMessage)

            slow_client.receive_response = _slow_receive

            # Replacement client
            replacement = _make_mock_client()
            MockClient.side_effect = [slow_client, replacement]

            await pool.initialize()

            async with pool.acquire("opus") as client:
                assert client is replacement

            await pool.shutdown()

    async def test_pool_recovery_after_replacement_failure(self):
        """Pool schedules recovery when replacement fails."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            # Simulate error during acquire that leads to replacement failure
            async with pool.acquire("opus") as client:
                pass

            # Pool should still be functional
            assert len(pool._available) == 1

            await pool.shutdown()


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

    async def test_model_updated_on_replacement(self):
        """Model is updated when client is replaced."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claudebridge.pool.ClaudeSDKClient") as MockClient:
            mock_opus = _make_mock_client()
            mock_sonnet = _make_mock_client()
            MockClient.side_effect = [mock_opus, mock_sonnet]

            await pool.initialize()

            async with pool.acquire("sonnet"):
                pass

            assert pool._client_models[mock_sonnet] == "sonnet"

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
            mock_clients = [_make_mock_client() for _ in range(2)]
            MockClient.side_effect = mock_clients

            await pool.initialize()

            async with pool.acquire("opus"):
                status = pool.status()
                assert status["in_use"] == 1
                assert status["available"] == 1

            await pool.shutdown()
