"""Lazy reusable Claude SDK client pool."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator, Callable

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeSDKClient

logger = logging.getLogger(__name__)

# Claude palette (24-bit true color)
_CLAUDE = "\033[38;2;218;119;86m"  # Terracotta — Claude's signature orange
_GREEN = "\033[32m"
_RESET = "\033[0m"

# Health check interval (seconds)
HEALTH_CHECK_INTERVAL = 60


def make_options(model: str, max_tokens: int | None = None):
    """Create ClaudeAgentOptions with model."""
    from claude_agent_sdk import ClaudeAgentOptions

    opts = ClaudeAgentOptions(
        max_turns=1,
        setting_sources=None,  # Don't load user filesystem settings
        system_prompt={"type": "preset", "preset": "claude_code"},
        model=model,
        env={"CLAUDE_CODE_BRIDGE": "1"},
        tools=[],  # No built-in tools - pure chat mode
    )
    if max_tokens is not None:
        opts.max_tokens = max_tokens
    return opts


class ClientPool:
    """Lazy reusable client pool with a hard concurrency limit.

    Clients are created on demand for the requested model and returned to the
    idle pool after successful use. The pool never creates model-specific
    clients at process startup.
    """

    def __init__(
        self,
        size: int = 3,
        default_model: str = "opus",
        on_change: Callable[[], None] | None = None,
    ):
        """Initialize client pool.

        Args:
            size: Maximum concurrent clients / idle slots.
            default_model: Fallback model for diagnostics.
            on_change: Optional callback invoked on every state mutation.
        """
        self.size = size
        self.default_model = default_model
        self._on_change = on_change
        self._client_models: dict[ClaudeSDKClient, str] = {}  # client -> model
        self._available: list[ClaudeSDKClient] = []  # idle reusable clients
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(size)  # limits concurrent usage
        self._initialized = False
        self._in_use = 0
        self._acquire_timeout = int(os.environ.get("CLAUDE_TIMEOUT", 120))
        self._health_check_task: asyncio.Task | None = None

    def _fire_change(self) -> None:
        """Invoke on_change callback if set."""
        if self._on_change is not None:
            self._on_change()

    async def initialize(self) -> None:
        """Mark the pool ready without creating any model-specific clients."""
        if self._initialized:
            return

        self._initialized = True
        logger.info(
            f"{_CLAUDE}[pool]{_RESET} {_GREEN}Ready{_RESET} | max={self.size} available=0"
        )
        self._fire_change()

        # Start periodic health check
        self._health_check_task = asyncio.create_task(self._periodic_health_check())

    async def _create_client(self, model: str) -> ClaudeSDKClient:
        """Create and connect a new client with specified model."""
        from claude_agent_sdk import ClaudeSDKClient

        client = ClaudeSDKClient(make_options(model))
        try:
            await asyncio.wait_for(client.connect(), timeout=30)
        except BaseException:
            try:
                await client.disconnect()
            except BaseException:
                pass
            raise
        self._client_models[client] = model
        return client

    async def _disconnect_client(self, client: ClaudeSDKClient) -> None:
        """Disconnect client and remove from tracking."""
        try:
            await client.disconnect()
        except BaseException:
            pass
        self._client_models.pop(client, None)

    def _log_tag(self, request_id: str | None = None) -> str:
        """Return log prefix with optional request ID."""
        if request_id:
            return f"[pool] [{request_id}]"
        return "[pool]"

    def _log_status(self, action: str, request_id: str | None = None) -> None:
        """Log current pool status with model breakdown."""
        available_models = [self._client_models[c] for c in self._available]
        available_str = f"[{', '.join(available_models)}]"
        tag = self._log_tag(request_id)

        logger.info(
            f"{tag} {action} | in_use={self._in_use} available={len(self._available)} "
            f"models={available_str}"
        )

    def status(self) -> dict:
        """Return current pool status metrics."""
        # Include in-use clients too
        all_models = list(self._client_models.values())
        return {
            "size": self.size,
            "available": len(self._available),
            "in_use": self._in_use,
            "models": all_models,
        }

    def snapshot(self) -> dict:
        """Return pool state snapshot for error diagnostics."""
        available_models = [self._client_models[c] for c in self._available]
        all_models = list(self._client_models.values())
        return {
            "size": self.size,
            "in_use": self._in_use,
            "available": len(self._available),
            "available_models": available_models,
            "all_models": all_models,
        }

    @asynccontextmanager
    async def acquire(
        self, model: str, request_id: str | None = None
    ) -> AsyncIterator[ClaudeSDKClient]:
        """Get a client for the specified model.

        Takes an idle client with matching model if available, otherwise creates
        a new one. If the pool is full of idle clients for other models, the
        oldest idle client is discarded to make room for the requested model.

        Args:
            model: Model name (opus, sonnet, haiku).
            request_id: Optional request ID for log correlation.

        Yields:
            A ClaudeSDKClient configured for the specified model.

        Raises:
            HTTPException: 503 if acquire times out waiting for a slot.
        """
        tag = self._log_tag(request_id)
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._acquire_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"{tag} Semaphore timeout after {self._acquire_timeout}s | pool={self.snapshot()}"
            )
            from fastapi import HTTPException

            raise HTTPException(
                status_code=503,
                detail=(
                    f"All pool clients busy. Timed out after {self._acquire_timeout}s "
                    "waiting for an available client."
                ),
            )

        try:
            client: ClaudeSDKClient | None = None
            old_to_discard: ClaudeSDKClient | None = None
            completed = False

            async with self._lock:
                # Take an idle client with matching model if available.
                matching = [
                    c for c in self._available if self._client_models[c] == model
                ]
                if matching:
                    client = matching[0]
                    self._available.remove(client)
                    logger.info(f"{tag} Reusing idle {model} client")
                elif len(self._client_models) >= self.size and self._available:
                    # Pool is full, but only with idle clients for other models.
                    old_to_discard = self._available.pop(0)

                self._in_use += 1
                self._log_status("Acquired", request_id)
                self._fire_change()

            # Disconnect non-matching idle client outside the lock.
            if old_to_discard:
                old_model = self._client_models.get(old_to_discard, "unknown")
                logger.info(
                    f"{tag} Discarding idle {old_model} client, need {model}"
                )
                try:
                    await self._disconnect_client(old_to_discard)
                except asyncio.CancelledError:
                    logger.warning(
                        f"{tag} Cancelled while discarding {old_model} client"
                    )
                    raise

            # Create fresh if no reusable client was available.
            if client is None:
                try:
                    client = await self._create_client(model)
                    logger.info(f"{tag} Created fresh {model} client")
                except asyncio.CancelledError:
                    logger.warning(f"{tag} Cancelled while creating {model} client")
                    raise

            yield client
            completed = True
        except Exception:
            raise
        finally:
            disconnect_client: ClaudeSDKClient | None = None
            returned = False
            async with self._lock:
                self._in_use -= 1
                if (
                    client is not None
                    and completed
                    and self._initialized
                    and len(self._available) + self._in_use < self.size
                ):
                    self._available.append(client)
                    returned = True
                else:
                    disconnect_client = client
                self._fire_change()
            if returned:
                logger.info(f"{tag} Returned {model} client to idle pool")
            elif disconnect_client is not None:
                status = "after cancellation" if not completed else "during shutdown"
                logger.info(f"{tag} Destroyed {model} client {status}")
                await self._disconnect_client(disconnect_client)
            self._semaphore.release()
            self._log_status("Released", request_id)

    async def _periodic_health_check(self) -> None:
        """Background task that checks idle client health every HEALTH_CHECK_INTERVAL seconds."""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            try:
                await self._check_idle_clients()
            except Exception as e:
                logger.warning(f"[pool] Health check error: {e}")

    async def _check_idle_clients(self) -> None:
        """Check each idle client's process is alive, removing dead clients."""
        async with self._lock:
            clients_to_check = list(self._available)

        for client in clients_to_check:
            try:
                # Check if the underlying process is still alive
                transport = getattr(client, "_transport", None)
                if transport is not None:
                    process = getattr(transport, "_process", None)
                    if process is not None and process.returncode is not None:
                        raise RuntimeError("Process exited")
            except Exception:
                model = self._client_models.get(client, self.default_model)
                logger.warning(f"[pool] Health check: {model} client unresponsive")
                removed = False
                async with self._lock:
                    if client in self._available:
                        self._available.remove(client)
                        removed = True
                if removed:
                    await self._disconnect_client(client)
                    self._fire_change()

    async def shutdown(self) -> None:
        """Disconnect all clients and clean up."""
        logger.info(f"[pool] Shutting down {len(self._client_models)} clients...")

        # Prevent in-flight requests from returning clients to the idle pool.
        self._initialized = False

        # Cancel health check task
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        # Drain in-flight requests before disconnecting clients
        drain_timeout = int(os.environ.get("SHUTDOWN_DRAIN_TIMEOUT", 30))
        deadline = time.monotonic() + drain_timeout
        while self._in_use > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    f"[pool] Shutdown drain timeout after {drain_timeout}s "
                    f"({self._in_use} request(s) still in flight — forcing cleanup)"
                )
                break
            await asyncio.sleep(0.1)

        for client in list(self._client_models.keys()):
            await self._disconnect_client(client)

        self._available.clear()
        self._in_use = 0
        self._fire_change()

        logger.info("[pool] Shutdown complete")
