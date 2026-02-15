"""Single-use client pool with pre-warming for zero cross-contamination."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

logger = logging.getLogger(__name__)

# ANSI colors
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

# Health check interval (seconds)
HEALTH_CHECK_INTERVAL = 60


def make_options(model: str, max_tokens: int | None = None) -> ClaudeAgentOptions:
    """Create ClaudeAgentOptions with model."""
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
    """Single-use client pool with background pre-warming.

    Each request gets a fresh (or pre-warmed) client. Clients are NEVER reused —
    they are destroyed after each request. Background pre-warming hides creation latency.
    """

    def __init__(self, size: int = 3, default_model: str = "opus", on_change: Callable[[], None] | None = None):
        """Initialize client pool.

        Args:
            size: Maximum concurrent clients / pre-warm slots.
            default_model: Model to use for initial pool population.
            on_change: Optional callback invoked on every state mutation.
        """
        self.size = size
        self.default_model = default_model
        self._on_change = on_change
        self._client_models: dict[ClaudeSDKClient, str] = {}  # client -> model
        self._available: list[ClaudeSDKClient] = []  # pre-warmed clients
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(size)  # limits concurrent usage
        self._initialized = False
        self._in_use = 0
        self._acquire_timeout = int(os.environ.get("CLAUDE_TIMEOUT", 120))
        self._health_check_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

    def _fire_change(self) -> None:
        """Invoke on_change callback if set."""
        if self._on_change is not None:
            self._on_change()

    async def initialize(self) -> None:
        """Pre-spawn all clients with default model."""
        if self._initialized:
            return

        label = "client" if self.size == 1 else "clients"
        logger.info(f"{_CYAN}[pool]{_RESET} Warming {self.size} {self.default_model} {label}...")
        for i in range(self.size):
            client = ClaudeSDKClient(make_options(self.default_model))
            await client.connect()
            self._client_models[client] = self.default_model
            self._available.append(client)
            if self.size > 1:
                logger.info(f"{_CYAN}[pool]{_RESET} Client {i + 1}/{self.size} connected {_DIM}({self.default_model}){_RESET}")

        self._initialized = True
        logger.info(f"{_CYAN}[pool]{_RESET} {_GREEN}Ready{_RESET} | available={len(self._available)}")
        self._fire_change()

        # Start periodic health check
        self._health_check_task = asyncio.create_task(self._periodic_health_check())

    async def _create_client(self, model: str) -> ClaudeSDKClient:
        """Create and connect a new client with specified model."""
        client = ClaudeSDKClient(make_options(model))
        await client.connect()
        self._client_models[client] = model
        return client

    async def _disconnect_client(self, client: ClaudeSDKClient) -> None:
        """Disconnect client and remove from tracking."""
        try:
            await client.disconnect()
        except Exception:
            pass
        self._client_models.pop(client, None)

    def _spawn_background(self, coro) -> None:
        """Create a tracked background task that cleans up after itself."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _disconnect_client_background(self, client: ClaudeSDKClient) -> None:
        """Disconnect client in background, logging errors."""
        try:
            await self._disconnect_client(client)
        except Exception as e:
            logger.warning(f"[pool] Background disconnect failed: {e}")

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
        models = [self._client_models[c] for c in self._available]
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
    async def acquire(self, model: str, request_id: str | None = None) -> AsyncIterator[ClaudeSDKClient]:
        """Get a fresh client for the specified model.

        Takes a pre-warmed client with matching model if available,
        otherwise creates a new one. Client is ALWAYS destroyed after use.

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
            logger.error(f"{tag} Semaphore timeout after {self._acquire_timeout}s | pool={self.snapshot()}")
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=f"All pool clients busy. Timed out after {self._acquire_timeout}s waiting for an available client."
            )

        client: ClaudeSDKClient | None = None
        completed = False

        try:
            async with self._lock:
                # Take a pre-warmed client with matching model if available
                matching = [c for c in self._available if self._client_models[c] == model]
                if matching:
                    client = matching[0]
                    self._available.remove(client)
                    logger.info(f"{tag} Using pre-warmed {model} client")
                elif self._available:
                    # Discard non-matching pre-warmed client
                    old = self._available.pop(0)
                    old_model = self._client_models.get(old, "unknown")
                    logger.info(f"{tag} Discarding pre-warmed {old_model} client, need {model}")
                    self._spawn_background(self._disconnect_client_background(old))

                self._in_use += 1
                self._log_status("Acquired", request_id)
                self._fire_change()

            # Create fresh if no pre-warmed client was available
            if client is None:
                client = await self._create_client(model)
                logger.info(f"{tag} Created fresh {model} client")

            yield client
            completed = True
        except Exception:
            raise
        finally:
            async with self._lock:
                self._in_use -= 1
                self._fire_change()
            # ALWAYS destroy after use — never return to pool
            if client is not None:
                self._spawn_background(self._disconnect_client_background(client))
                status = "after use" if completed else "after cancellation"
                logger.info(f"{tag} Destroyed {model} client {status}")
            # Pre-warm a replacement
            self._spawn_background(self._prewarm_client(model))
            self._semaphore.release()
            self._log_status("Released", request_id)

    async def _prewarm_client(self, model: str) -> None:
        """Create a fresh client and add to available pool for next request."""
        try:
            client = await self._create_client(model)
            async with self._lock:
                if len(self._available) + self._in_use < self.size:
                    self._available.append(client)
                    logger.info(f"[pool] Pre-warmed {model} client")
                    self._fire_change()
                else:
                    await self._disconnect_client(client)
                    self._fire_change()
        except Exception as e:
            logger.warning(f"[pool] Pre-warm failed: {e}")

    async def _periodic_health_check(self) -> None:
        """Background task that checks idle client health every HEALTH_CHECK_INTERVAL seconds."""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            try:
                await self._check_idle_clients()
            except Exception as e:
                logger.warning(f"[pool] Health check error: {e}")

    async def _check_idle_clients(self) -> None:
        """Check each idle client's process is alive, replace if not."""
        async with self._lock:
            clients_to_check = list(self._available)

        for client in clients_to_check:
            try:
                # Check if the underlying process is still alive
                if hasattr(client, '_process') and client._process is not None:
                    if client._process.returncode is not None:
                        raise RuntimeError("Process exited")
            except Exception:
                model = self._client_models.get(client, self.default_model)
                logger.warning(f"[pool] Health check: {model} client unresponsive, replacing")
                async with self._lock:
                    if client in self._available:
                        self._available.remove(client)
                await self._disconnect_client(client)
                self._fire_change()
                try:
                    new_client = await self._create_client(model)
                    async with self._lock:
                        self._available.append(new_client)
                    self._fire_change()
                    logger.info(f"[pool] Health check: replaced unresponsive {model} client")
                except Exception as e:
                    logger.error(f"[pool] Health check: failed to replace client: {e}")

    async def shutdown(self) -> None:
        """Disconnect all clients and clean up."""
        logger.info(f"[pool] Shutting down {len(self._client_models)} clients...")

        # Cancel health check task
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        # Wait for in-flight background tasks (disconnects, pre-warms)
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        for client in list(self._client_models.keys()):
            await self._disconnect_client(client)

        self._available.clear()
        self._initialized = False
        self._in_use = 0
        self._fire_change()

        logger.info("[pool] Shutdown complete")
