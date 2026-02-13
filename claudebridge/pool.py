"""Client pool for reduced latency via connection reuse with dynamic model replacement."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

logger = logging.getLogger(__name__)

# Timeout for clear command (seconds)
CLEAR_TIMEOUT = 10
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
    """Pool of persistent ClaudeSDKClient instances with dynamic model replacement.

    Maintains a pool of pre-connected clients that can be reused across requests.
    Tracks model per client and replaces clients on-demand when different model is requested.
    Prefers reusing clients with matching models to avoid replacement overhead.
    """

    def __init__(self, size: int = 3, default_model: str = "opus"):
        """Initialize client pool.

        Args:
            size: Number of clients to maintain in the pool.
            default_model: Model to use for initial pool population.
        """
        self.size = size
        self.default_model = default_model
        self._client_models: dict[ClaudeSDKClient, str] = {}  # client -> model
        self._available: list[ClaudeSDKClient] = []  # available clients
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(size)  # limits concurrent usage
        self._initialized = False
        self._in_use = 0
        self._acquire_timeout = int(os.environ.get("CLAUDE_TIMEOUT", 120))
        self._health_check_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Pre-spawn all clients with default model."""
        if self._initialized:
            return

        logger.info(f"[pool] Initializing with {self.size} {self.default_model} clients...")
        for i in range(self.size):
            client = ClaudeSDKClient(make_options(self.default_model))
            await client.connect()
            self._client_models[client] = self.default_model
            self._available.append(client)
            logger.info(f"[pool] Client {i + 1}/{self.size} connected ({self.default_model})")

        self._initialized = True
        self._log_status("Initialized")

        # Start periodic health check
        self._health_check_task = asyncio.create_task(self._periodic_health_check())

    async def _clear_client_state(self, client: ClaudeSDKClient) -> None:
        """Clear conversation state using /clear command with timeout."""
        try:
            await asyncio.wait_for(self._do_clear(client), timeout=CLEAR_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("[pool] /clear timed out, replacing client")
            raise

    async def _do_clear(self, client: ClaudeSDKClient) -> None:
        """Execute the /clear command."""
        await client.query("/clear")
        async for msg in client.receive_response():
            if isinstance(msg, ResultMessage):
                break

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

    def _log_status(self, action: str) -> None:
        """Log current pool status with model breakdown."""
        available_models = [self._client_models[c] for c in self._available]
        available_str = f"[{', '.join(available_models)}]"

        logger.info(
            f"[pool] {action} | in_use={self._in_use} available={len(self._available)} "
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

    @asynccontextmanager
    async def acquire(self, model: str) -> AsyncIterator[ClaudeSDKClient]:
        """Get a client for the specified model, replacing if necessary.

        If a client with the matching model is available, use it.
        Otherwise, replace an available client with a new one for the requested model.

        Args:
            model: Model name (opus, sonnet, haiku).

        Yields:
            A ClaudeSDKClient configured for the specified model.

        Raises:
            HTTPException: 503 if acquire times out waiting for a client.
        """
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._acquire_timeout
            )
        except asyncio.TimeoutError:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=f"All pool clients busy. Timed out after {self._acquire_timeout}s waiting for an available client."
            )

        client: ClaudeSDKClient | None = None
        replaced = False

        try:
            async with self._lock:
                # Find client with matching model
                matching = [c for c in self._available if self._client_models[c] == model]
                if matching:
                    client = matching[0]
                    self._available.remove(client)
                    logger.info(f"[pool] Using existing {model} client")
                elif self._available:
                    # Take client out; will replace outside lock
                    old_client = self._available.pop(0)
                    old_model = self._client_models[old_client]
                    logger.info(f"[pool] Replacing {old_model} client with {model}")
                    replaced = True
                else:
                    # No available clients (shouldn't happen with semaphore)
                    logger.warning("[pool] No available clients, creating new one")

                self._in_use += 1
                self._log_status("Acquired" + (" (replacing)" if replaced else ""))

            # Client replacement happens outside the lock
            if replaced:
                await self._disconnect_client(old_client)
                client = await self._create_client(model)
            elif client is None:
                client = await self._create_client(model)

            # Clear conversation state before use (with timeout)
            try:
                await self._clear_client_state(client)
            except (asyncio.TimeoutError, Exception) as clear_err:
                # Clear failed — replace the client
                logger.warning(f"[pool] Clear failed: {clear_err}, replacing client")
                await self._disconnect_client(client)
                client = await self._create_client(model)
                # Don't try to clear the fresh client — it's already clean

            yield client

        except Exception as e:
            # If something went wrong, try to replace the client
            logger.warning(f"[pool] Client error: {e}")
            if client is not None:
                try:
                    await self._disconnect_client(client)
                    client = await self._create_client(model)
                    logger.info("[pool] Client replaced after error")
                except Exception:
                    logger.error("[pool] Failed to replace client after error")
                    # Schedule background recovery to restore pool size
                    asyncio.create_task(self._recover_client(model))
                    client = None
            raise
        finally:
            async with self._lock:
                self._in_use -= 1
                if client is not None:
                    self._available.append(client)
                elif self._in_use + len(self._available) < self.size:
                    # Pool shrank — schedule recovery
                    asyncio.create_task(self._recover_client(self.default_model))
                self._log_status("Released")
            self._semaphore.release()

    async def _recover_client(self, model: str) -> None:
        """Background task to restore pool to full size after a client is lost."""
        for attempt in range(3):
            try:
                client = await self._create_client(model)
                async with self._lock:
                    if len(self._available) + self._in_use < self.size:
                        self._available.append(client)
                        logger.info("[pool] Recovery: client restored")
                        self._log_status("Recovered")
                    else:
                        # Pool is already full, disconnect the extra client
                        await self._disconnect_client(client)
                return
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"[pool] Recovery attempt {attempt + 1} failed: {e}, retrying in {wait}s")
                await asyncio.sleep(wait)

        logger.error("[pool] Recovery failed after 3 attempts, pool may be degraded")

    async def _periodic_health_check(self) -> None:
        """Background task that checks idle client health every HEALTH_CHECK_INTERVAL seconds."""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            try:
                await self._check_idle_clients()
            except Exception as e:
                logger.warning(f"[pool] Health check error: {e}")

    async def _check_idle_clients(self) -> None:
        """Check each idle client is responsive, replace if not."""
        async with self._lock:
            clients_to_check = list(self._available)

        for client in clients_to_check:
            try:
                await asyncio.wait_for(self._do_clear(client), timeout=CLEAR_TIMEOUT)
            except Exception:
                model = self._client_models.get(client, self.default_model)
                logger.warning(f"[pool] Health check: {model} client unresponsive, replacing")
                async with self._lock:
                    if client in self._available:
                        self._available.remove(client)
                await self._disconnect_client(client)
                try:
                    new_client = await self._create_client(model)
                    async with self._lock:
                        self._available.append(new_client)
                    logger.info(f"[pool] Health check: replaced unresponsive {model} client")
                except Exception as e:
                    logger.error(f"[pool] Health check: failed to replace client: {e}")
                    asyncio.create_task(self._recover_client(model))

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

        for client in list(self._client_models.keys()):
            await self._disconnect_client(client)

        self._available.clear()
        self._initialized = False
        self._in_use = 0

        logger.info("[pool] Shutdown complete")
