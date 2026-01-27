"""Client pool for reduced latency via connection reuse with dynamic model replacement."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

logger = logging.getLogger(__name__)


def make_options(model: str) -> ClaudeAgentOptions:
    """Create ClaudeAgentOptions with model."""
    return ClaudeAgentOptions(
        max_turns=1,
        setting_sources=None,  # Don't load user filesystem settings
        system_prompt={"type": "preset", "preset": "claude_code"},
        model=model,
        env={"CLAUDE_CODE_BRIDGE": "1"},
        tools=[],  # No built-in tools - pure chat mode
    )


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

    async def _clear_client_state(self, client: ClaudeSDKClient) -> None:
        """Clear conversation state using /clear command."""
        try:
            await client.query("/clear")
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    break
        except Exception:
            raise

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
        if client in self._client_models:
            del self._client_models[client]

    def _log_status(self, action: str) -> None:
        """Log current pool status with model breakdown."""
        model_counts: dict[str, int] = {}
        for model in self._client_models.values():
            model_counts[model] = model_counts.get(model, 0) + 1

        available_models = [self._client_models[c] for c in self._available]
        available_str = f"[{', '.join(available_models)}]" if available_models else "[]"

        logger.info(
            f"[pool] {action} | in_use={self._in_use} available={len(self._available)} "
            f"models={available_str}"
        )

    @asynccontextmanager
    async def acquire(self, model: str) -> AsyncIterator[ClaudeSDKClient]:
        """Get a client for the specified model, replacing if necessary.

        If a client with the matching model is available, use it.
        Otherwise, replace an available client with a new one for the requested model.

        Args:
            model: Model name (opus, sonnet, haiku).

        Yields:
            A ClaudeSDKClient configured for the specified model.
        """
        await self._semaphore.acquire()  # Wait if all clients in use

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
                    # Replace a client with different model
                    old_client = self._available.pop(0)
                    old_model = self._client_models[old_client]
                    logger.info(f"[pool] Replacing {old_model} client with {model}")
                    await self._disconnect_client(old_client)
                    client = await self._create_client(model)
                    replaced = True
                else:
                    # No available clients (shouldn't happen with semaphore)
                    logger.warning("[pool] No available clients, creating new one")
                    client = await self._create_client(model)

                self._in_use += 1
                self._log_status("Acquired" + (" (replaced)" if replaced else ""))

            # Clear conversation state before use
            await self._clear_client_state(client)
            yield client

        except Exception as e:
            # If something went wrong, try to replace the client
            logger.warning(f"[pool] Client error: {e}")
            if client is not None:
                async with self._lock:
                    try:
                        await self._disconnect_client(client)
                        client = await self._create_client(model)
                        logger.info(f"[pool] Client replaced after error")
                    except Exception:
                        logger.error("[pool] Failed to replace client after error")
                        client = None
            raise
        finally:
            async with self._lock:
                self._in_use -= 1
                if client is not None:
                    self._available.append(client)
                self._log_status("Released")
            self._semaphore.release()

    async def shutdown(self) -> None:
        """Disconnect all clients and clean up."""
        logger.info(f"[pool] Shutting down {len(self._client_models)} clients...")

        for client in list(self._client_models.keys()):
            await self._disconnect_client(client)

        self._available.clear()
        self._initialized = False
        self._in_use = 0

        logger.info("[pool] Shutdown complete")
