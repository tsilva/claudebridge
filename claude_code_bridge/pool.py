"""Client pool for reduced latency via connection reuse."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

logger = logging.getLogger(__name__)


class ClientPool:
    """Pool of persistent ClaudeSDKClient instances for reduced latency.

    Maintains a pool of pre-connected clients that can be reused across requests.
    Uses /clear command between requests to reset conversation state while keeping
    the subprocess warm.
    """

    def __init__(self, size: int = 3, options: ClaudeAgentOptions | None = None):
        """Initialize client pool.

        Args:
            size: Number of clients to maintain in the pool.
            options: Claude agent options for all clients.
        """
        self.size = size
        self.options = options
        self._available: asyncio.Queue[ClaudeSDKClient] = asyncio.Queue()
        self._all_clients: list[ClaudeSDKClient] = []
        self._initialized = False
        self._in_use = 0  # Track clients currently in use

    async def initialize(self) -> None:
        """Pre-spawn all clients and connect them."""
        if self._initialized:
            return

        logger.info(f"[pool] Initializing pool with {self.size} clients...")
        for i in range(self.size):
            client = ClaudeSDKClient(self.options)
            await client.connect()
            self._all_clients.append(client)
            await self._available.put(client)
            logger.info(f"[pool] Client {i + 1}/{self.size} connected")

        self._initialized = True
        logger.info(f"[pool] Pool ready: {self.size} clients available")

    async def _clear_client_state(self, client: ClaudeSDKClient) -> None:
        """Clear conversation state using /clear command."""
        try:
            await client.query("/clear")
            # Drain the clear acknowledgment
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    break
        except Exception:
            # If /clear fails, the client might be in a bad state
            # We'll handle this in acquire by replacing the client
            raise

    async def _replace_client(self, old_client: ClaudeSDKClient) -> ClaudeSDKClient:
        """Replace an unhealthy client with a fresh one."""
        try:
            await old_client.disconnect()
        except Exception:
            pass  # Ignore disconnect errors

        if old_client in self._all_clients:
            self._all_clients.remove(old_client)

        new_client = ClaudeSDKClient(self.options)
        await new_client.connect()
        self._all_clients.append(new_client)
        return new_client

    def _log_status(self, action: str) -> None:
        """Log current pool status."""
        available = self._available.qsize()
        total = len(self._all_clients)
        logger.info(f"[pool] {action} | in_use={self._in_use} available={available} total={total}")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[ClaudeSDKClient]:
        """Get a client from the pool, clear its state, yield it, return to pool.

        Yields:
            A ClaudeSDKClient ready for use with clean conversation state.
        """
        # Check if we'll need to wait
        if self._available.empty():
            logger.warning(f"[pool] All {len(self._all_clients)} clients in use, request will wait...")

        client = await self._available.get()
        self._in_use += 1
        self._log_status("Acquired")
        client_to_return: ClaudeSDKClient | None = client

        try:
            # Clear conversation state before use
            await self._clear_client_state(client)
            yield client
        except Exception as e:
            # If something went wrong, replace the client
            logger.warning(f"[pool] Client error, replacing: {e}")
            try:
                client_to_return = await self._replace_client(client)
                logger.info("[pool] Client replaced successfully")
            except Exception:
                # If replacement also fails, try creating a fresh client
                logger.warning("[pool] Replacement failed, creating fresh client...")
                try:
                    new_client = ClaudeSDKClient(self.options)
                    await new_client.connect()
                    self._all_clients.append(new_client)
                    client_to_return = new_client
                    logger.info("[pool] Fresh client created")
                except Exception:
                    # Can't create a new client - pool capacity reduced
                    logger.error("[pool] Failed to create client, pool capacity reduced")
                    client_to_return = None
            raise e
        finally:
            self._in_use -= 1
            if client_to_return is not None:
                await self._available.put(client_to_return)
            self._log_status("Released")

    async def shutdown(self) -> None:
        """Disconnect all clients and clean up."""
        logger.info(f"[pool] Shutting down {len(self._all_clients)} clients...")
        for client in self._all_clients:
            try:
                await client.disconnect()
            except Exception:
                pass  # Ignore disconnect errors

        self._all_clients.clear()
        self._initialized = False
        self._in_use = 0

        # Clear the queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("[pool] Shutdown complete")
