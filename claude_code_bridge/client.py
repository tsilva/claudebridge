"""
BridgeClient - Reusable client for Claude Code Bridge.

Provides both sync and async methods for programmatic use, plus CLI for ad-hoc testing.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator

import httpx
from openai import AsyncOpenAI, OpenAI
import openai
from dotenv import load_dotenv

# Load .env file from current directory or parents
load_dotenv()

# Configuration via environment variables
DEFAULT_BASE_URL = os.environ.get("BRIDGE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL")


class BridgeClient:
    """Client for Claude Code Bridge with sync and async methods."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        model: str | None = None,
    ):
        """
        Initialize BridgeClient.

        Args:
            base_url: API base URL (default: $BRIDGE_URL or http://localhost:8000)
            api_key: API key (default: $OPENROUTER_API_KEY or "not-needed")
            timeout: Request timeout in seconds (default: 300)
            model: Default model to use (default: $OPENROUTER_MODEL or "sonnet")
        """
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or DEFAULT_API_KEY or "not-needed"
        self.timeout = timeout
        self.model = model or DEFAULT_MODEL or "sonnet"

        # Ensure /v1 suffix for OpenAI clients
        self._openai_base_url = self.base_url
        if not self._openai_base_url.endswith("/v1"):
            self._openai_base_url = f"{self._openai_base_url}/v1"

        # Lazy-initialized clients
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        self._http_client: httpx.Client | None = None

    @property
    def sync_client(self) -> OpenAI:
        """Get or create sync OpenAI client."""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                base_url=self._openai_base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self._openai_base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._async_client

    @property
    def http_client(self) -> httpx.Client:
        """Get or create httpx client for non-OpenAI endpoints."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self.timeout)
        return self._http_client

    # -------------------------------------------------------------------------
    # Sync methods (for simple tests)
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if server is healthy. Returns True if server responds OK."""
        try:
            response = self.http_client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200 and response.json().get("status") == "ok"
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def list_models(self) -> list[str]:
        """List available model IDs."""
        models = self.sync_client.models.list()
        return [model.id for model in models.data]

    def complete_sync(
        self,
        prompt: str,
        model: str | None = None,
        stream: bool = False,
    ) -> str:
        """
        Send a completion request synchronously.

        Args:
            prompt: User message
            model: Model to use (default: instance model)
            stream: Whether to stream (collects all chunks if True)

        Returns:
            Complete response content
        """
        messages = [{"role": "user", "content": prompt}]
        return self.complete_messages_sync(messages, model=model, stream=stream)

    def complete_messages_sync(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = False,
    ) -> str:
        """
        Send a chat completion with full message list synchronously.

        Args:
            messages: List of message dicts with role and content
            model: Model to use (default: instance model)
            stream: Whether to stream (collects all chunks if True)

        Returns:
            Complete response content
        """
        model = model or self.model

        if stream:
            response = self.sync_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            chunks = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            return "".join(chunks)
        else:
            response = self.sync_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content or ""

    # -------------------------------------------------------------------------
    # Async methods (for parallel testing)
    # -------------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        stream: bool = True,
    ) -> str:
        """
        Send a completion request asynchronously.

        Args:
            prompt: User message
            model: Model to use (default: instance model)
            stream: Whether to stream (default: True)

        Returns:
            Complete response content
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.complete_messages(messages, model=model, stream=stream)

    async def complete_messages(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = True,
    ) -> str:
        """
        Send a chat completion with full message list asynchronously.

        Args:
            messages: List of message dicts with role and content
            model: Model to use (default: instance model)
            stream: Whether to stream (default: True)

        Returns:
            Complete response content
        """
        model = model or self.model

        if stream:
            chunks = []
            async for chunk in self.stream_messages(messages, model=model):
                chunks.append(chunk)
            return "".join(chunks)
        else:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content or ""

    async def stream(
        self,
        prompt: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion response.

        Args:
            prompt: User message
            model: Model to use (default: instance model)

        Yields:
            Content chunks as they arrive
        """
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.stream_messages(messages, model=model):
            yield chunk

    async def stream_messages(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion with full message list.

        Args:
            messages: List of message dicts with role and content
            model: Model to use (default: instance model)

        Yields:
            Content chunks as they arrive
        """
        model = model or self.model

        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # -------------------------------------------------------------------------
    # Context manager support
    # -------------------------------------------------------------------------

    async def close(self) -> None:
        """Close all clients."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def close_sync(self) -> None:
        """Close all clients (sync version)."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        # Note: async client should be closed with close() in async context

    async def __aenter__(self) -> "BridgeClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __enter__(self) -> "BridgeClient":
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit."""
        self.close_sync()


# -----------------------------------------------------------------------------
# CLI for ad-hoc testing
# -----------------------------------------------------------------------------


async def _run_parallel(
    client: BridgeClient,
    prompt: str,
    count: int,
    stream: bool,
) -> None:
    """Run multiple requests in parallel with staggered starts."""
    show_index = count > 1
    stagger_delay = 0.1  # 100ms between request launches

    async def run_one(index: int, delay: float) -> tuple[int, str]:
        if delay > 0:
            await asyncio.sleep(delay)
        content = await client.complete(prompt, stream=stream)
        return index, content

    tasks = [
        asyncio.create_task(run_one(i + 1, i * stagger_delay))
        for i in range(count)
    ]

    # Print responses as they complete
    for coro in asyncio.as_completed(tasks):
        index, content = await coro
        prefix = f"[{index}] " if show_index else ""
        print(f"{prefix}{content}")
        print()  # Blank line between responses


def _cli_main() -> None:
    """CLI entry point for ad-hoc testing."""
    parser = argparse.ArgumentParser(
        description="Send a prompt to Claude Code Bridge (or any OpenAI-compatible API).",
        epilog="Examples:\n"
               "  python -m claude_code_bridge.client 'What is Python?'\n"
               "  echo 'Hello' | python -m claude_code_bridge.client\n"
               "  python -m claude_code_bridge.client --model opus 'Explain decorators'\n"
               "  python -m claude_code_bridge.client -n 3 'Hello'  # 3 parallel requests\n"
               "  python -m claude_code_bridge.client --no-stream 'Hello'\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send (reads from stdin if not provided)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL or "sonnet",
        help="Model to use (default: $OPENROUTER_MODEL or 'sonnet')",
    )
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_BASE_URL,
        help="API base URL (default: $BRIDGE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=DEFAULT_API_KEY,
        help="API key for authentication (default: $OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (wait for full response)",
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        default=1,
        help="Number of parallel requests to make (default: 1)",
    )

    args = parser.parse_args()

    # Get prompt from argument or stdin
    if args.prompt:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        parser.error("No prompt provided. Pass as argument or pipe via stdin.")

    if not prompt:
        parser.error("Empty prompt provided.")

    if args.parallel < 1:
        parser.error("Parallel count must be at least 1.")

    client = BridgeClient(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
    )

    try:
        asyncio.run(_run_parallel(
            client=client,
            prompt=prompt,
            count=args.parallel,
            stream=not args.no_stream,
        ))
    except openai.APIConnectionError:
        print(f"Error: Could not connect to {args.url}", file=sys.stderr)
        print("Make sure the server is running.", file=sys.stderr)
        sys.exit(1)
    except openai.AuthenticationError:
        print("Error: Authentication failed", file=sys.stderr)
        print("Check your API key (--api-key or $OPENROUTER_API_KEY)", file=sys.stderr)
        sys.exit(1)
    except openai.APIStatusError as e:
        print(f"Error: HTTP {e.status_code}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
