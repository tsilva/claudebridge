"""Simple CLI client for Claude Code Bridge."""

import argparse
import asyncio
import os
import sys

import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load .env file from current directory or parents
load_dotenv()

# Configuration via environment variables (OpenRouter-compatible)
DEFAULT_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL")


async def stream_response_async(
    client: AsyncOpenAI,
    model: str | None,
    prompt: str,
    index: int,
    delay: float = 0.0,
) -> tuple[int, str]:
    """Stream a single response, returning index and collected content."""
    if delay > 0:
        await asyncio.sleep(delay)

    kwargs = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if model:
        kwargs["model"] = model
    else:
        # OpenAI client requires a model, use placeholder that server will map
        kwargs["model"] = "default"

    chunks: list[str] = []

    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    return index, "".join(chunks)


async def get_response_async(
    client: AsyncOpenAI,
    model: str | None,
    prompt: str,
    index: int,
    delay: float = 0.0,
) -> tuple[int, str]:
    """Get a non-streaming response, returning index and content."""
    if delay > 0:
        await asyncio.sleep(delay)

    kwargs = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if model:
        kwargs["model"] = model
    else:
        # OpenAI client requires a model, use placeholder that server will map
        kwargs["model"] = "default"

    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content

    return index, content or ""


async def run_parallel(
    url: str,
    model: str | None,
    prompt: str,
    count: int,
    stream: bool,
    api_key: str | None = None,
) -> None:
    """Run multiple requests in parallel with staggered starts."""
    show_index = count > 1
    stagger_delay = 0.1  # 100ms between request launches

    # Ensure base_url ends with /v1 for OpenAI client
    base_url = url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or "not-needed",
        timeout=300.0,
    )

    try:
        if stream:
            tasks = [
                asyncio.create_task(
                    stream_response_async(
                        client, model, prompt, i + 1,
                        delay=i * stagger_delay,
                    )
                )
                for i in range(count)
            ]
        else:
            tasks = [
                asyncio.create_task(
                    get_response_async(
                        client, model, prompt, i + 1,
                        delay=i * stagger_delay,
                    )
                )
                for i in range(count)
            ]

        # Print responses as they complete
        for coro in asyncio.as_completed(tasks):
            index, content = await coro
            prefix = f"[{index}] " if show_index else ""
            print(f"{prefix}{content}")
            print()  # Blank line between responses
    finally:
        await client.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Send a prompt to an OpenAI-compatible API (local bridge or OpenRouter).",
        epilog="Examples:\n"
               "  # Local bridge (default)\n"
               "  claude-code-client 'What is Python?'\n"
               "  echo 'Hello' | claude-code-client\n"
               "  claude-code-client --model opus 'Explain decorators'\n"
               "  claude-code-client -n 3 'Hello'  # Run 3 parallel requests\n"
               "\n"
               "  # OpenRouter (via env vars)\n"
               "  export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n"
               "  export OPENROUTER_API_KEY=sk-or-v1-xxxxx\n"
               "  export OPENROUTER_MODEL=anthropic/claude-sonnet-4\n"
               "  claude-code-client 'Hello'\n"
               "\n"
               "  # OpenRouter (via CLI args)\n"
               "  claude-code-client -u https://openrouter.ai/api/v1 \\\n"
               "      -k sk-or-v1-xxxxx -m anthropic/claude-sonnet-4 'Hello'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send (reads from stdin if not provided)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help="Model to use (default: $OPENROUTER_MODEL or local Claude Code settings)",
    )
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_BASE_URL,
        help="API base URL (default: $OPENROUTER_BASE_URL or http://localhost:8000)",
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

    try:
        asyncio.run(run_parallel(
            url=args.url,
            model=args.model,
            prompt=prompt,
            count=args.parallel,
            stream=not args.no_stream,
            api_key=args.api_key,
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
    main()
