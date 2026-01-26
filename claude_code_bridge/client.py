"""Simple CLI client for Claude Code Bridge."""

import argparse
import asyncio
import sys
import json

import httpx


DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = None  # Use local Claude Code settings


async def stream_response_async(
    client: httpx.AsyncClient,
    url: str,
    model: str | None,
    prompt: str,
    index: int,
    delay: float = 0.0,
) -> tuple[int, str]:
    """Stream a single response, returning index and collected content."""
    if delay > 0:
        await asyncio.sleep(delay)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if model:
        payload["model"] = model

    chunks: list[str] = []

    async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        chunks.append(content)
                except json.JSONDecodeError:
                    continue

    return index, "".join(chunks)


async def get_response_async(
    client: httpx.AsyncClient,
    url: str,
    model: str | None,
    prompt: str,
    index: int,
    delay: float = 0.0,
) -> tuple[int, str]:
    """Get a non-streaming response, returning index and content."""
    if delay > 0:
        await asyncio.sleep(delay)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if model:
        payload["model"] = model

    response = await client.post(f"{url}/v1/chat/completions", json=payload)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return index, content


async def run_parallel(
    url: str,
    model: str | None,
    prompt: str,
    count: int,
    stream: bool,
) -> None:
    """Run multiple requests in parallel with staggered starts."""
    show_index = count > 1
    stagger_delay = 0.1  # 100ms between request launches

    async with httpx.AsyncClient(timeout=300.0) as client:
        if stream:
            tasks = [
                asyncio.create_task(
                    stream_response_async(
                        client, url, model, prompt, i + 1,
                        delay=i * stagger_delay,
                    )
                )
                for i in range(count)
            ]
        else:
            tasks = [
                asyncio.create_task(
                    get_response_async(
                        client, url, model, prompt, i + 1,
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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Send a prompt to Claude Code Bridge and get a response.",
        epilog="Examples:\n"
               "  claude-code-client 'What is Python?'\n"
               "  echo 'Hello' | claude-code-client\n"
               "  claude-code-client --model opus 'Explain decorators'\n"
               "  claude-code-client -n 3 'Hello'  # Run 3 parallel requests",
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
        help="Model to use: opus, sonnet, haiku (default: local Claude Code settings)",
    )
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_URL,
        help=f"Bridge URL (default: {DEFAULT_URL})",
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
        ))
    except httpx.ConnectError:
        print(f"Error: Could not connect to bridge at {args.url}", file=sys.stderr)
        print("Make sure the bridge is running: claude-code-bridge", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: HTTP {e.response.status_code}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
