"""Simple CLI client for the Claude Code OpenAI Proxy."""

import argparse
import sys
import json

import httpx


DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "sonnet"


def stream_response(client: httpx.Client, url: str, model: str, prompt: str) -> None:
    """Stream a response from the proxy and print as it arrives."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
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
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue
    print()  # Final newline


def get_response(client: httpx.Client, url: str, model: str, prompt: str) -> str:
    """Get a non-streaming response from the proxy."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    response = client.post(f"{url}/v1/chat/completions", json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Send a prompt to the Claude proxy and get a response.",
        epilog="Examples:\n"
               "  claude-client 'What is Python?'\n"
               "  echo 'Hello' | claude-client\n"
               "  claude-client --model opus 'Explain decorators'",
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
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_URL,
        help=f"Proxy URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (wait for full response)",
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

    try:
        with httpx.Client(timeout=300.0) as client:
            if args.no_stream:
                response = get_response(client, args.url, args.model, prompt)
                print(response)
            else:
                stream_response(client, args.url, args.model, prompt)
    except httpx.ConnectError:
        print(f"Error: Could not connect to proxy at {args.url}", file=sys.stderr)
        print("Make sure the proxy is running: claude-bridge", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: HTTP {e.response.status_code}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
