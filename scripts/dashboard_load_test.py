#!/usr/bin/env python3
"""Dashboard load test — fires concurrent requests to exercise the dashboard."""

import argparse
import asyncio
import base64
import time
from pathlib import Path

import httpx

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
IMAGE_PATH = FIXTURES_DIR / "ocr_test_document.png"

MODELS = ["sonnet", "haiku", "opus"]

TEXT_PROMPTS = [
    "Count from 1 to 5",
    "What is 2+2?",
    "Name three colors",
    "What day comes after Monday?",
    "Say hello in three languages",
    "List the first 4 prime numbers",
    "What is the capital of France?",
    "Spell the word 'bridge' backwards",
]

IMAGE_PROMPT = "Describe what you see in this image in one sentence."


def load_image_data_url() -> str:
    """Load the test image as a base64 data URL."""
    raw = IMAGE_PATH.read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:image/png;base64,{b64}"


async def complete_messages(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict],
    model: str,
    stream: bool,
) -> None:
    """Send a chat completion request using httpx.AsyncClient."""
    payload = {"model": model, "messages": messages, "stream": stream}
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    if stream:
        async with client.stream("POST", endpoint, json=payload) as resp:
            resp.raise_for_status()
            async for _ in resp.aiter_lines():
                pass
    else:
        resp = await client.post(endpoint, json=payload)
        resp.raise_for_status()


async def run_request(
    client: httpx.AsyncClient,
    base_url: str,
    index: int,
    total: int,
    kind: str,
    messages: list[dict],
    stream: bool,
    model: str,
) -> dict:
    """Run a single request and return a summary dict."""
    tag = f"[{index+1}/{total}]"
    mode = "stream" if stream else "no-stream"
    print(f"{tag} START  {kind:<12} {mode:<10} {model}")

    t0 = time.monotonic()
    try:
        await complete_messages(client, base_url, messages, model=model, stream=stream)
        elapsed = time.monotonic() - t0
        print(f"{tag} DONE   {kind:<12} {mode:<10} {model}  {elapsed:.1f}s")
        return {"index": index, "kind": kind, "model": model, "stream": stream, "elapsed": elapsed, "status": "ok"}
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"{tag} FAIL   {kind:<12} {mode:<10} {model}  {elapsed:.1f}s  {e}")
        return {"index": index, "kind": kind, "model": model, "stream": stream, "elapsed": elapsed, "status": str(e)}


async def main(count: int, models: list[str], url: str) -> None:
    image_data_url = load_image_data_url()

    # Derive server root from API base URL (strip /api suffix) for health check
    server_root = url.removesuffix("/api").removesuffix("/")
    try:
        resp = httpx.get(f"{server_root}/health", timeout=5.0)
        if resp.status_code != 200:
            print(f"Error: server not healthy at {server_root}/health")
            return
    except (httpx.ConnectError, httpx.TimeoutException):
        print(f"Error: server not reachable at {server_root}")
        return

    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"Sending {count} requests to {url} (models={', '.join(models)})\n")

        tasks = []
        for i in range(count):
            prompt_text = TEXT_PROMPTS[i % len(TEXT_PROMPTS)]
            model = models[i % len(models)]

            if i % 3 == 0:
                # Streaming text
                kind = "text-stream"
                messages = [{"role": "user", "content": prompt_text}]
                stream = True
            elif i % 3 == 1:
                # Non-streaming text
                kind = "text"
                messages = [{"role": "user", "content": prompt_text}]
                stream = False
            else:
                # Multimodal with image (streaming)
                kind = "image-stream"
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]}]
                stream = True

            async def launch(idx=i, k=kind, m=messages, s=stream, mdl=model):
                await asyncio.sleep(idx * 0.2)
                return await run_request(client, url, idx, count, k, m, s, model=mdl)

            tasks.append(asyncio.create_task(launch()))

        results = await asyncio.gather(*tasks)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = count - ok
    avg = sum(r["elapsed"] for r in results) / count if count else 0
    print(f"\nDone: {ok} ok, {fail} failed, avg {avg:.1f}s")

    # Per-model breakdown
    model_names = sorted(set(r["model"] for r in results))
    if len(model_names) > 1:
        for m in model_names:
            m_results = [r for r in results if r["model"] == m]
            m_ok = sum(1 for r in m_results if r["status"] == "ok")
            m_fail = len(m_results) - m_ok
            m_avg = sum(r["elapsed"] for r in m_results) / len(m_results)
            print(f"  {m:<10} {m_ok} ok, {m_fail} failed, avg {m_avg:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dashboard load test")
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of requests (default: 5)")
    parser.add_argument("--models", "-m", nargs="+", default=MODELS, help="Models to cycle through (default: sonnet haiku opus)")
    parser.add_argument("--url", "-u", default="http://localhost:8082/api", help="Server URL (default: http://localhost:8082/api)")
    args = parser.parse_args()
    asyncio.run(main(args.count, args.models, args.url))
