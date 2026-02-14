"""FastAPI routes for the dashboard."""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

from .dashboard_state import DashboardState


def _mask_api_key(api_key: str | None) -> str:
    """Mask API key for display, showing only first 8 chars."""
    if not api_key or api_key == "anonymous":
        return "anonymous"
    if len(api_key) <= 8:
        return api_key
    return api_key[:8] + "..."

TEMPLATES_DIR = Path(__file__).parent / "templates" / "dashboard"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _parse_log_file(path: Path) -> dict | None:
    """Parse a session log file into a structured dict.

    Returns dict with: request_id, model, timestamp, duration_ms, acquire_ms,
    query_ms, error, messages, response.  Returns None if the file cannot be parsed.
    """
    try:
        text = path.read_text()
    except OSError:
        return None

    result: dict = {
        "request_id": None,
        "model": None,
        "api_key": None,
        "timestamp": None,
        "duration_ms": None,
        "acquire_ms": None,
        "query_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "error": None,
        "messages": [],
        "response": None,
    }

    for line in text.splitlines():
        stripped = line.strip()

        # Header fields
        if stripped.startswith("SESSION: "):
            result["request_id"] = stripped[len("SESSION: "):]
        elif stripped.startswith("MODEL: "):
            result["model"] = stripped[len("MODEL: "):]
        elif stripped.startswith("API_KEY: "):
            val = stripped[len("API_KEY: "):]
            result["api_key"] = None if val == "anonymous" else val
        elif stripped.startswith("TIMESTAMP: "):
            result["timestamp"] = stripped[len("TIMESTAMP: "):]
        elif stripped.startswith("Duration: "):
            m = re.match(r"Duration:\s*(\d+)ms", stripped)
            if m:
                result["duration_ms"] = int(m.group(1))
        elif stripped.startswith("Acquire: "):
            m = re.match(r"Acquire:\s*(\d+)ms", stripped)
            if m:
                result["acquire_ms"] = int(m.group(1))
        elif stripped.startswith("Query: "):
            m = re.match(r"Query:\s*(\d+)ms", stripped)
            if m:
                result["query_ms"] = int(m.group(1))
        elif stripped.startswith("Input tokens: "):
            m = re.match(r"Input tokens:\s*(\d+)", stripped)
            if m:
                result["input_tokens"] = int(m.group(1))
        elif stripped.startswith("Output tokens: "):
            m = re.match(r"Output tokens:\s*(\d+)", stripped)
            if m:
                result["output_tokens"] = int(m.group(1))
        elif "] ERROR: " in stripped:
            # Lines like [HH:MM:SS.mmm] ERROR: some error
            error_match = re.search(r"\] ERROR: (.+)$", stripped)
            if error_match:
                result["error"] = error_match.group(1)

    # Parse messages: lines like [user] Hello / [assistant] Hi
    msg_pattern = re.compile(r"^\[(\w+)\] (.+)$")
    in_messages = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Messages:":
            in_messages = True
            continue
        if in_messages:
            m = msg_pattern.match(stripped)
            if m:
                result["messages"].append({"role": m.group(1), "content": m.group(2)})
            elif stripped.startswith("Parameters:") or stripped == "":
                if stripped.startswith("Parameters:"):
                    in_messages = False

    # Parse full response: text between "Full response:" and "===" line
    resp_match = re.search(
        r"Full response:\n(.*?)(?=\n={3,})", text, re.DOTALL
    )
    if resp_match:
        result["response"] = resp_match.group(1)

    # Only return if we got at least a request_id
    if result["request_id"] is None:
        return None

    # Load attachment metadata if available
    request_id = result["request_id"]
    att_json = path.parent / f"{request_id}_attachments.json"
    if att_json.exists():
        try:
            att_meta = json.loads(att_json.read_text())
            # Group by msg_index
            by_msg: dict[int, list] = {}
            for entry in att_meta:
                idx = entry["msg_index"]
                by_msg.setdefault(idx, []).append(entry)
            # Attach to each message
            for i, msg in enumerate(result["messages"]):
                msg["attachments"] = by_msg.get(i, [])
        except Exception:
            pass

    return result


def _get_recent_logs(limit: int = 20) -> list[dict]:
    """Read recent session log files, newest first."""
    log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
    if not log_dir.exists():
        return []

    log_files = sorted(
        log_dir.glob("*.log"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    results = []
    for path in log_files[:limit]:
        parsed = _parse_log_file(path)
        if parsed is not None:
            results.append(parsed)

    return results


def create_dashboard_router(
    state: DashboardState,
    pool_status_fn: Callable[[], dict],
) -> APIRouter:
    """Create and return a dashboard APIRouter.

    Args:
        state: DashboardState instance for tracking active requests.
        pool_status_fn: Callable returning pool status dict with keys
            'size', 'available', 'in_use'.
    """
    router = APIRouter()

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page():
        """Serve the main dashboard page."""
        page_path = TEMPLATES_DIR / "page.html"
        return HTMLResponse(content=page_path.read_text())

    @router.get("/dashboard/pool", response_class=HTMLResponse)
    async def dashboard_pool(request: Request):
        """Render pool status fragment."""
        status = pool_status_fn()
        return templates.TemplateResponse(
            request,
            "pool.html",
            {
                "size": status.get("size", 0),
                "available": status.get("available", 0),
                "in_use": status.get("in_use", 0),
            },
        )

    def _get_merged_requests(limit: int = 20) -> list[dict]:
        """Merge active and completed requests into a single list.

        Active requests appear first (newest first), then completed (newest first).
        Each item has an ``is_active`` flag.
        """
        active = state.get_active_requests()
        for req in active:
            req["is_active"] = True
            req["api_key"] = _mask_api_key(req.get("api_key"))
        active.sort(key=lambda r: r["elapsed_s"])  # lowest elapsed = newest

        completed = _get_recent_logs(limit=limit)
        for log in completed:
            log["is_active"] = False
            log["api_key"] = _mask_api_key(log.get("api_key"))
            # Supplement with cached usage if log file didn't have it
            if log.get("input_tokens") is None:
                usage = state.get_usage(log["request_id"])
                if usage:
                    log["input_tokens"] = usage.get("prompt_tokens")
                    log["output_tokens"] = usage.get("completion_tokens")

        merged = active + completed
        return merged[:limit]

    @router.get("/dashboard/requests")
    async def dashboard_requests(request: Request):
        """SSE endpoint that pushes unified requests HTML on change."""

        async def event_stream():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    merged = _get_merged_requests()
                    rendered = templates.get_template("requests.html").render(
                        requests=merged
                    )
                    # SSE multi-line: each line must be prefixed with "data: "
                    lines = rendered.splitlines()
                    sse_data = "\n".join(f"data: {line}" for line in lines)
                    yield f"event: message\n{sse_data}\n\n"
                    await state.wait_for_change(timeout=2.0)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Error in requests SSE stream")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/dashboard/request/{request_id}", response_class=HTMLResponse)
    async def dashboard_request_detail(request_id: str, request: Request):
        """Render request detail — checks active state first, then log file."""
        # Check active requests first
        active_requests = state.get_active_requests()
        active = None
        for req in active_requests:
            if req["request_id"] == request_id:
                active = req
                break

        if active is not None:
            return templates.TemplateResponse(
                request,
                "detail.html",
                {
                    "request_id": request_id,
                    "model": active["model"],
                    "api_key": _mask_api_key(active.get("api_key")),
                    "timestamp": "",
                    "duration_ms": int(active["elapsed_s"] * 1000),
                    "acquire_ms": None,
                    "query_ms": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "error": None,
                    "is_active": True,
                    "buffered_text": "",
                    "messages": [],
                    "response": None,
                },
            )

        # Fall back to log file
        log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
        log_path = log_dir / f"{request_id}.log"
        parsed = _parse_log_file(log_path)
        if parsed is None:
            raise HTTPException(status_code=404, detail="Request not found")

        return templates.TemplateResponse(
            request,
            "detail.html",
            {
                "request_id": parsed["request_id"],
                "model": parsed["model"],
                "api_key": _mask_api_key(parsed.get("api_key")),
                "timestamp": parsed.get("timestamp", ""),
                "duration_ms": parsed.get("duration_ms", 0),
                "acquire_ms": parsed.get("acquire_ms"),
                "query_ms": parsed.get("query_ms"),
                "input_tokens": parsed.get("input_tokens"),
                "output_tokens": parsed.get("output_tokens"),
                "error": parsed.get("error"),
                "is_active": False,
                "buffered_text": "",
                "messages": parsed.get("messages", []),
                "response": parsed.get("response"),
            },
        )

    @router.get("/dashboard/attachment/{request_id}/{filename}")
    async def dashboard_attachment(request_id: str, filename: str):
        """Serve a saved attachment file."""
        # Validate request_id format
        if not re.fullmatch(r"chatcmpl-[a-f0-9]+", request_id):
            raise HTTPException(status_code=400, detail="Invalid request ID")
        # Validate filename — no path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
        file_path = log_dir / f"{request_id}_attachments" / filename
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Attachment not found")

        return FileResponse(file_path)

    @router.get("/dashboard/stream/{request_id}")
    async def dashboard_stream(request_id: str):
        """SSE endpoint for live token streaming."""
        queue = state.subscribe(request_id)
        if queue is None:
            raise HTTPException(status_code=404, detail="Request not active")

        async def event_stream():
            try:
                while True:
                    msg = await queue.get()
                    if msg["type"] == "chunk":
                        escaped = (
                            msg["text"]
                            .replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )
                        # SSE multi-line: prefix each line with "data: "
                        data_lines = "\n".join(f"data: {l}" for l in escaped.splitlines()) if "\n" in escaped else f"data: {escaped}"
                        yield f"event: chunk\n{data_lines}\n\n"
                    elif msg["type"] == "done":
                        yield "event: done\ndata: complete\n\n"
                        return
                    elif msg["type"] == "error":
                        yield f"event: error\ndata: {msg['error']}\n\n"
                        return
            finally:
                state.unsubscribe(request_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
