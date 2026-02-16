"""Dashboard state tracking and FastAPI routes."""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

class _ActiveRequest:
    """Tracks a single in-flight request."""

    __slots__ = ("request_id", "model", "api_key", "start_time", "status", "chunks_received", "messages", "buffered_text", "_subscribers")

    def __init__(self, request_id: str, model: str, api_key: str | None = None, messages: list[dict] | None = None):
        self.request_id = request_id
        self.model = model
        self.api_key = api_key
        self.start_time = time.monotonic()
        self.status = "active"
        self.chunks_received = 0
        self.messages = messages or []
        self.buffered_text = ""
        self._subscribers: list[asyncio.Queue] = []

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "api_key": self.api_key,
            "elapsed_s": round(time.monotonic() - self.start_time, 2),
            "status": self.status,
            "chunks_received": self.chunks_received,
            "messages": self.messages,
            "buffered_text": self.buffered_text,
        }


class DashboardState:
    """Tracks active requests so the dashboard can display them and stream tokens."""

    # How many recently completed requests to keep usage data for
    _RECENT_LIMIT = 50

    def __init__(self):
        self._active: dict[str, _ActiveRequest] = {}
        self._change_event = asyncio.Event()
        self._pool_change_event = asyncio.Event()
        self._recent_usage: dict[str, dict[str, int]] = {}
        self._recent_order: list[str] = []

    def _notify(self) -> None:
        """Signal that the active requests list has changed."""
        self._change_event.set()

    def notify_pool_change(self) -> None:
        """Signal that pool status has changed."""
        self._pool_change_event.set()

    async def wait_for_pool_change(self, timeout: float = 5.0) -> None:
        """Wait for a pool change notification or timeout, then clear the event."""
        try:
            await asyncio.wait_for(self._pool_change_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        self._pool_change_event.clear()

    async def wait_for_change(self, timeout: float = 2.0) -> None:
        """Wait for a change notification or timeout, then clear the event."""
        try:
            await asyncio.wait_for(self._change_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        self._change_event.clear()

    def request_started(self, request_id: str, model: str, api_key: str | None = None, messages: list[dict] | None = None) -> None:
        self._active[request_id] = _ActiveRequest(request_id, model, api_key=api_key, messages=messages)
        self._notify()

    def chunk_received(self, request_id: str, text: str) -> None:
        req = self._active.get(request_id)
        if req is None:
            return
        req.chunks_received += 1
        req.buffered_text += text
        msg = {"type": "chunk", "text": text}
        for q in req._subscribers:
            q.put_nowait(msg)

    def request_completed(self, request_id: str, usage: dict | None = None) -> None:
        req = self._active.pop(request_id, None)
        if req is None:
            return
        if usage:
            self._recent_usage[request_id] = usage
            self._recent_order.append(request_id)
            # Evict old entries
            while len(self._recent_order) > self._RECENT_LIMIT:
                old_id = self._recent_order.pop(0)
                self._recent_usage.pop(old_id, None)
        for q in req._subscribers:
            q.put_nowait({"type": "done"})
        self._notify()

    def get_usage(self, request_id: str) -> dict[str, int] | None:
        """Return cached usage for a recently completed request."""
        return self._recent_usage.get(request_id)

    def request_errored(self, request_id: str, error: str) -> None:
        req = self._active.pop(request_id, None)
        if req is None:
            return
        for q in req._subscribers:
            q.put_nowait({"type": "error", "error": error})
        self._notify()

    def get_active_requests(self) -> list[dict]:
        return [r.to_dict() for r in self._active.values()]

    def subscribe(self, request_id: str) -> asyncio.Queue | None:
        req = self._active.get(request_id)
        if req is None:
            return None
        q: asyncio.Queue = asyncio.Queue()
        req._subscribers.append(q)
        return q

    def unsubscribe(self, request_id: str, queue: asyncio.Queue) -> None:
        req = self._active.get(request_id)
        if req is None:
            return
        try:
            req._subscribers.remove(queue)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

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
    """Parse a session log file (JSON format) into a structured dict.

    Returns dict with session data or None if the file cannot be parsed.
    Only accepts dicts with a ``request_id`` key (skips attachment manifests, etc.).
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "request_id" not in data:
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def _get_recent_logs(limit: int = 20) -> list[dict]:
    """Read recent session log files, newest first."""
    log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
    if not log_dir.exists():
        return []

    log_files = sorted(
        log_dir.glob("*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    results = []
    for path in log_files[:limit]:
        parsed = _parse_log_file(path)
        if parsed is not None:
            results.append(parsed)

    return results


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

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

    @router.get("/dashboard/pool/stream")
    async def dashboard_pool_stream(request: Request):
        """SSE endpoint that pushes pool status HTML on change."""

        async def event_stream():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    status = pool_status_fn()
                    rendered = templates.get_template("pool.html").render(
                        size=status.get("size", 0),
                        available=status.get("available", 0),
                        in_use=status.get("in_use", 0),
                    )
                    lines = rendered.splitlines()
                    sse_data = "\n".join(f"data: {line}" for line in lines)
                    yield f"event: message\n{sse_data}\n\n"
                    await state.wait_for_pool_change(timeout=5.0)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Error in pool SSE stream")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
                usage_data = log.get("usage")
                if usage_data:
                    log["input_tokens"] = usage_data.get("input_tokens")
                    log["output_tokens"] = usage_data.get("output_tokens")
                else:
                    usage = state.get_usage(log.get("request_id", ""))
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
                    "buffered_text": active.get("buffered_text", ""),
                    "messages": active.get("messages", []),
                    "response": None,
                },
            )

        # Fall back to log file
        log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
        log_path = log_dir / f"{request_id}.json"
        parsed = _parse_log_file(log_path)
        if parsed is None:
            raise HTTPException(status_code=404, detail="Request not found")

        timing = parsed.get("timing", {})
        usage = parsed.get("usage", {})
        return templates.TemplateResponse(
            request,
            "detail.html",
            {
                "request_id": parsed.get("request_id", request_id),
                "model": parsed.get("model"),
                "api_key": _mask_api_key(parsed.get("api_key")),
                "timestamp": parsed.get("timestamp", ""),
                "duration_ms": timing.get("duration_ms", 0),
                "acquire_ms": timing.get("acquire_ms"),
                "query_ms": timing.get("query_ms"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
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
