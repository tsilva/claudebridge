"""In-memory state tracking for the dashboard."""

import asyncio
import time


class _ActiveRequest:
    """Tracks a single in-flight request."""

    __slots__ = ("request_id", "model", "api_key", "start_time", "status", "chunks_received", "_subscribers")

    def __init__(self, request_id: str, model: str, api_key: str | None = None):
        self.request_id = request_id
        self.model = model
        self.api_key = api_key
        self.start_time = time.monotonic()
        self.status = "active"
        self.chunks_received = 0
        self._subscribers: list[asyncio.Queue] = []

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "api_key": self.api_key,
            "elapsed_s": round(time.monotonic() - self.start_time, 2),
            "status": self.status,
            "chunks_received": self.chunks_received,
        }


class DashboardState:
    """Tracks active requests so the dashboard can display them and stream tokens."""

    # How many recently completed requests to keep usage data for
    _RECENT_LIMIT = 50

    def __init__(self):
        self._active: dict[str, _ActiveRequest] = {}
        self._change_event = asyncio.Event()
        self._recent_usage: dict[str, dict[str, int]] = {}
        self._recent_order: list[str] = []

    def _notify(self) -> None:
        """Signal that the active requests list has changed."""
        self._change_event.set()

    async def wait_for_change(self, timeout: float = 2.0) -> None:
        """Wait for a change notification or timeout, then clear the event."""
        try:
            await asyncio.wait_for(self._change_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        self._change_event.clear()

    def request_started(self, request_id: str, model: str, api_key: str | None = None) -> None:
        self._active[request_id] = _ActiveRequest(request_id, model, api_key=api_key)
        self._notify()

    def chunk_received(self, request_id: str, text: str) -> None:
        req = self._active.get(request_id)
        if req is None:
            return
        req.chunks_received += 1
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
