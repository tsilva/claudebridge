"""In-memory state tracking for the dashboard."""

import asyncio
import time


class _ActiveRequest:
    """Tracks a single in-flight request."""

    __slots__ = ("request_id", "model", "start_time", "status", "chunks_received", "_subscribers")

    def __init__(self, request_id: str, model: str):
        self.request_id = request_id
        self.model = model
        self.start_time = time.monotonic()
        self.status = "active"
        self.chunks_received = 0
        self._subscribers: list[asyncio.Queue] = []

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "elapsed_s": round(time.monotonic() - self.start_time, 2),
            "status": self.status,
            "chunks_received": self.chunks_received,
        }


class DashboardState:
    """Tracks active requests so the dashboard can display them and stream tokens."""

    def __init__(self):
        self._active: dict[str, _ActiveRequest] = {}

    def request_started(self, request_id: str, model: str) -> None:
        self._active[request_id] = _ActiveRequest(request_id, model)

    def chunk_received(self, request_id: str, text: str) -> None:
        req = self._active.get(request_id)
        if req is None:
            return
        req.chunks_received += 1
        msg = {"type": "chunk", "text": text}
        for q in req._subscribers:
            q.put_nowait(msg)

    def request_completed(self, request_id: str) -> None:
        req = self._active.pop(request_id, None)
        if req is None:
            return
        for q in req._subscribers:
            q.put_nowait({"type": "done"})

    def request_errored(self, request_id: str, error: str) -> None:
        req = self._active.pop(request_id, None)
        if req is None:
            return
        for q in req._subscribers:
            q.put_nowait({"type": "error", "error": error})

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
