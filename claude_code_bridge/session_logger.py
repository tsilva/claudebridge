"""Session logging for Claude requests."""

import os
from datetime import datetime, timezone
from pathlib import Path

from .image_utils import extract_text_from_content


class SessionLogger:
    """Logs a single Claude request/response session to a plain text file."""

    def __init__(self, request_id: str, model: str):
        self.request_id = request_id
        self.model = model
        self.start_time = datetime.now(timezone.utc)
        self.chunks: list[tuple[datetime, str]] = []
        self.finish_reason: str | None = None
        self.error: str | None = None

        # Ensure log directory exists
        self.log_dir = Path(os.environ.get("LOG_DIR", "logs/sessions"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{request_id}.log"

    def _format_time(self, dt: datetime) -> str:
        """Format datetime for display."""
        return dt.strftime("%H:%M:%S.%f")[:-3]

    def _format_timestamp(self, dt: datetime) -> str:
        """Format datetime as ISO timestamp."""
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def log_chunk(self, content: str) -> None:
        """Record a streaming chunk with timestamp."""
        self.chunks.append((datetime.now(timezone.utc), content))

    def log_finish(self, reason: str) -> None:
        """Record the finish reason."""
        self.finish_reason = reason

    def log_error(self, error: str) -> None:
        """Record an error."""
        self.error = error

    def write(self, messages: list, stream: bool, temperature: float | None, max_tokens: int | None) -> None:
        """Write the complete session log to file."""
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - self.start_time).total_seconds() * 1000)
        full_response = "".join(content for _, content in self.chunks)

        lines = [
            "=" * 80,
            f"SESSION: {self.request_id}",
            f"TIMESTAMP: {self._format_timestamp(self.start_time)}",
            f"MODEL: {self.model}",
            "=" * 80,
            "",
            "--- REQUEST ---",
            "Messages:",
        ]

        # Format messages (handle both text-only and multimodal)
        for msg in messages:
            role = msg.role
            content = extract_text_from_content(msg.content)
            lines.append(f"[{role}] {content}")

        lines.extend([
            "",
            "Parameters:",
            f"  stream: {stream}",
            f"  temperature: {temperature}",
            f"  max_tokens: {max_tokens}",
            "",
            "--- RESPONSE ---",
        ])

        # Format chunks with timestamps
        if stream and self.chunks:
            for chunk_time, content in self.chunks:
                # Escape newlines for readability
                escaped = content.replace("\n", "\\n")
                lines.append(f"[{self._format_time(chunk_time)}] CHUNK: {escaped}")
        elif self.chunks:
            # Non-streaming: just show the full response
            lines.append(f"[{self._format_time(self.chunks[0][0])}] RESPONSE: {len(full_response)} chars")

        if self.finish_reason:
            lines.append(f"[{self._format_time(end_time)}] FINISH: {self.finish_reason}")

        if self.error:
            lines.append(f"[{self._format_time(end_time)}] ERROR: {self.error}")

        lines.extend([
            "",
            "--- TIMING ---",
            f"Start: {self._format_time(self.start_time)}",
            f"End: {self._format_time(end_time)}",
            f"Duration: {duration_ms}ms",
            "",
            "--- COMPLETE ---",
            "Full response:",
            full_response,
            "=" * 80,
            "",
        ])

        with open(self.log_path, "w") as f:
            f.write("\n".join(lines))
