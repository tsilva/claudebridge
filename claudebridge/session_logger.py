"""Session logging for Claude requests."""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .image_utils import extract_attachments_from_messages, extract_text_from_content

logger = logging.getLogger(__name__)

# Maximum number of log files to keep
MAX_LOG_FILES = int(os.environ.get("MAX_LOG_FILES", 1000))


class SessionLogger:
    """Logs a single Claude request/response session to a plain text file."""

    def __init__(self, request_id: str, model: str):
        self.request_id = request_id
        self.model = model
        self.start_time = datetime.now(timezone.utc)
        self.chunks: list[tuple[datetime, str]] = []
        self.finish_reason: str | None = None
        self.error: str | None = None
        self.acquire_ms: int | None = None
        self.query_ms: int | None = None
        self.pool_snapshot: dict | None = None
        self.exception_type: str | None = None
        self.traceback_str: str | None = None

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

    def log_timing(self, acquire_ms: int, query_ms: int) -> None:
        """Record timing breakdown."""
        self.acquire_ms = acquire_ms
        self.query_ms = query_ms

    def log_error(self, error: str, *, exception_type: str | None = None,
                  traceback_str: str | None = None, pool_snapshot: dict | None = None) -> None:
        """Record an error with optional diagnostic details."""
        self.error = error
        if exception_type is not None:
            self.exception_type = exception_type
        if traceback_str is not None:
            self.traceback_str = traceback_str
        if pool_snapshot is not None:
            self.pool_snapshot = pool_snapshot

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
            if self.exception_type:
                lines.append(f"  Exception: {self.exception_type}")
            if self.traceback_str:
                lines.append(f"  Traceback:")
                for tb_line in self.traceback_str.strip().splitlines():
                    lines.append(f"    {tb_line}")

        lines.extend([
            "",
            "--- TIMING ---",
            f"Start: {self._format_time(self.start_time)}",
            f"End: {self._format_time(end_time)}",
            f"Duration: {duration_ms}ms",
        ])
        if self.acquire_ms is not None:
            lines.append(f"Acquire: {self.acquire_ms}ms")
        if self.query_ms is not None:
            lines.append(f"Query: {self.query_ms}ms")

        if self.pool_snapshot:
            lines.extend([
                "",
                "--- POOL STATE ---",
            ])
            for key, value in self.pool_snapshot.items():
                lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "--- COMPLETE ---",
            "Full response:",
            full_response,
            "=" * 80,
            "",
        ])

        with open(self.log_path, "w") as f:
            f.write("\n".join(lines))

        # Save attachments (images, PDFs) as binary files
        self._save_attachments(messages)

        self._cleanup_old_logs()

    def _save_attachments(self, messages: list) -> None:
        """Save binary attachments and metadata JSON alongside the log file."""
        try:
            attachments = extract_attachments_from_messages(messages)
            if not attachments:
                return

            att_dir = self.log_dir / f"{self.request_id}_attachments"
            att_dir.mkdir(parents=True, exist_ok=True)

            metadata = []
            for att in attachments:
                entry = {
                    "msg_index": att.msg_index,
                    "att_index": att.att_index,
                    "media_type": att.media_type,
                    "content_type": att.content_type,
                    "filename": att.filename,
                }
                if att.content_type == "base64" and att.data:
                    (att_dir / att.filename).write_bytes(att.data)
                elif att.content_type == "url" and att.url:
                    entry["url"] = att.url
                metadata.append(entry)

            meta_path = self.log_dir / f"{self.request_id}_attachments.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"[session_logger] Saved {len(attachments)} attachment(s) for {self.request_id}")
        except Exception as e:
            logger.warning(f"[session_logger] Failed to save attachments: {e}")

    def _cleanup_old_logs(self) -> None:
        """Delete oldest log files if count exceeds MAX_LOG_FILES."""
        try:
            log_files = sorted(
                self.log_dir.glob("*.log"),
                key=lambda f: f.stat().st_mtime,
            )
            if len(log_files) > MAX_LOG_FILES:
                to_delete = log_files[:len(log_files) - MAX_LOG_FILES]
                for f in to_delete:
                    # Also delete associated attachment dir and metadata
                    stem = f.stem
                    att_dir = self.log_dir / f"{stem}_attachments"
                    att_json = self.log_dir / f"{stem}_attachments.json"
                    if att_dir.is_dir():
                        shutil.rmtree(att_dir)
                    if att_json.exists():
                        att_json.unlink()
                    f.unlink()
                logger.info(f"[session_logger] Cleaned up {len(to_delete)} old log files")
        except Exception as e:
            logger.warning(f"[session_logger] Log cleanup failed: {e}")
