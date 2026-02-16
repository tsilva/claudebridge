"""
Unit tests for session logger (JSON format).

Usage:
- pytest tests/test_session_logger.py -v
"""

import json
import os
import time
from datetime import datetime
from unittest.mock import patch

import pytest

from claudebridge.server import SessionLogger, MAX_LOG_FILES
from claudebridge.models import Message


@pytest.mark.unit
class TestSessionLoggerInit:
    """Tests for SessionLogger initialization."""

    def test_creates_log_directory(self, tmp_path):
        """Log directory is created if it doesn't exist."""
        log_dir = tmp_path / "test_logs" / "sessions"
        os.environ["LOG_DIR"] = str(log_dir)
        try:
            logger = SessionLogger("test-123", "sonnet")
            assert log_dir.exists()
        finally:
            del os.environ["LOG_DIR"]

    def test_log_path_uses_request_id(self, tmp_path):
        """Log file path includes request ID with .json extension."""
        os.environ["LOG_DIR"] = str(tmp_path)
        try:
            logger = SessionLogger("chatcmpl-abc123", "sonnet")
            assert logger.log_path.name == "chatcmpl-abc123.json"
        finally:
            del os.environ["LOG_DIR"]

    def test_initial_state(self, tmp_path):
        """Logger has clean initial state."""
        os.environ["LOG_DIR"] = str(tmp_path)
        try:
            logger = SessionLogger("test-123", "opus")
            assert logger.request_id == "test-123"
            assert logger.model == "opus"
            assert logger.chunks == []
            assert logger.finish_reason is None
            assert logger.error is None
        finally:
            del os.environ["LOG_DIR"]


@pytest.mark.unit
class TestSessionLoggerOperations:
    """Tests for logging operations."""

    @pytest.fixture(autouse=True)
    def setup_log_dir(self, tmp_path):
        """Set up temp log directory."""
        os.environ["LOG_DIR"] = str(tmp_path)
        yield
        del os.environ["LOG_DIR"]

    def test_log_chunk(self):
        """Chunks are recorded with timestamps."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_chunk("Hello ")
        logger.log_chunk("World")
        assert len(logger.chunks) == 2
        assert logger.chunks[0][1] == "Hello "
        assert logger.chunks[1][1] == "World"
        assert isinstance(logger.chunks[0][0], datetime)

    def test_log_finish(self):
        """Finish reason is recorded."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_finish("stop")
        assert logger.finish_reason == "stop"

    def test_log_finish_tool_calls(self):
        """Tool calls finish reason."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_finish("tool_calls")
        assert logger.finish_reason == "tool_calls"

    def test_log_error(self):
        """Errors are recorded."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_error("Timeout after 120s")
        assert logger.error == "Timeout after 120s"


@pytest.mark.unit
class TestSessionLoggerWrite:
    """Tests for writing session logs to JSON file."""

    @pytest.fixture(autouse=True)
    def setup_log_dir(self, tmp_path):
        """Set up temp log directory."""
        self.log_dir = tmp_path
        os.environ["LOG_DIR"] = str(tmp_path)
        yield
        del os.environ["LOG_DIR"]

    def _read_log(self, logger):
        """Read and parse the JSON log file."""
        return json.loads(logger.log_path.read_text())

    def test_write_creates_json_file(self):
        """Write creates JSON log file."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_chunk("Hello")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Say hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        assert logger.log_path.exists()
        assert logger.log_path.suffix == ".json"

    def test_write_valid_json(self):
        """Log file is valid JSON."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_chunk("Hello")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Say hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        data = self._read_log(logger)
        assert isinstance(data, dict)

    def test_write_contains_session_info(self):
        """JSON log contains session information."""
        logger = SessionLogger("test-abc", "opus")
        logger.log_chunk("Response text")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Test prompt")]
        logger.write(messages, stream=False, temperature=0.7, max_tokens=100)

        data = self._read_log(logger)
        assert data["request_id"] == "test-abc"
        assert data["model"] == "opus"
        assert data["parameters"]["temperature"] == 0.7
        assert data["parameters"]["max_tokens"] == 100

    def test_write_contains_messages(self):
        """JSON log contains formatted messages."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_chunk("Response")
        logger.log_finish("stop")
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        data = self._read_log(logger)
        assert len(data["messages"]) == 3
        assert data["messages"][0]["role"] == "system"
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"] == "Hello"

    def test_write_contains_response(self):
        """JSON log contains full response text."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_chunk("Hello ")
        logger.log_chunk("World!")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Greet me")]
        logger.write(messages, stream=True, temperature=None, max_tokens=None)

        data = self._read_log(logger)
        assert data["response"] == "Hello World!"
        assert data["finish_reason"] == "stop"

    def test_write_with_error(self):
        """JSON log contains error information."""
        logger = SessionLogger("test-err", "sonnet")
        logger.log_error("Connection timeout")
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        data = self._read_log(logger)
        assert data["error"] == "Connection timeout"

    def test_write_with_timing(self):
        """JSON log includes timing breakdown."""
        logger = SessionLogger("test-timing", "opus")
        logger.log_chunk("Response")
        logger.log_finish("stop")
        logger.log_timing(acquire_ms=45, query_ms=3200)
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        data = self._read_log(logger)
        assert data["timing"]["acquire_ms"] == 45
        assert data["timing"]["query_ms"] == 3200
        assert "duration_ms" in data["timing"]

    def test_write_with_usage(self):
        """JSON log includes token usage."""
        logger = SessionLogger("test-usage", "opus")
        logger.log_chunk("Response")
        logger.log_finish("stop")
        logger.log_usage(100, 50)
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        data = self._read_log(logger)
        assert data["usage"]["input_tokens"] == 100
        assert data["usage"]["output_tokens"] == 50

    def test_write_timestamp_format(self):
        """Timestamp is ISO format."""
        logger = SessionLogger("test-ts", "sonnet")
        logger.log_chunk("Response")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        data = self._read_log(logger)
        assert data["timestamp"].endswith("Z")
        assert "T" in data["timestamp"]


@pytest.mark.unit
class TestSessionLoggerCleanup:
    """Tests for log file cleanup."""

    @pytest.fixture(autouse=True)
    def setup_log_dir(self, tmp_path):
        """Set up temp log directory."""
        self.log_dir = tmp_path
        os.environ["LOG_DIR"] = str(tmp_path)
        yield
        del os.environ["LOG_DIR"]

    def test_cleanup_deletes_oldest_files(self):
        """Cleanup removes oldest files when over limit."""
        for i in range(5):
            p = self.log_dir / f"old-{i}.json"
            p.write_text(json.dumps({"request_id": f"old-{i}"}))
            os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))

        with patch("claudebridge.server.MAX_LOG_FILES", 3):
            logger = SessionLogger("test-cleanup", "sonnet")
            logger.log_chunk("Response")
            logger.log_finish("stop")
            messages = [Message(role="user", content="Hi")]
            logger.write(messages, stream=False, temperature=None, max_tokens=None)

        remaining = list(self.log_dir.glob("*.json"))
        assert len(remaining) == 3
        assert not (self.log_dir / "old-0.json").exists()
        assert not (self.log_dir / "old-1.json").exists()
        assert not (self.log_dir / "old-2.json").exists()

    def test_no_cleanup_when_under_limit(self):
        """No cleanup when file count is under limit."""
        for i in range(2):
            (self.log_dir / f"existing-{i}.json").write_text(
                json.dumps({"request_id": f"existing-{i}"})
            )

        with patch("claudebridge.server.MAX_LOG_FILES", 100):
            logger = SessionLogger("test-no-cleanup", "sonnet")
            logger.log_chunk("Response")
            logger.log_finish("stop")
            messages = [Message(role="user", content="Hi")]
            logger.write(messages, stream=False, temperature=None, max_tokens=None)

        remaining = list(self.log_dir.glob("*.json"))
        assert len(remaining) == 3  # 2 existing + 1 new
