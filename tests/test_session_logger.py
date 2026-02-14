"""
Unit tests for session logger.

Usage:
- pytest tests/test_session_logger.py -v
"""

import os
import time
from datetime import datetime
from unittest.mock import patch

import pytest

from claudebridge.session_logger import SessionLogger
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
        """Log file path includes request ID."""
        os.environ["LOG_DIR"] = str(tmp_path)
        try:
            logger = SessionLogger("chatcmpl-abc123", "sonnet")
            assert logger.log_path.name == "chatcmpl-abc123.log"
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
        # Timestamps should be UTC datetimes
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
    """Tests for writing session logs to file."""

    @pytest.fixture(autouse=True)
    def setup_log_dir(self, tmp_path):
        """Set up temp log directory."""
        self.log_dir = tmp_path
        os.environ["LOG_DIR"] = str(tmp_path)
        yield
        del os.environ["LOG_DIR"]

    def test_write_creates_file(self):
        """Write creates log file."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_chunk("Hello")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Say hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        assert logger.log_path.exists()

    def test_write_contains_session_info(self):
        """Log file contains session information."""
        logger = SessionLogger("test-abc", "opus")
        logger.log_chunk("Response text")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Test prompt")]
        logger.write(messages, stream=False, temperature=0.7, max_tokens=100)

        content = logger.log_path.read_text()
        assert "test-abc" in content
        assert "opus" in content
        assert "Test prompt" in content
        assert "temperature: 0.7" in content
        assert "max_tokens: 100" in content

    def test_write_contains_response(self):
        """Log file contains full response."""
        logger = SessionLogger("test-123", "sonnet")
        logger.log_chunk("Hello ")
        logger.log_chunk("World!")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Greet me")]
        logger.write(messages, stream=True, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "Hello World!" in content
        assert "FINISH: stop" in content

    def test_write_with_error(self):
        """Log file contains error information."""
        logger = SessionLogger("test-err", "sonnet")
        logger.log_error("Connection timeout")
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "Connection timeout" in content

    def test_write_streaming_shows_chunks(self):
        """Streaming log shows individual chunks."""
        logger = SessionLogger("test-stream", "sonnet")
        logger.log_chunk("Part 1")
        logger.log_chunk("Part 2")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Test")]
        logger.write(messages, stream=True, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "CHUNK:" in content

    def test_write_non_streaming_shows_length(self):
        """Non-streaming log shows response length."""
        logger = SessionLogger("test-nonstream", "sonnet")
        logger.log_chunk("Complete response here")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Test")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "RESPONSE:" in content
        assert "chars" in content

    def test_write_multiple_messages(self):
        """Log file handles multi-turn conversation."""
        logger = SessionLogger("test-multi", "sonnet")
        logger.log_chunk("Response")
        logger.log_finish("stop")
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="How are you?"),
        ]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "[system]" in content
        assert "[user]" in content
        assert "[assistant]" in content


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
        # Create 5 existing log files with staggered mtimes
        for i in range(5):
            p = self.log_dir / f"old-{i}.log"
            p.write_text(f"log {i}")
            # Ensure distinct modification times
            os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))

        # Set low limit
        with patch("claudebridge.session_logger.MAX_LOG_FILES", 3):
            logger = SessionLogger("test-cleanup", "sonnet")
            logger.log_chunk("Response")
            logger.log_finish("stop")
            messages = [Message(role="user", content="Hi")]
            logger.write(messages, stream=False, temperature=None, max_tokens=None)

        # Should have 3 files total (limit), keeping newest
        remaining = list(self.log_dir.glob("*.log"))
        assert len(remaining) == 3
        # The two oldest should be gone
        assert not (self.log_dir / "old-0.log").exists()
        assert not (self.log_dir / "old-1.log").exists()
        assert not (self.log_dir / "old-2.log").exists()

    def test_no_cleanup_when_under_limit(self):
        """No cleanup when file count is under limit."""
        # Create 2 existing files
        for i in range(2):
            (self.log_dir / f"existing-{i}.log").write_text(f"log {i}")

        with patch("claudebridge.session_logger.MAX_LOG_FILES", 100):
            logger = SessionLogger("test-no-cleanup", "sonnet")
            logger.log_chunk("Response")
            logger.log_finish("stop")
            messages = [Message(role="user", content="Hi")]
            logger.write(messages, stream=False, temperature=None, max_tokens=None)

        # All files should remain
        remaining = list(self.log_dir.glob("*.log"))
        assert len(remaining) == 3  # 2 existing + 1 new


@pytest.mark.unit
class TestSessionLoggerTimingBreakdown:
    """Tests for timing breakdown in session logs."""

    @pytest.fixture(autouse=True)
    def setup_log_dir(self, tmp_path):
        """Set up temp log directory."""
        self.log_dir = tmp_path
        os.environ["LOG_DIR"] = str(tmp_path)
        yield
        del os.environ["LOG_DIR"]

    def test_write_with_timing_breakdown(self):
        """Log file includes acquire and query timing when set."""
        logger = SessionLogger("test-timing", "opus")
        logger.log_chunk("Response")
        logger.log_finish("stop")
        logger.log_timing(acquire_ms=45, query_ms=3200)
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "Acquire: 45ms" in content
        assert "Query: 3200ms" in content

    def test_write_without_timing_breakdown(self):
        """Log file omits acquire/query lines when not set."""
        logger = SessionLogger("test-no-timing", "opus")
        logger.log_chunk("Response")
        logger.log_finish("stop")
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "Acquire:" not in content
        assert "Query:" not in content


@pytest.mark.unit
class TestSessionLoggerPoolSnapshot:
    """Tests for pool snapshot in session logs."""

    @pytest.fixture(autouse=True)
    def setup_log_dir(self, tmp_path):
        """Set up temp log directory."""
        self.log_dir = tmp_path
        os.environ["LOG_DIR"] = str(tmp_path)
        yield
        del os.environ["LOG_DIR"]

    def test_write_with_pool_snapshot(self):
        """Log file includes POOL STATE section when snapshot is present."""
        logger = SessionLogger("test-snap", "opus")
        logger.log_error(
            "Timeout after 120s",
            pool_snapshot={"size": 3, "in_use": 1, "available": 2, "available_models": ["opus", "opus"], "all_models": ["opus", "opus", "opus"]},
        )
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "--- POOL STATE ---" in content
        assert "size: 3" in content
        assert "in_use: 1" in content

    def test_write_without_pool_snapshot(self):
        """Log file omits POOL STATE section when no snapshot."""
        logger = SessionLogger("test-no-snap", "opus")
        logger.log_error("Some error")
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "POOL STATE" not in content


@pytest.mark.unit
class TestSessionLoggerErrorDetails:
    """Tests for exception details in session logs."""

    @pytest.fixture(autouse=True)
    def setup_log_dir(self, tmp_path):
        """Set up temp log directory."""
        self.log_dir = tmp_path
        os.environ["LOG_DIR"] = str(tmp_path)
        yield
        del os.environ["LOG_DIR"]

    def test_log_error_with_exception_details(self):
        """Log file includes exception type and traceback when provided."""
        logger = SessionLogger("test-exc", "opus")
        logger.log_error(
            "connection refused",
            exception_type="ConnectionError",
            traceback_str="Traceback (most recent call last):\n  File \"server.py\", line 10\nConnectionError: refused",
        )
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "Exception: ConnectionError" in content
        assert "Traceback:" in content
        assert "ConnectionError: refused" in content

    def test_log_error_without_exception_details(self):
        """Log file omits exception details when not provided."""
        logger = SessionLogger("test-no-exc", "opus")
        logger.log_error("simple error")
        messages = [Message(role="user", content="Hello")]
        logger.write(messages, stream=False, temperature=None, max_tokens=None)

        content = logger.log_path.read_text()
        assert "simple error" in content
        assert "Exception:" not in content
        assert "Traceback:" not in content
