"""Tests for AgentBridge user configuration handling."""

import os

import pytest

from agentbridge import config

pytestmark = pytest.mark.unit


def reset_config_loader():
    config._loaded_env_path = None


def test_ensure_user_config_creates_env_and_default_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("LOG_DIR", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_NAME", raising=False)
    reset_config_loader()

    config_dir = config.ensure_user_config()

    assert config_dir == tmp_path
    assert (tmp_path / ".env").is_file()
    assert "OPENROUTER_API_KEY=" in (tmp_path / ".env").read_text()
    assert (tmp_path / "logs" / "sessions").is_dir()
    assert config.session_log_dir() == tmp_path / "logs" / "sessions"
    assert "OPENROUTER_API_KEY" not in os.environ
    assert os.environ["OPENROUTER_APP_NAME"] == "agentbridge"


def test_load_user_env_preserves_process_environment(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_API_KEY=from-file\nOPENROUTER_APP_NAME=file-app\n")
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-process")
    monkeypatch.delenv("OPENROUTER_APP_NAME", raising=False)
    reset_config_loader()

    config.load_user_env()

    assert os.environ["OPENROUTER_API_KEY"] == "from-process"
    assert os.environ["OPENROUTER_APP_NAME"] == "file-app"


def test_session_log_dir_respects_log_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path / "config"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "custom-logs"))

    assert config.session_log_dir(create=True) == tmp_path / "custom-logs"
    assert (tmp_path / "custom-logs").is_dir()
