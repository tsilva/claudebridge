"""User configuration paths and .env loading for AgentBridge."""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR_ENV = "AGENTBRIDGE_CONFIG_DIR"
DEFAULT_ENV_CONTENT = """# AgentBridge local configuration
# Keep API keys on this machine only. Values in the process environment override this file.
OPENROUTER_API_KEY=
OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=agentbridge
"""

_loaded_env_path: Path | None = None


def user_config_dir(*, create: bool = False) -> Path:
    """Return the user config directory, defaulting to ~/.config/agentbridge."""
    raw = os.environ.get(CONFIG_DIR_ENV)
    path = Path(raw).expanduser() if raw else Path.home() / ".config" / "agentbridge"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def user_env_path(*, create_parent: bool = False) -> Path:
    """Return the AgentBridge .env path."""
    return user_config_dir(create=create_parent) / ".env"


def _write_default_env(path: Path) -> None:
    path.write_text(DEFAULT_ENV_CONTENT, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _unquote_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_user_env(*, create: bool = False) -> Path | None:
    """Load ~/.config/agentbridge/.env into os.environ without overriding existing values."""
    global _loaded_env_path

    path = user_env_path(create_parent=create)
    if create and not path.exists():
        _write_default_env(path)
    if not path.is_file():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        parsed_value = _unquote_env_value(value)
        if parsed_value:
            os.environ.setdefault(key, parsed_value)

    _loaded_env_path = path
    return path


def session_log_dir(*, create: bool = False) -> Path:
    """Return the session log directory, defaulting inside the AgentBridge config dir."""
    raw = os.environ.get("LOG_DIR")
    if raw:
        path = Path(os.path.expandvars(raw)).expanduser()
    else:
        path = user_config_dir(create=create) / "logs" / "sessions"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_user_config() -> Path:
    """Create the AgentBridge config directory, .env, and default log directory."""
    load_user_env(create=True)
    session_log_dir(create=True)
    return user_config_dir(create=True)
