"""Tests for URL utilities."""

import os
import socket
from unittest.mock import patch

import pytest

from claude_code_bridge.url_utils import resolve_bridge_url

# Patch target for socket.gethostbyname in the url_utils module
SOCKET_PATCH = "claude_code_bridge.url_utils.socket.gethostbyname"


@pytest.mark.unit
class TestResolveBridgeUrl:
    """Tests for resolve_bridge_url function."""

    def test_default_url_outside_container(self):
        """Outside container, returns localhost URL."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(SOCKET_PATCH, side_effect=socket.gaierror):
                result = resolve_bridge_url()
                assert result == "http://localhost:8082"

    def test_custom_default(self):
        """Custom default is respected."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(SOCKET_PATCH, side_effect=socket.gaierror):
                result = resolve_bridge_url("http://localhost:9999")
                assert result == "http://localhost:9999"

    def test_env_var_non_localhost(self):
        """Non-localhost BRIDGE_URL is returned as-is."""
        with patch.dict(os.environ, {"BRIDGE_URL": "http://example.com:9000"}):
            result = resolve_bridge_url()
            assert result == "http://example.com:9000"

    def test_env_var_localhost_outside_container(self):
        """Localhost BRIDGE_URL stays localhost outside container."""
        with patch.dict(os.environ, {"BRIDGE_URL": "http://localhost:3000"}):
            with patch(SOCKET_PATCH, side_effect=socket.gaierror):
                result = resolve_bridge_url()
                assert result == "http://localhost:3000"

    def test_env_var_127_outside_container(self):
        """127.0.0.1 BRIDGE_URL stays 127.0.0.1 outside container."""
        with patch.dict(os.environ, {"BRIDGE_URL": "http://127.0.0.1:3000"}):
            with patch(SOCKET_PATCH, side_effect=socket.gaierror):
                result = resolve_bridge_url()
                assert result == "http://127.0.0.1:3000"

    def test_localhost_inside_container(self):
        """Localhost is swapped to host.docker.internal inside container."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(SOCKET_PATCH, return_value="192.168.65.2"):
                result = resolve_bridge_url()
                assert result == "http://host.docker.internal:8082"

    def test_127_inside_container(self):
        """127.0.0.1 is swapped to host.docker.internal inside container."""
        with patch.dict(os.environ, {"BRIDGE_URL": "http://127.0.0.1:9000"}):
            with patch(SOCKET_PATCH, return_value="192.168.65.2"):
                result = resolve_bridge_url()
                assert result == "http://host.docker.internal:9000"

    def test_non_localhost_inside_container(self):
        """Non-localhost URLs are not modified even inside container."""
        with patch.dict(os.environ, {"BRIDGE_URL": "http://myserver.local:8000"}):
            # Even if host.docker.internal resolves, non-localhost URLs stay unchanged
            with patch(SOCKET_PATCH, return_value="192.168.65.2"):
                result = resolve_bridge_url()
                assert result == "http://myserver.local:8000"

    def test_localhost_no_port(self):
        """Localhost without port is handled correctly."""
        with patch.dict(os.environ, {"BRIDGE_URL": "http://localhost"}):
            with patch(SOCKET_PATCH, return_value="192.168.65.2"):
                result = resolve_bridge_url()
                assert result == "http://host.docker.internal"
