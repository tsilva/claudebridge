"""URL utilities for container-aware host resolution."""

import os
import socket
from urllib.parse import urlparse, urlunparse


def resolve_bridge_url(default: str = "http://localhost:8082") -> str:
    """
    Get bridge URL with automatic container detection.

    - Returns BRIDGE_URL env var if set and not localhost
    - For localhost/127.0.0.1, checks if running in container
      and swaps to host.docker.internal if so
    """
    url = os.environ.get("BRIDGE_URL", default)
    parsed = urlparse(url)

    # If not localhost, return as-is
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        return url

    # Check if host.docker.internal resolves (indicates container)
    try:
        socket.gethostbyname("host.docker.internal")
    except socket.gaierror:
        return url  # Not in container, keep localhost

    # Replace host with host.docker.internal
    new_netloc = f"host.docker.internal:{parsed.port}" if parsed.port else "host.docker.internal"
    return urlunparse(parsed._replace(netloc=new_netloc))
