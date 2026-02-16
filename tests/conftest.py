"""Pytest configuration for claudebridge tests."""

import os
from pathlib import Path

import pytest
import httpx

SERVER_URL = os.environ.get("BRIDGE_URL", "http://localhost:8082")
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _generate_pdf_fixture():
    """Generate PDF test fixture if it doesn't exist."""
    pdf_path = FIXTURES_DIR / "pdf_test_document.pdf"
    if pdf_path.exists():
        return

    try:
        from .generate_pdf_fixture import generate_test_pdf
    except ImportError:
        # fpdf2 not installed, skip generation
        return

    FIXTURES_DIR.mkdir(exist_ok=True)
    generate_test_pdf(pdf_path)


def pytest_configure(config):
    """Register custom markers and generate fixtures."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (no server required)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires running server)"
    )
    # Generate PDF fixture if needed
    _generate_pdf_fixture()


def pytest_collection_modifyitems(config, items):
    """Check server availability unless running unit tests only."""
    # Check if running only unit tests via marker expression
    markexpr = config.option.markexpr
    if markexpr and "unit" in markexpr:
        return  # Skip server check when running unit tests

    # Check if all collected tests are unit tests
    all_unit = all(item.get_closest_marker("unit") for item in items)
    if all_unit:
        return  # Skip server check for pure unit test runs

    try:
        response = httpx.get(f"{SERVER_URL}/health", timeout=5.0)
        if response.status_code != 200 or response.json().get("status") != "ok":
            pytest.exit(
                f"\n\nServer not responding correctly at {SERVER_URL}\n"
                f"Start the server with: claudebridge\n",
                returncode=1
            )
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.exit(
            f"\n\nServer not available at {SERVER_URL}\n"
            f"Start the server with: claudebridge\n",
            returncode=1
        )
