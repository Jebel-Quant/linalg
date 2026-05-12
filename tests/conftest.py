"""Pytest configuration and fixtures for the linalg test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def resource_dir():
    """Provide the path to the test resources directory.

    Returns:
        pathlib.Path: Path to the test resources directory

    """
    return Path(__file__).parent / "resources"
