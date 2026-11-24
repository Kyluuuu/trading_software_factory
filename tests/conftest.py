"""Shared deterministic pytest configuration."""

from random import seed


def pytest_sessionstart() -> None:
    seed(0)
