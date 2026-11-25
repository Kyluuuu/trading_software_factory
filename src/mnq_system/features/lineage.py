"""Deterministic lineage for the feature formula and generation policy."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from mnq_system.features.metadata import DEFINITIONS, VERSION

GENERATOR_VERSION = "feature-engine-2.5.0"
FEATURE_CONFIGURATION: dict[str, str] = {
    "feed": "alpaca-iex-raw-1min",
    "calendar": "XNYS",
    "availability": "source-minute-plus-1min-theoretical",
    "gap_policy": "reset-rolling-retain-session-vwap-observed-only",
    "month_policy": "reset-history",
    "vwap": "observed-typical-price-volume-weighted",
}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def formula_digest() -> str:
    """Include actual generator code as well as definitions, not catalog alone."""
    catalog = json.dumps([asdict(d) for d in DEFINITIONS], sort_keys=True).encode()
    implementation = Path(__file__).with_name("engine.py").read_bytes()
    return _digest(catalog + implementation)


def fingerprint() -> dict[str, str | int]:
    configuration = json.dumps(FEATURE_CONFIGURATION, sort_keys=True).encode()
    return {
        "version": VERSION,
        "formula_hash": formula_digest(),
        "configuration_hash": _digest(configuration),
        "generator_version": GENERATOR_VERSION,
    }
