"""Canonical deterministic development command runner."""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from mnq_system.config.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "config" / "schema" / "settings.schema.json"


def run(command: Sequence[str]) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def schema_text() -> str:
    return json.dumps(Settings.model_json_schema(), indent=2, sort_keys=True) + "\n"


def check_config(write: bool = False) -> None:
    expected = schema_text()
    if write:
        SCHEMA.write_text(expected, encoding="utf-8")
    elif not SCHEMA.is_file() or SCHEMA.read_text(encoding="utf-8") != expected:
        raise RuntimeError("configuration schema is stale; run: make config")


def format_code() -> None:
    run(("ruff", "check", "--fix", "."))
    run(("ruff", "format", "."))


def lint() -> None:
    run(("ruff", "format", "--check", "."))
    run(("ruff", "check", "."))


def typecheck() -> None:
    run(("pyright",))


def architecture() -> None:
    run(("pytest", "--no-cov", "tests/architecture"))


def test() -> None:
    run(("pytest",))


def security() -> None:
    run(("bandit", "-c", "pyproject.toml", "-r", "src", "scripts"))
    run(("pip-audit",))


def verify() -> None:
    run(("uv", "lock", "--check"))
    check_config()
    lint()
    typecheck()
    architecture()
    test()
    security()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "format",
            "lint",
            "typecheck",
            "architecture",
            "test",
            "security",
            "config",
            "check-config",
            "verify",
        ),
    )
    command = parser.parse_args().command
    commands = {
        "format": format_code,
        "lint": lint,
        "typecheck": typecheck,
        "architecture": architecture,
        "test": test,
        "security": security,
        "config": lambda: check_config(write=True),
        "check-config": check_config,
        "verify": verify,
    }
    commands[command]()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
