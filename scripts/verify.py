"""Canonical deterministic development command runner."""

import argparse
import json

# Only fixed, repository-owned command tuples are executed; see `run` below.
import subprocess  # nosec B404
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from mnq_system.config.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "config" / "schema" / "settings.schema.json"
SECRETS_BASELINE = ROOT / ".secrets.baseline"
_SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        ".hypothesis",
        ".pytest_cache",
        ".pyright",
        ".ruff_cache",
        "__pycache__",
        "htmlcov",
        "data",
        "models",
        "artifacts",
        "reports",
    }
)
# uv.lock holds dependency hashes, not secrets, and would otherwise dominate the
# scanner's high-entropy findings. The baseline cannot scan itself.
_SKIPPED_FILES = frozenset({"uv.lock", ".secrets.baseline", ".coverage", "coverage.xml", ".env"})
_SKIPPED_SUFFIXES = frozenset(
    {".pyc", ".db", ".sqlite", ".sqlite3", ".parquet", ".csv", ".joblib", ".pkl"}
)


def run(command: Sequence[str]) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    # `command` is always a literal tuple defined in this file.
    subprocess.run(command, cwd=ROOT, check=True)  # nosec B603 # noqa: S603


def scannable_files() -> Iterator[str]:
    """Yield repository files that secret scanning must cover."""
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if _SKIPPED_DIRECTORIES.intersection(relative.parts):
            continue
        if relative.name in _SKIPPED_FILES or relative.suffix in _SKIPPED_SUFFIXES:
            continue
        yield str(relative)


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


def secrets() -> None:
    run(("detect-secrets-hook", "--baseline", str(SECRETS_BASELINE), *scannable_files()))


def security() -> None:
    run(("bandit", "-c", "pyproject.toml", "-r", "src", "scripts"))
    secrets()
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
            "secrets",
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
        "secrets": secrets,
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
