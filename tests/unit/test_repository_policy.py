"""Static policy checks that keep safety tooling from silently weakening."""

import tomllib
from pathlib import Path
from typing import cast

import pytest
import yaml

IMMUTABLE_SHA_LENGTH = 40


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = cast("object", yaml.safe_load(path.read_text(encoding="utf-8")))
    assert isinstance(loaded, dict)
    return cast("dict[str, object]", loaded)


def _pyproject() -> dict[str, object]:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def _tool_table(name: str) -> dict[str, object]:
    tool = cast("dict[str, object]", _pyproject()["tool"])
    return cast("dict[str, object]", tool[name])


def test_third_party_pre_commit_repos_are_pinned_to_commit_shas() -> None:
    config = _load_yaml(Path(".pre-commit-config.yaml"))
    repos = cast("list[dict[str, object]]", config["repos"])
    external = [repo for repo in repos if repo["repo"] != "local"]
    assert external, "expected external hook repositories"
    for repo in external:
        revision = cast("str", repo["rev"])
        assert len(revision) == IMMUTABLE_SHA_LENGTH, f"{repo['repo']} must pin an immutable SHA"
        assert all(character in "0123456789abcdef" for character in revision)


def test_subprocess_security_rules_are_not_globally_disabled() -> None:
    bandit_skips = cast("list[str]", _tool_table("bandit").get("skips", []))
    assert bandit_skips == []
    ruff_ignores = cast(
        "list[str]", cast("dict[str, object]", _tool_table("ruff")["lint"]).get("ignore", [])
    )
    assert "S603" not in ruff_ignores
    assert "S404" not in ruff_ignores


def test_hardcoded_secret_rules_are_not_disabled_for_tests() -> None:
    lint = cast("dict[str, object]", _tool_table("ruff")["lint"])
    per_file = cast("dict[str, list[str]]", lint.get("per-file-ignores", {}))
    for pattern, rules in per_file.items():
        assert "S105" not in rules, f"{pattern} must not disable hardcoded-secret lint"
        assert "S106" not in rules, f"{pattern} must not disable hardcoded-secret lint"


@pytest.mark.parametrize("path", [Path("scripts/verify.py"), Path(".pre-commit-config.yaml")])
def test_secret_scanner_is_wired_into_shared_commands(path: Path) -> None:
    assert "detect-secrets-hook" in path.read_text(encoding="utf-8")


def test_ci_runs_the_canonical_security_command() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "scripts/verify.py security" in workflow
