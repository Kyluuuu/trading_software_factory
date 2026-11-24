# ADR 0001: Python foundation toolchain

**Status:** Accepted

Use Python 3.12 provisioned by `uv`, a `src` layout, pinned direct dependencies, and committed `uv.lock`. Ruff owns formatting/import sorting/linting; Pyright runs strict typing; pytest/Hypothesis test behavior; Bandit and pip-audit scan security. Pydantic Settings validates configuration.

One Python runner centralizes commands so Make, VS Code, pre-commit, and GitHub Actions cannot silently diverge. Custom AST rules were chosen over documentation-only boundaries and avoid adding a second dependency graph configuration language.
