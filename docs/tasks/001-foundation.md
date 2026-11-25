# Task 001: Foundation

Status: implemented; validation evidence is recorded in the completion report.

Scope is repository/tooling, immutable safety values, typed fail-closed configuration, explicit position states, risk-approved order capability, disabled live adapter, architecture enforcement, tests, security checks, and CI. No strategy, indicators, training, downloads, broker authentication/submission, or trading behavior is included.

An independent review of this milestone produced remediation recorded in ADR 0002: runtime order validation at execution and adapter boundaries, a token-guarded single approval authority, an import-resolving architecture scanner, removal of the unsafe `PENDING_EXIT`/`CANCEL` transition, exact-integer contract quantities, one configuration pipeline so environment variables always win, globally enabled subprocess rules with line-level exceptions, `detect-secrets` wired into the shared command, SHA-pinned hook repositories, and session tests. Each fix landed with a failing regression test first.

A regression fix must begin with a failing focused test. CI maps directly to `scripts/verify.py`; generated settings schema and `uv.lock` are checked for drift. Repository setup installs the pre-commit hook, with `make hooks` available to repair an existing checkout.
