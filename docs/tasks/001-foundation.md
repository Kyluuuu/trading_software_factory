# Task 001: Foundation

Status: implemented; validation evidence is recorded in the completion report.

Scope is repository/tooling, immutable safety values, typed fail-closed configuration, explicit position states, risk-approved order capability, disabled live adapter, architecture enforcement, tests, security checks, and CI. No strategy, indicators, training, downloads, broker authentication/submission, or trading behavior is included.

A regression fix must begin with a failing focused test. CI maps directly to `scripts/verify.py`; generated settings schema and `uv.lock` are checked for drift.
