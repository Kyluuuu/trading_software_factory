# Dependency rules

Allowed dependencies point inward. Configuration may construct domain values; domain never imports configuration or infrastructure. Risk and session policy definitions are reusable by backtest and execution. Execution does not import live adapters. Live adapters cannot import risk implementation, calculate size, or accept an unvalidated order. Only risk may import the private approval capability; only execution/live may expose submission operations.

Enforcement combines Pyright protocol signatures, runtime capability checks, the AST architecture scanner, and focused tests. To add a rule, first add a rejected synthetic case in `tests/architecture/test_boundaries.py`, implement the scanner rule, and verify the real source tree remains clean.
