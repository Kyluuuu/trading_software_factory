# MNQ System Foundation

Safety-oriented Python 3.12 foundation for future MNQ futures research. It contains no strategy, market-data downloader, broker connection, order submission, paper/live trading, or ML implementation.

## Setup

Install `uv`, then run `make setup`. On Windows or without Make, run `uv python install 3.12` followed by `uv sync --locked --python 3.12`.

## Commands

`make format`, `make lint`, `make typecheck`, `make architecture`, `make test`, `make security`, and `make verify` call `scripts/verify.py`. Without Make use `uv run python scripts/verify.py <command>`. `verify` is the required pre-commit/CI-equivalent check; dependency auditing may access vulnerability indexes.

## Configuration

`config/default.yaml` supplies conservative committed defaults. `MNQ_` environment variables override model defaults using `__` for nesting. Secrets are environment-only, excluded from snapshots and representations, and optional in development/tests. Unknown keys and every live/submission-enabled configuration fail closed.

See `docs/architecture/` for boundaries and CI mapping. Unresolved decisions: confirmed MNQ contract specifications, data provider, broker, commission, and slippage assumptions. All remain disabled or zero pending independent confirmation.
