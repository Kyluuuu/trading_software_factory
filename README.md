# MNQ System Foundation

Safety-oriented Python 3.12 foundation for future MNQ futures research. It contains no strategy, market-data downloader, broker connection, order submission, paper/live trading, or ML implementation.

## Setup

Install `uv`, then run `make setup`; this also installs the Git pre-commit hook. On Windows or without Make, run `uv python install 3.12`, `uv sync --locked --python 3.12`, then `uv run pre-commit install`. Existing checkouts can install or repair the hook with `make hooks`.

## Commands

`make format`, `make lint`, `make typecheck`, `make architecture`, `make test`, `make security`, `make secrets`, and `make verify` call `scripts/verify.py`. Without Make use `uv run python scripts/verify.py <command>`. `verify` is the required pre-commit/CI-equivalent check; dependency auditing may access vulnerability indexes.

Secret scanning uses `detect-secrets` with the committed `.secrets.baseline`, and runs from `make secrets`, `make verify`, pre-commit and CI. Mark a confirmed false positive with an inline `pragma: allowlist secret` comment instead of excluding whole directories.

## Configuration

`config/default.yaml` supplies conservative committed defaults. `MNQ_` environment variables override model defaults using `__` for nesting. Secrets are loaded from environment or an ignored local `.env`, excluded from snapshots and representations, and optional in development/tests. Unknown keys and every live/submission-enabled configuration fail closed.

## Offline stock/ETF testing (no orders)

The offline risk policy allows AAPL, MSFT, NVDA, SPY, QQQ and VOO by default. It requires a stop on the loss side of each proposal and a caller-supplied `AccountSnapshot` with current equity, session-opening equity and high-water equity. It checks the configured weekday/session window, 1% per-order equity budget, 3% daily loss budget and 10% drawdown budget, including the proposal's estimated loss. Costs are modeled as round-trip per-share commission and price slippage; configure nonzero, verified values before drawing conclusions. A stop is **not** a guaranteed fill price. No holiday calendar, ongoing position monitor, existing-exposure aggregation, strategy, or broker execution is implemented.

For optional **read-only** Alpaca paper-account discovery, put paper keys in the ignored `.env` file (`MNQ_SECRETS__ALPACA_API_KEY` and `MNQ_SECRETS__ALPACA_API_SECRET`). Never use live credentials. No network request occurs on import or construction; `AlpacaReadOnlyClient(load_settings()).tradable_symbols()` queries active assets and filters the allowlist, and `.account_equity()` returns display-only equity and previous-close equity. These values do not provide the verified session-opening and peak equity needed for a risk decision. Run offline tests with `make test`; all Alpaca responses are mocked. Never treat this as an automated paper trader.

See `docs/architecture/` for boundaries and CI mapping. Unresolved decisions: confirmed MNQ contract specifications, data provider, broker, commission, and slippage assumptions. Broker submission remains disabled.
