# Repository operation

- Run `make verify` (or `uv run python scripts/verify.py verify`) before completion.
- Dependencies flow `domain <- sessions/risk <- backtest/execution <- live`; proposal code belongs in `features` or `models`, adapters in boundary packages.
- Put typed settings in `src/mnq_system/config` and committed defaults in `config/`.
- Never weaken checks to make a build pass and never commit secrets.
- Add a failing regression test before fixing a defect; convert repeated mistakes into static checks, architecture rules, schemas, tests, or runtime guards.
- Update the relevant `docs/tasks/` record when behavior changes.
- Never enable live trading or broker submission.
