# Architecture and safety boundaries

Dependency direction is `domain <- sessions/risk <- backtest/execution <- live`. `domain` owns immutable values, enums, protocols, and errors. `risk` exclusively issues approved orders. `sessions` owns permitted-time decisions. `execution` accepts only approved orders. `live` is adapter-only and currently contains a permanently disabled adapter. `features` and `models` may eventually propose intents but cannot approve or submit them.

`mnq_system.architecture` scans imports and sensitive operation names. `tests/architecture` scans the repository and feeds deliberately invalid modules to each important rule. Add a rule to `_FORBIDDEN_IMPORTS` or `check_source`, add a synthetic failing example, then run `make architecture`.

Backtest and future live adapters must consume the same `RiskPolicy` and `SessionPolicy` ports. Domain code cannot depend on YAML/Pydantic configuration formats.
