# Architecture and safety boundaries

Dependency direction is `domain <- sessions/risk <- backtest/execution <- live`. `domain` owns immutable values, enums, protocols, and errors. `risk` exclusively issues approved orders. `sessions` owns permitted-time decisions. `execution` accepts only approved orders. `live` is adapter-only and currently contains a permanently disabled adapter. `features` and `models` may eventually propose intents but cannot approve or submit them.

`mnq_system.architecture` resolves absolute, root-level, relative and aliased imports, restricts the approval authority to `risk`, and checks order-submission operations. `tests/architecture` scans the repository, feeds deliberately invalid modules to every important rule, and asserts permitted relationships stay allowed. Add a rule to `_FORBIDDEN_IMPORTS` or `check_source`, add both a rejected and a permitted example, then run `make architecture`.

Approval cannot be forged by ordinary code: `RiskApprovalAuthority` is token-guarded, a single instance exists, `RiskApprovedOrder` verifies that instance by identity, and `require_risk_approved_order` re-validates at the execution and adapter boundaries so dynamic callers cannot bypass type annotations.

Backtest and future live adapters must consume the same `RiskPolicy` and `SessionPolicy` ports. Domain code cannot depend on YAML/Pydantic configuration formats.
