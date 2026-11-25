# ADR 0002: Order approval boundary and check hardening

**Status:** Accepted

An independent review showed that type annotations alone allowed an `OrderIntent` to reach an adapter, and that the previous public approval proof could be constructed by any caller.

Decisions:

1. `RiskApprovalAuthority` is token-guarded and a single module-level instance exists. `RiskApprovedOrder` verifies that instance by identity and validates its payload types.
2. `require_risk_approved_order` re-checks the exact type at the execution coordinator and at the broker adapter, so dynamic callers cannot rely on annotations being enforced.
3. The architecture scanner resolves all common import forms and restricts both the approval authority and order-submission vocabulary. Every rule has a rejected and a permitted test case.
4. `CANCEL` is not a valid event from `PENDING_EXIT`, because a cancelled exit leaves the original position open and the machine cannot yet represent that side. Expected transitions are asserted from a test-owned table rather than from the implementation.
5. `ContractQuantity` requires an exact `int`, rejecting `bool`, float and `Decimal` inputs.
6. One settings pipeline serves every configuration path, so environment variables always override file values.
7. Subprocess lint/security rules are enabled globally with line-level exceptions only in the command runner, and `detect-secrets` with a committed baseline runs from the shared verification command, pre-commit and CI.

Python cannot make direct construction impossible. The supported path is explicit, guarded at runtime, restricted statically, and covered by tests.
