"""Configured symbol guard."""

from mnq_system.domain.values import DomainValidationError, Symbol


def require_permitted_symbol(candidate: Symbol, permitted: Symbol) -> Symbol:
    if candidate != permitted:
        raise DomainValidationError(f"symbol {candidate.value} is not permitted")
    return candidate
