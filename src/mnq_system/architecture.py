"""Machine-enforced package dependency and authority rules."""

import ast
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = "mnq_system"


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    message: str


_FORBIDDEN_IMPORTS: dict[str, frozenset[str]] = {
    "domain": frozenset(
        {
            "config",
            "data",
            "sessions",
            "risk",
            "features",
            "models",
            "backtest",
            "execution",
            "live",
            "reporting",
        }
    ),
    "sessions": frozenset({"risk", "backtest", "execution", "live"}),
    "risk": frozenset({"backtest", "execution", "live"}),
    "features": frozenset({"risk", "execution", "live"}),
    "models": frozenset({"risk", "execution", "live"}),
    "backtest": frozenset({"live"}),
    "execution": frozenset({"live"}),
    "live": frozenset({"risk", "sessions", "features", "models", "backtest"}),
}

# Only the risk package may use the approval authority, and only the module that
# defines it may reference it internally.
_APPROVAL_SYMBOLS = frozenset({"RiskApprovalAuthority", "RISK_APPROVAL_AUTHORITY"})
_APPROVAL_OWNER = ("domain", "orders.py")
_APPROVAL_PACKAGES = frozenset({"risk"})

# Order submission vocabulary. Only execution adapters may expose these.
_SUBMISSION_NAMES = frozenset(
    {
        "submit",
        "submit_order",
        "place_order",
        "send",
        "send_order",
        "transmit",
        "transmit_order",
        "route_order",
        "dispatch_order",
        "execute_order",
    }
)
_ORDER_INTAKE_NAMES = _SUBMISSION_NAMES | {"accept"}
_APPROVED_ORDER_TYPE = "RiskApprovedOrder"

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _module_parts(path: Path) -> list[str] | None:
    """Return path parts from the package root, or None outside the package."""
    parts = list(path.parts)
    try:
        index = parts.index(PACKAGE_ROOT)
    except ValueError:
        return None
    return parts[index:]


def _package_of(path: Path) -> str | None:
    parts = _module_parts(path)
    if parts is None or len(parts) < 2 or parts[1].endswith(".py"):
        return None
    return parts[1]


def _package_parts(path: Path) -> list[str]:
    """Return the dotted package parts containing the module."""
    parts = _module_parts(path) or [PACKAGE_ROOT]
    return parts[:-1] if parts[-1].endswith(".py") else parts


def _resolve(node: ast.ImportFrom, path: Path) -> list[str]:
    """Resolve absolute and relative import targets to dotted parts."""
    module_parts = node.module.split(".") if node.module else []
    if not node.level:
        return module_parts
    base = _package_parts(path)
    ascent = node.level - 1
    base = base[: len(base) - ascent] if ascent else base
    return [*base, *module_parts]


def _imported_packages(absolute: list[str], names: list[str]) -> list[str]:
    """Return the top-level packages referenced by a resolved import."""
    if not absolute or absolute[0] != PACKAGE_ROOT:
        return []
    if len(absolute) >= 2:
        return [absolute[1]]
    return names


def _annotation_names(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _accepts_approved_order(node: _FunctionNode) -> bool:
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return any(_annotation_names(arg.annotation) == _APPROVED_ORDER_TYPE for arg in arguments)


def _check_imports(
    node: ast.AST,
    path: Path,
    package: str,
    forbidden: frozenset[str],
) -> list[Violation]:
    if isinstance(node, ast.Import):
        targets = [alias.name.split(".") for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        targets = [_resolve(node, path)]
    else:
        return []
    names = [alias.name for alias in node.names] if isinstance(node, ast.ImportFrom) else []
    violations: list[Violation] = []
    for absolute in targets:
        for imported in _imported_packages(absolute, names):
            if imported in forbidden:
                violations.append(Violation(str(path), f"{package} must not import {imported}"))
    return violations


def _check_approval_authority(node: ast.AST, path: Path, package: str) -> list[Violation]:
    parts = _module_parts(path) or []
    if package in _APPROVAL_PACKAGES or tuple(parts[1:]) == _APPROVAL_OWNER:
        return []
    referenced = False
    if isinstance(node, ast.ImportFrom):
        referenced = any(alias.name in _APPROVAL_SYMBOLS for alias in node.names)
    elif isinstance(node, ast.Name):
        referenced = node.id in _APPROVAL_SYMBOLS
    elif isinstance(node, ast.Attribute):
        referenced = node.attr in _APPROVAL_SYMBOLS
    if not referenced:
        return []
    return [Violation(str(path), "only risk may use the approval authority")]


def _check_functions(node: ast.AST, path: Path, package: str) -> list[Violation]:
    if not isinstance(node, _FunctionNode):
        return []
    violations: list[Violation] = []
    if package not in {"execution", "live"} and node.name in _SUBMISSION_NAMES:
        violations.append(Violation(str(path), f"{package} must not expose order submission"))
    if package == "live":
        if node.name in {"calculate_position_size", "position_size"}:
            violations.append(Violation(str(path), "live must not calculate position size"))
        if node.name in _ORDER_INTAKE_NAMES and not _accepts_approved_order(node):
            violations.append(Violation(str(path), "live submission must accept RiskApprovedOrder"))
    return violations


def check_source(path: Path, source: str) -> list[Violation]:
    package = _package_of(path)
    if package is None:
        return []
    forbidden = _FORBIDDEN_IMPORTS.get(package, frozenset())
    violations: list[Violation] = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        violations.extend(_check_imports(node, path, package, forbidden))
        violations.extend(_check_approval_authority(node, path, package))
        violations.extend(_check_functions(node, path, package))
    return violations


def check_tree(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(root.rglob("*.py")):
        violations.extend(check_source(path, path.read_text(encoding="utf-8")))
    return violations
