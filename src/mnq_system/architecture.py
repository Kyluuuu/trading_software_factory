"""Machine-enforced package dependency rules."""

import ast
from dataclasses import dataclass
from pathlib import Path


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


def _package_from_path(path: Path) -> str | None:
    parts = path.parts
    try:
        return parts[parts.index("mnq_system") + 1]
    except (ValueError, IndexError):
        return None


def check_source(path: Path, source: str) -> list[Violation]:
    package = _package_from_path(path)
    if package is None:
        return []
    tree = ast.parse(source, filename=str(path))
    violations: list[Violation] = []
    forbidden = _FORBIDDEN_IMPORTS.get(package, frozenset())
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        for module in modules:
            prefix = "mnq_system."
            if module.startswith(prefix):
                imported_package = module.removeprefix(prefix).split(".", maxsplit=1)[0]
                if imported_package in forbidden:
                    violations.append(
                        Violation(str(path), f"{package} must not import {imported_package}")
                    )
        if isinstance(node, ast.ImportFrom) and node.module == "mnq_system.domain.orders":
            imported_names = {alias.name for alias in node.names}
            if "RiskApprovalProof" in imported_names and package != "risk":
                violations.append(Violation(str(path), "only risk may import approval capability"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if package not in {"execution", "live"} and node.name in {
                "submit",
                "submit_order",
                "place_order",
            }:
                violations.append(
                    Violation(str(path), f"{package} must not expose order submission")
                )
            if package == "live" and node.name in {"calculate_position_size", "position_size"}:
                violations.append(Violation(str(path), "live must not calculate position size"))
            if package == "live" and node.name in {
                "accept",
                "submit",
                "submit_order",
                "place_order",
            }:
                parameters = node.args.args
                annotation = parameters[-1].annotation if parameters else None
                if not isinstance(annotation, ast.Name) or annotation.id != "RiskApprovedOrder":
                    violations.append(
                        Violation(str(path), "live submission must accept RiskApprovedOrder")
                    )
    return violations


def check_tree(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(root.rglob("*.py")):
        violations.extend(check_source(path, path.read_text(encoding="utf-8")))
    return violations
