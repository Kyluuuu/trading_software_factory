from pathlib import Path

import pytest

from mnq_system.architecture import check_source, check_tree

SRC = Path("src/mnq_system")


def test_repository_obeys_architecture_rules() -> None:
    assert check_tree(SRC) == []


@pytest.mark.parametrize(
    ("package", "source", "message"),
    [
        ("domain", "import mnq_system.live", "domain must not import live"),
        ("domain", "from mnq_system.config import settings", "domain must not import config"),
        ("risk", "from mnq_system.execution import coordinator", "risk must not import execution"),
        ("sessions", "import mnq_system.live", "sessions must not import live"),
        ("features", "import mnq_system.risk", "features must not import risk"),
        ("models", "import mnq_system.execution", "models must not import execution"),
        ("backtest", "import mnq_system.live", "backtest must not import live"),
        ("execution", "import mnq_system.live", "execution must not import live"),
        ("live", "import mnq_system.risk", "live must not import risk"),
        ("live", "def calculate_position_size():\n    pass", "live must not calculate"),
        ("features", "def submit(order):\n    pass", "must not expose order submission"),
        (
            "execution",
            "from mnq_system.domain.orders import RISK_APPROVAL_AUTHORITY",
            "only risk may use the approval authority",
        ),
        ("live", "def submit(self, order: object):\n    pass", "must accept RiskApprovedOrder"),
        ("live", "def accept(self, order):\n    pass", "must accept RiskApprovedOrder"),
        ("execution", "from mnq_system import live", "execution must not import live"),
        ("domain", "from mnq_system import config", "domain must not import config"),
        ("execution", "from .. import live", "execution must not import live"),
        ("execution", "from ..live import disabled", "execution must not import live"),
        ("execution", "from . import bad\nfrom ...mnq_system import live", "must not import live"),
        ("execution", "import mnq_system.live as broker", "execution must not import live"),
        (
            "execution",
            "import mnq_system.domain.orders as orders\nx = orders.RISK_APPROVAL_AUTHORITY",
            "only risk may use the approval authority",
        ),
        (
            "features",
            "from mnq_system.domain import orders\nx = orders.RiskApprovalAuthority",
            "only risk may use the approval authority",
        ),
        ("features", "def send_order(order):\n    pass", "must not expose order submission"),
        ("models", "def transmit(order):\n    pass", "must not expose order submission"),
        ("backtest", "async def place_order(order):\n    pass", "must not expose order submission"),
    ],
)
def test_deliberately_forbidden_relationship_is_detected(
    package: str,
    source: str,
    message: str,
) -> None:
    path = Path("src") / "mnq_system" / package / "bad.py"
    violations = check_source(path, source)
    assert any(message in violation.message for violation in violations)


@pytest.mark.parametrize(
    ("package", "source"),
    [
        ("risk", "from mnq_system.domain.orders import RISK_APPROVAL_AUTHORITY"),
        ("execution", "from mnq_system.domain.orders import RiskApprovedOrder"),
        ("live", 'def accept(self, order: "RiskApprovedOrder") -> None:\n    pass'),
        ("live", "def accept(self, order: RiskApprovedOrder) -> None:\n    pass"),
        ("execution", "from mnq_system.domain import orders"),
        ("backtest", "from mnq_system.sessions import policy"),
    ],
)
def test_permitted_relationships_are_not_flagged(package: str, source: str) -> None:
    path = Path("src") / "mnq_system" / package / "good.py"
    assert check_source(path, source) == []
