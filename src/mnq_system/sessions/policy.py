"""Session decision interfaces shared by backtest and future execution."""

from dataclasses import dataclass
from datetime import time
from typing import Protocol
from zoneinfo import ZoneInfo

from mnq_system.domain.values import Symbol, UtcTimestamp


@dataclass(frozen=True, slots=True)
class TradingSession:
    timezone: ZoneInfo
    start: time
    end: time

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("session start must be before session end")


class SessionPolicy(Protocol):
    def permits(self, symbol: Symbol, timestamp: UtcTimestamp) -> bool: ...
