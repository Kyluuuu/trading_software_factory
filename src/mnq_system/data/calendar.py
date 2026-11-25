"""Narrow typed boundary around the XNYS exchange calendar."""

from collections.abc import Callable, Sequence
from datetime import date, datetime
from importlib import import_module
from typing import Protocol, cast


class CalendarStamp(Protocol):
    def to_pydatetime(self) -> datetime: ...

    def date(self) -> date: ...


class ExchangeCalendar(Protocol):
    def sessions_in_range(self, start: date, end: date) -> Sequence[CalendarStamp]: ...

    def session_minutes(self, session: CalendarStamp | date) -> Sequence[CalendarStamp]: ...

    def is_session(self, session: date) -> bool: ...


def xnys() -> ExchangeCalendar:
    """Versioned exchange_calendars supplies actual holidays and early closes."""
    module = import_module("exchange_calendars")
    factory = cast("Callable[[str], ExchangeCalendar]", vars(module)["get_calendar"])
    return factory("XNYS")
