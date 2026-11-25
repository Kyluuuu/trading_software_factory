"""Chronological purge and embargo for overlapping supervised outcomes."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise

from mnq_system.labels.engine import LabelRow


@dataclass(frozen=True, slots=True)
class PurgedSplit:
    train: tuple[LabelRow, ...]
    validation: tuple[LabelRow, ...]
    test: tuple[LabelRow, ...]
    purged: tuple[LabelRow, ...]


def purged_split(
    rows: tuple[LabelRow, ...], validation_start: datetime, test_start: datetime, embargo: timedelta
) -> PurgedSplit:
    """No earlier set may have an outcome whose close overlaps the next set.

    Embargo applies after each boundary. A row is excluded rather than having
    its maturity shortened to fit a split.
    """
    if (
        validation_start.tzinfo is None
        or test_start.tzinfo is None
        or validation_start >= test_start
        or embargo < timedelta(0)
    ):
        raise ValueError("ordered timezone-aware split boundaries and nonnegative embargo required")
    if any(
        left.prediction_timestamp >= right.prediction_timestamp for left, right in pairwise(rows)
    ):
        raise ValueError("label rows must be strictly ordered")
    train: list[LabelRow] = []
    validation: list[LabelRow] = []
    test: list[LabelRow] = []
    purged: list[LabelRow] = []
    for row in rows:
        prediction = row.prediction_timestamp
        if prediction < validation_start and row.maturity_timestamp <= validation_start:
            train.append(row)
        elif (
            validation_start + embargo <= prediction < test_start
            and row.maturity_timestamp <= test_start
        ):
            validation.append(row)
        elif prediction >= test_start + embargo:
            test.append(row)
        else:
            purged.append(row)
    return PurgedSplit(tuple(train), tuple(validation), tuple(test), tuple(purged))
