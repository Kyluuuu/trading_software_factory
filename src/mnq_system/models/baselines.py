"""Offline, fixed-parameter, point-in-time direction baseline research only."""

import argparse
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mnq_system.data.ingest import DEFAULT_ROOT, months
from mnq_system.data.storage import PartitionStore
from mnq_system.domain.values import Symbol
from mnq_system.features.asof import join_as_of
from mnq_system.features.engine import FeatureRow
from mnq_system.features.storage import FeatureStore
from mnq_system.features.training_readiness import assess
from mnq_system.labels.build import DEFAULT_LABEL_ROOT
from mnq_system.labels.engine import LabelConfig, LabelRow
from mnq_system.labels.splits import PurgedSplit, purged_split
from mnq_system.labels.storage import LabelStore


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


Sample = tuple[tuple[float, ...], int, float, datetime, datetime]

APPROVED = frozenset({"SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"})
CLASSES = ("down", "neutral", "up")
HORIZONS = (15, 30)


def choose_symbols(values: Sequence[str]) -> tuple[Symbol, ...]:
    if not values or len(set(values)) != len(values):
        raise ValueError("symbols must be nonempty and unique")
    if set(values) - APPROVED:
        raise ValueError(f"not approved for baseline training: {sorted(set(values) - APPROVED)}")
    return tuple(Symbol(value) for value in values)


def split_samples(
    rows: tuple[LabelRow, ...], validation: datetime, test: datetime, embargo: timedelta
) -> PurgedSplit:
    """Use full label-row maturity, conservatively purging all horizons at once."""
    return purged_split(rows, validation, test, embargo)


def align(
    labels: tuple[LabelRow, ...],
    features: tuple[FeatureRow, ...],
    columns: tuple[str, ...],
    horizon: int,
) -> tuple[tuple[tuple[float, ...], int, float, datetime, datetime], ...]:
    """Join only the latest feature known at prediction; drop incomplete vectors."""
    if horizon not in HORIZONS or not columns:
        raise ValueError("unsupported horizon or empty feature set")
    if not labels:
        return ()
    joined = join_as_of(features, tuple(row.prediction_timestamp for row in labels))
    result: list[tuple[tuple[float, ...], int, float, datetime, datetime]] = []
    for row, feature in zip(labels, joined, strict=True):
        if (
            feature.available_at != row.prediction_timestamp
            or row.available_at != row.prediction_timestamp
        ):
            raise ValueError("prediction-time feature mismatch")
        outcome = row.forward[horizon]
        if outcome.direction not in CLASSES or outcome.maturity_timestamp > row.maturity_timestamp:
            raise ValueError("invalid direction or maturity")
        values = tuple(feature.values[name] for name in columns)
        if any(value is None for value in values):
            continue
        result.append(
            (
                tuple(float(value) for value in values if value is not None),
                CLASSES.index(outcome.direction),
                float(outcome.raw_return),
                row.prediction_timestamp,
                outcome.maturity_timestamp,
            )
        )
    return tuple(result)


def load_symbol(
    symbol: Symbol,
    start: datetime,
    end: datetime,
    raw: PartitionStore,
    features: FeatureStore,
    labels: LabelStore,
) -> tuple[tuple[LabelRow, ...], tuple[FeatureRow, ...], tuple[str, ...]]:
    readiness = assess(raw, features, symbol, start, end, Decimal("99"))
    if readiness.status != "TRAINABLE (RESEARCH ONLY)":
        raise ValueError(f"{symbol.value} not trainable: {readiness.reasons}")
    intervals = months(start, end)
    all_labels: list[LabelRow] = []
    all_features: list[FeatureRow] = []
    for index, (year, month, _left, _right) in enumerate(intervals):
        rm, fm = raw.metadata(symbol, year, month), features.metadata(symbol, year, month)
        if rm is None or fm is None:
            raise ValueError("missing input manifest")
        following = intervals[index + 1] if index + 1 < len(intervals) else None
        next_meta = raw.metadata(symbol, following[0], following[1]) if following else None
        if following and next_meta is None:
            raise ValueError("missing next raw manifest")
        sources = {
            "raw": str(rm["sha256"]),
            "features": str(fm["sha256"]),
            "next_raw": str(next_meta["sha256"]) if next_meta else "none",
        }
        if not labels.current(symbol, year, month, sources, LabelConfig()):
            raise ValueError(f"missing or stale labels: {symbol.value} {year}-{month:02d}")
        all_labels.extend(labels.read(symbol, year, month, LabelConfig()))
        all_features.extend(features.read(symbol, year, month))
    if any(row.maturity_timestamp > end for row in all_labels):
        raise ValueError("labels exceed dataset cutoff")
    return tuple(all_labels), tuple(all_features), readiness.admissible


def _matrix(samples: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([sample[0] for sample in samples], dtype=np.float64), np.asarray(
        [sample[1] for sample in samples], dtype=np.int64
    )


def _trading(
    samples: Sequence[tuple[tuple[float, ...], int, float, datetime, datetime]],
    predictions: np.ndarray,
    horizon: int,
) -> dict[str, float | int | None]:
    """One non-overlapping hypothetical position per symbol; zero-cost gross returns."""
    pnl: list[float] = []
    next_entry: datetime | None = None
    for sample, prediction in zip(samples, predictions, strict=True):
        _, _, raw_return, entry, maturity = sample
        if prediction == 1 or (next_entry is not None and entry < next_entry):
            continue
        next_entry = maturity
        pnl.append(raw_return if prediction == 2 else -raw_return)
    if not pnl:
        return {
            "trades": 0,
            "hit_rate": None,
            "expected_return": None,
            "average_return": None,
            "profit_factor": None,
            "maximum_drawdown": None,
            "sharpe_ratio": None,
        }
    values = np.asarray(pnl)
    equity = np.cumprod(1 + values)
    peak = np.maximum.accumulate(np.r_[1.0, equity])
    gains = float(np.maximum(values, 0).sum())
    losses = float(np.maximum(-values, 0).sum())
    average = float(values.mean())
    deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "trades": len(pnl),
        "hit_rate": float(np.mean(values > 0)),
        "expected_return": average,
        "average_return": average,
        "profit_factor": gains / losses if losses else None,
        "maximum_drawdown": float(np.max(1 - equity / peak[1:])),
        "sharpe_ratio": average / deviation * np.sqrt(252 * 390 / horizon) if deviation else None,
    }


def _evaluate(
    samples: Sequence[tuple[tuple[float, ...], int, float, datetime, datetime]],
    probability: np.ndarray,
    horizon: int,
) -> dict[str, object]:
    _, truth = _matrix(samples)
    if probability.shape != (len(samples), 3):
        raise ValueError("probabilities must include down, neutral and up")
    prediction = probability.argmax(axis=1)
    try:
        auc: float | None = float(
            roc_auc_score(truth, probability, labels=[0, 1, 2], multi_class="ovr", average="macro")
        )
    except ValueError:
        auc = None
    return {
        "samples": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, average="macro", zero_division=0)),
        "recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "roc_auc": auc,
        "log_loss": float(log_loss(truth, probability, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "trading": _trading(samples, prediction, horizon),
    }


def _models() -> dict[str, Classifier]:
    return {
        "logistic_regression": cast(
            "Classifier",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=300, random_state=42)),
                ]
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=40, max_depth=8, min_samples_leaf=20, n_jobs=4, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=40,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_jobs=4,
            random_state=42,
        ),
        "lightgbm": cast(
            "Classifier",
            LGBMClassifier(
                n_estimators=40,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=15,
                min_child_samples=30,
                verbosity=-1,
                n_jobs=4,
                random_state=42,
            ),
        ),
    }


def _rank(model: Classifier, columns: tuple[str, ...]) -> list[dict[str, float | str]]:
    if isinstance(model, Pipeline):
        linear = cast("LogisticRegression", model.named_steps["classifier"])
        importance = np.mean(np.abs(linear.coef_), axis=0)
    elif isinstance(model, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
        importance = model.feature_importances_
    else:
        raise ValueError("unsupported baseline estimator")
    values = np.asarray(importance, dtype=float)
    total = float(values.sum())
    return [
        {"feature": name, "importance": float(value / total) if total else 0.0}
        for name, value in sorted(zip(columns, values, strict=True), key=lambda item: -item[1])
    ][:20]


def _baselines(
    train: Sequence[tuple[tuple[float, ...], int, float, datetime, datetime]],
    test: Sequence[tuple[tuple[float, ...], int, float, datetime, datetime]],
    horizon: int,
) -> dict[str, dict[str, object]]:
    _, labels = _matrix(train)
    majority = int(np.bincount(labels, minlength=3).argmax())
    random = np.random.default_rng(42).integers(0, 3, size=len(test))
    result: dict[str, dict[str, object]] = {}
    for name, prediction in (
        ("random_prediction", random),
        ("always_long", np.full(len(test), 2)),
        ("always_short", np.zeros(len(test), dtype=int)),
        ("majority_class", np.full(len(test), majority)),
    ):
        # Smoothed probabilities keep log loss finite for deterministic baselines.
        probs = np.full((len(test), 3), 0.001)
        probs[np.arange(len(test)), prediction] = 0.998
        result[name] = _evaluate(test, probs, horizon)
    return result


def _fit(
    train: Sequence[tuple[tuple[float, ...], int, float, datetime, datetime]],
    validation: Sequence[tuple[tuple[float, ...], int, float, datetime, datetime]],
    test: Sequence[tuple[tuple[float, ...], int, float, datetime, datetime]],
    columns: tuple[str, ...],
    horizon: int,
) -> dict[str, object]:
    if not train or not validation or not test:
        raise ValueError("each split must contain labeled complete feature vectors")
    x_train, y_train = _matrix(train)
    if set(y_train) != {0, 1, 2}:
        raise ValueError("training split must contain all direction classes")
    x_valid, _ = _matrix(validation)
    x_test, _ = _matrix(test)
    results: dict[str, object] = {
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "baselines": _baselines(train, test, horizon),
    }
    for name, model in _models().items():
        begun = time.perf_counter()
        model.fit(x_train, y_train)
        valid_prob = model.predict_proba(x_valid)
        test_prob = model.predict_proba(x_test)
        results[name] = {
            "fit_seconds": round(time.perf_counter() - begun, 3),
            "validation": _evaluate(validation, valid_prob, horizon),
            "test": _evaluate(test, test_prob, horizon),
            "top_features": _rank(model, columns),
        }
    return results


def run(
    symbols: tuple[Symbol, ...],
    start: datetime,
    end: datetime,
    embargo: timedelta,
    raw: PartitionStore,
    features: FeatureStore,
    labels: LabelStore,
) -> dict[str, object]:
    if not symbols or any(symbol.value not in APPROVED for symbol in symbols):
        raise ValueError("not approved for baseline training")
    if end - start < timedelta(days=365) or embargo < timedelta(0):
        raise ValueError("365-day dataset and nonnegative embargo required")
    validation = start + (end - start) * 3 / 4
    test = start + (end - start) * 9 / 10
    begun = time.perf_counter()
    report: dict[str, object] = {
        "symbols": [symbol.value for symbol in symbols],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "validation_start": validation.isoformat(),
        "test_start": test.isoformat(),
        "embargo_minutes": embargo.total_seconds() / 60,
        "classes": CLASSES,
        "horizons": HORIZONS,
        "evaluation_note": (
            "Gross hypothetical non-overlapping per-symbol returns; no execution costs, "
            "slippage or portfolio allocation; overlapping observations remain in "
            "classification metrics."
        ),
        "results": {},
    }
    # The admissible column set is determined from each symbol's verified readiness.
    # Hold only one decoded symbol's partitions at a time.
    results: dict[str, dict[str, object]] = {
        str(horizon): {"by_symbol": {}} for horizon in HORIZONS
    }
    pooled: dict[int, dict[str, dict[str, tuple[Sample, ...]]]] = {
        horizon: {} for horizon in HORIZONS
    }
    for symbol in symbols:
        label_rows, feature_rows, admissible = load_symbol(
            symbol, start, end, raw, features, labels
        )
        split = split_samples(label_rows, validation, test, embargo)
        for horizon in HORIZONS:
            sample_sets = {
                name: align(getattr(split, name), feature_rows, admissible, horizon)
                for name in ("train", "validation", "test")
            }
            by_symbol = cast("dict[str, object]", results[str(horizon)]["by_symbol"])
            by_symbol[symbol.value] = {
                "features": admissible,
                "purged": len(split.purged),
                **_fit(
                    sample_sets["train"],
                    sample_sets["validation"],
                    sample_sets["test"],
                    admissible,
                    horizon,
                ),
            }
            pooled[horizon][symbol.value] = sample_sets
        del label_rows, feature_rows, split
    for horizon in HORIZONS:
        by_symbol = cast("dict[str, dict[str, object]]", results[str(horizon)]["by_symbol"])
        first_features = cast("tuple[str, ...]", by_symbol[symbols[0].value]["features"])
        common = tuple(
            name
            for name in first_features
            if all(
                name in cast("tuple[str, ...]", by_symbol[symbol.value]["features"])
                for symbol in symbols
            )
        )
        combined: dict[str, list[Sample]] = {name: [] for name in ("train", "validation", "test")}
        for symbol in symbols:
            columns = cast("tuple[str, ...]", by_symbol[symbol.value]["features"])
            positions = tuple(columns.index(name) for name in common)
            for name in combined:
                combined[name].extend(
                    (tuple(sample[0][i] for i in positions), *sample[1:])
                    for sample in pooled[horizon][symbol.value][name]
                )
        pooled_result = _fit(
            combined["train"], combined["validation"], combined["test"], common, horizon
        )
        for model_name in _models():
            model_result = cast("dict[str, object]", pooled_result[model_name])
            for phase in ("validation", "test"):
                cast("dict[str, object]", model_result[phase]).pop("trading")
        pooled_baselines = cast("dict[str, dict[str, object]]", pooled_result["baselines"])
        for baseline_result in pooled_baselines.values():
            baseline_result.pop("trading")
        results[str(horizon)]["across_symbols"] = {"features": common, **pooled_result}
        del combined, pooled_result

    report["results"] = results
    report["runtime_seconds"] = round(time.perf_counter() - begun, 3)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline approved-symbol baseline research")
    parser.add_argument("--symbols", nargs="+", default=sorted(APPROVED))
    parser.add_argument("--start", default="2025-11-25T00:00:00+00:00")
    parser.add_argument("--end", default="2026-11-25T00:00:00+00:00")
    parser.add_argument("--embargo-minutes", type=int, default=60)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=Path("data/features/iex/raw/1min"))
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output", type=Path, default=Path("data/research/baseline-results.json"))
    args = parser.parse_args(argv)
    report = run(
        choose_symbols(args.symbols),
        datetime.fromisoformat(args.start).astimezone(UTC),
        datetime.fromisoformat(args.end).astimezone(UTC),
        timedelta(minutes=args.embargo_minutes),
        PartitionStore(args.raw_root),
        FeatureStore(args.feature_root),
        LabelStore(args.label_root),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Offline research report: {args.output}; runtime={report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
