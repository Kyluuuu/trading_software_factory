"""Offline feature diagnostics and fixed-baseline group ablation research."""

import argparse
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from mnq_system.data.ingest import DEFAULT_ROOT
from mnq_system.data.storage import PartitionStore
from mnq_system.features.storage import FeatureStore
from mnq_system.labels.build import DEFAULT_LABEL_ROOT
from mnq_system.labels.storage import LabelStore
from mnq_system.models.baselines import (
    APPROVED,
    HORIZONS,
    Sample,
    _models,  # pyright: ignore[reportPrivateUsage]
    align,
    choose_symbols,
    load_symbol,
    split_samples,
)

GROUPS: dict[str, tuple[str, ...]] = {
    "time": ("minute_of_day", "hour", "day_of_week", "session_progress"),
    "price": ("open", "high", "low", "close", "returns", "log_returns"),
    "volatility": ("atr", "realized_volatility", "rolling_std", "volatility_percentile"),
    "mean_reversion": (
        "rsi",
        "bollinger_upper",
        "bollinger_lower",
        "bollinger_distance",
        "z_score",
        "distance_from_mean",
        "session_vwap",
        "vwap_distance",
        "vwap_deviation",
    ),
    "trend": (
        "roc",
        "momentum_5",
        "momentum_20",
        "trend_acceleration",
        "ema_distance",
        "ema_slope",
        "adx",
        "trend_strength",
    ),
    "volume": ("volume", "relative_volume", "volume_percentile", "volume_spike_score"),
    "breakout": ("donchian_high", "donchian_low", "compression", "breakout_distance"),
    "market_structure": (
        "higher_highs",
        "lower_lows",
        "swing_structure",
        "support_distance",
        "resistance_distance",
    ),
}


def group_columns(columns: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate feature columns")
    unknown = set(columns) - {name for members in GROUPS.values() for name in members}
    if unknown:
        raise ValueError(f"unclassified features: {sorted(unknown)}")
    return {
        group: tuple(name for name in columns if name in members)
        for group, members in GROUPS.items()
    }


def ablation_columns(columns: tuple[str, ...], group: str) -> tuple[str, ...]:
    grouped = group_columns(columns)
    if group not in grouped:
        raise ValueError(f"unknown group: {group}")
    if not grouped[group]:
        raise ValueError(f"no admissible features in {group}")
    remaining = tuple(name for name in columns if name not in grouped[group])
    if not remaining:
        raise ValueError("no remaining features")
    return remaining


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks for tied values, preserving a monotone rank IC."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    return ((starts + cumulative - 1) / 2 + 1)[inverse]


def feature_statistics(
    x: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    columns: tuple[str, ...],
    *,
    sample_limit: int = 20000,
) -> dict[str, dict[str, float | int | None]]:
    """Descriptive held-out MI against direction; IC/rank IC against signed forward return.

    Deterministic evenly spaced sample caps MI cost, not model evaluation. These
    correlated, overlapping outcomes are not independent significance tests.
    """
    if x.ndim != 2 or x.shape != (len(y), len(columns)) or len(returns) != len(y):
        raise ValueError("feature/target shape mismatch")
    if sample_limit < 4 or len(y) < 4 or not np.isfinite(x).all() or not np.isfinite(returns).all():
        raise ValueError("insufficient or nonfinite observations")
    take = np.linspace(0, len(y) - 1, min(len(y), sample_limit), dtype=int)
    mi = mutual_info_classif(x[take], y[take], random_state=42, n_neighbors=3)
    result: dict[str, dict[str, float | int | None]] = {}
    for i, name in enumerate(columns):
        value = x[:, i]
        constant = np.std(value) == 0
        result[name] = {
            "mutual_information": 0.0 if constant else float(mi[i]),
            "ic": _correlation(value, returns),
            "rank_ic": _correlation(_rank(value), _rank(returns)),
            "observations": len(y),
            "mi_observations": len(take),
        }
    return result


def _arrays(samples: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([sample[0] for sample in samples], dtype=np.float64),
        np.asarray([sample[1] for sample in samples], dtype=np.int64),
        np.asarray([sample[2] for sample in samples], dtype=np.float64),
    )


def _scores(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    prediction = probabilities.argmax(axis=1)
    try:
        auc: float | None = float(
            roc_auc_score(
                truth, probabilities, labels=[0, 1, 2], multi_class="ovr", average="macro"
            )
        )
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "roc_auc": auc,
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1, 2])),
    }


def evaluate_groups(
    train: Sequence[Sample],
    test: Sequence[Sample],
    columns: tuple[str, ...],
    baseline: dict[str, dict[str, float | None]],
) -> dict[str, object]:
    """Use exactly the same observations and fixed-parameter estimators for every ablation."""
    x_train, y_train, _ = _arrays(train)
    x_test, y_test, returns = _arrays(test)
    if len(set(y_train)) != 3 or not len(test):
        raise ValueError("empty test split or missing training class")
    groups = group_columns(columns)
    diagnostics = feature_statistics(x_test, y_test, returns, columns)
    variants = {
        **{f"only_{group}": members for group, members in groups.items() if members},
        **{
            f"without_{group}": ablation_columns(columns, group)
            for group, members in groups.items()
            if members
        },
    }
    measured: dict[str, object] = {}
    for variant, selected in variants.items():
        indices = [columns.index(name) for name in selected]
        models: dict[str, dict[str, float | None]] = {}
        for name, estimator in _models().items():
            estimator.fit(x_train[:, indices], y_train)
            models[name] = _scores(y_test, estimator.predict_proba(x_test[:, indices]))
        measured[variant] = {"features": selected, "models": models}
    return {
        "test_rows": len(test),
        "train_rows": len(train),
        "admissible_features": columns,
        "groups": groups,
        "univariate_test_diagnostics": diagnostics,
        "reference_full_model": baseline,
        "variants": measured,
        "not_testable_groups": [name for name, members in groups.items() if not members],
    }


def run(
    phase4: dict[str, object],
    raw: PartitionStore,
    features: FeatureStore,
    labels: LabelStore,
) -> dict[str, object]:
    """Re-use Phase 4's exact period, membership and split; fit only ablated baselines."""
    symbols = choose_symbols(cast("list[str]", phase4["symbols"]))
    if {symbol.value for symbol in symbols} != set(APPROVED):
        raise ValueError("Phase 4 report must contain all six approved symbols")
    start = datetime.fromisoformat(cast("str", phase4["start"])).astimezone(UTC)
    end = datetime.fromisoformat(cast("str", phase4["end"])).astimezone(UTC)
    if end - start < timedelta(days=365):
        raise ValueError("Phase 4 period too short")
    validation = datetime.fromisoformat(cast("str", phase4["validation_start"])).astimezone(UTC)
    test_start = datetime.fromisoformat(cast("str", phase4["test_start"])).astimezone(UTC)
    embargo = timedelta(minutes=cast("float", phase4["embargo_minutes"]))
    reference = cast("dict[str, dict[str, object]]", phase4["results"])
    pooled_reference = cast("dict[str, object]", reference["15"]["across_symbols"])
    common = tuple(cast("list[str]", pooled_reference["features"]))
    if not common or set(cast("list[int]", phase4["horizons"])) != set(HORIZONS):
        raise ValueError("invalid Phase 4 feature or horizon contract")
    all_samples: dict[int, dict[str, list[Sample]]] = {
        horizon: {"train": [], "test": []} for horizon in HORIZONS
    }
    by_symbol: dict[str, object] = {}
    begun = time.perf_counter()
    for symbol in symbols:
        label_rows, feature_rows, admissible = load_symbol(
            symbol, start, end, raw, features, labels
        )
        if any(name not in admissible for name in common):
            raise ValueError("pooled feature no longer admissible")
        positions = tuple(admissible.index(name) for name in common)
        split = split_samples(label_rows, validation, test_start, embargo)
        per_horizon: dict[str, object] = {}
        for horizon in HORIZONS:
            # Reproduce Phase 4: filter on symbol-admissible columns first,
            # then project only retained rows onto the pooled common columns.
            symbol_train = align(split.train, feature_rows, admissible, horizon)
            symbol_test = align(split.test, feature_rows, admissible, horizon)
            train = tuple((tuple(row[0][i] for i in positions), *row[1:]) for row in symbol_train)
            test = tuple((tuple(row[0][i] for i in positions), *row[1:]) for row in symbol_test)
            original = cast("dict[str, object]", reference[str(horizon)]["by_symbol"])
            symbol_reference = cast("dict[str, object]", original[symbol.value])
            counts = cast("dict[str, int]", symbol_reference["counts"])
            if len(train) != counts["train"] or len(test) != counts["test"]:
                raise ValueError("Phase 4 symbol split mismatch")
            x, y, forward_return = _arrays(test)
            per_horizon[str(horizon)] = {
                "test_rows": len(test),
                "univariate_test_diagnostics": feature_statistics(x, y, forward_return, common),
            }
            all_samples[horizon]["train"].extend(train)
            all_samples[horizon]["test"].extend(test)
        by_symbol[symbol.value] = per_horizon
        del label_rows, feature_rows, split
    pooled: dict[str, object] = {}
    for horizon in HORIZONS:
        original = cast("dict[str, object]", reference[str(horizon)]["across_symbols"])
        counts = cast("dict[str, int]", original["counts"])
        train = all_samples[horizon]["train"]
        test = all_samples[horizon]["test"]
        if (
            len(train) != counts["train"]
            or len(test) != counts["test"]
            or tuple(cast("list[str]", original["features"])) != common
        ):
            raise ValueError("Phase 4 pooled split mismatch")
        full = {
            name: cast("dict[str, dict[str, float | None]]", original[name])["test"]
            for name in _models()
        }
        pooled[str(horizon)] = evaluate_groups(train, test, common, full)
        del train, test
    return {
        "source": "data/research/baseline-results.json",
        "symbols": [symbol.value for symbol in symbols],
        "horizons": HORIZONS,
        "groups_catalog": GROUPS,
        "notes": (
            "Point-in-time admissible columns only; held-out univariate descriptive statistics "
            "are not independent significance tests; group-only and leave-group-out models "
            "reuse original fixed parameters and identical purged/embargoed observations. "
            "Repeated use of Phase 4's test period is exploratory, not a fresh "
            "out-of-sample test."
        ),
        "by_symbol": by_symbol,
        "pooled": pooled,
        "runtime_seconds": round(time.perf_counter() - begun, 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline feature-group signal diagnostics")
    parser.add_argument("--phase4", type=Path, default=Path("data/research/baseline-results.json"))
    parser.add_argument("--output", type=Path, default=Path("data/research/signal-discovery.json"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=Path("data/features/iex/raw/1min"))
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    args = parser.parse_args(argv)
    report = run(
        json.loads(args.phase4.read_text(encoding="utf-8")),
        PartitionStore(args.raw_root),
        FeatureStore(args.feature_root),
        LabelStore(args.label_root),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Signal-discovery report: {args.output}; runtime={report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
