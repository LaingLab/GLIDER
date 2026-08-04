"""Render a training summary as a report a person can read.

``train_model`` returns a summary dict of two dozen keys, some of them nested
tables. The Train tab used to ``pprint`` it, which put the two numbers anyone
actually wants -- test accuracy, and which class the model is failing on --
somewhere in the middle of a wall of punctuation.

Deliberately dependency-light and Qt-free: the GUI imports this on a hot path
where pulling in matplotlib (as :mod:`report`, its file-writing sibling, does)
would be paid for nothing. Plain text rather than HTML, because the pane it
lands in is already monospaced, the result stays copy-pasteable into a lab
notebook, and it cannot fight the theme.

Everything is defensive. The summary is a plain dict assembled by the
pipeline, and a key that is absent, ``None``, or an unexpected shape must
degrade to a missing line rather than take out the results pane of a run that
has just spent ten minutes fitting.
"""

from __future__ import annotations

from typing import Any

__all__ = ["format_training_summary"]

#: Hard cap on line length. The pane is a fixed-width card; a table that wraps
#: is less readable than one that truncates, because a wrapped row silently
#: stops lining up with its header.
WIDTH = 76

_MISSING = "—"


def _num(value: Any, places: int = 3) -> str:
    """A float to ``places``, or a dash. Never the string ``None``."""
    try:
        if value is None:
            return _MISSING
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return _MISSING


def _count(value: Any) -> str:
    """An integer with thousands separators, or a dash."""
    try:
        if value is None:
            return _MISSING
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _MISSING


def _plain(value: Any) -> str:
    """An integer with no grouping, or a dash. For grid cells."""
    try:
        if value is None:
            return _MISSING
        return str(int(value))
    except (TypeError, ValueError):
        return _MISSING


def _clip(text: str, width: int) -> str:
    """Truncate to ``width``, marking that something was cut."""
    text = str(text)
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _rule(char: str = "─", width: int = WIDTH) -> str:
    return char * width


def format_training_summary(summary: Any) -> str:
    """A readable report for one ``train_model`` summary.

    Sections that have no data are omitted rather than printed empty: a run
    with no held-out set carries empty per-class metrics and confusion matrix,
    and showing those as blank tables implies the model was evaluated when it
    was not.
    """
    if not isinstance(summary, dict) or not summary:
        return "No training summary available."

    lines: list[str] = []
    lines += _headline(summary)
    lines += _accuracy(summary)
    lines += _data(summary)
    lines += _per_class(summary)
    lines += _confusion(summary)
    lines += _labels(summary)
    lines += _features(summary)
    lines += _settings(summary)
    lines += ["", "The complete summary is stored inside the saved model bundle."]
    return "\n".join(lines)


def _headline(s: dict) -> list[str]:
    classifier = str(s.get("classifier_type") or "classifier")
    strategy = str(s.get("split_strategy") or "unknown split")
    right = f"{classifier} · {strategy}"
    left = " TRAINING RESULT"
    gap = max(1, WIDTH - len(left) - len(right) - 1)
    return [_rule("═"), f"{left}{' ' * gap}{right} ", _rule("═"), ""]


def _accuracy(s: dict) -> list[str]:
    strategy = s.get("split_strategy")
    test_size = s.get("test_size") or 0
    out = []
    if test_size:
        note = {
            "cross_session": "held-out animals",
            "group_shuffle": "held-out zones, same animals",
        }.get(strategy, "held-out rows")
        out.append(f"  Test accuracy      {_num(s.get('test_accuracy')):<8} ← {note}")
    out.append(f"  Train accuracy     {_num(s.get('train_accuracy'))}")

    n_sessions = s.get("n_sessions")
    n_holdout = s.get("n_holdout_sessions") or 0
    if n_holdout:
        out.append(f"  Split              {_count(n_sessions)} train, {_count(n_holdout)} holdout")
    elif not test_size:
        # The number most likely to be misread. 0.98 on the data it was fit to
        # says nothing about a new animal, so it is labelled where it appears.
        out.append(f"  Split              {_count(n_sessions)} sessions, no held-out set")
        out.append("                     train accuracy is a fit, not a generalization estimate")
    else:
        out.append(f"  Split              {_count(n_sessions)} sessions, within-session holdout")
    return out


def _data(s: dict) -> list[str]:
    kept, dropped = s.get("n_rows_kept"), s.get("n_rows_dropped")
    out = ["", "DATA"]
    out.append(f"  Rows        {_count(kept)} kept")
    if dropped:
        # Its own line: on a real session this reads "131,992 dropped", and
        # the explanation is the part that matters. The keep mask drops
        # unlabelled rows, ambiguous ones, and rows with incomplete features
        # — and the unlabelled frames outnumber the others hundreds to one,
        # so naming only the ambiguous ones reads as "your annotations were
        # thrown away".
        out.append(f"              {_count(dropped)} dropped (unlabelled, ambiguous, incomplete)")
    window, fps = s.get("window"), s.get("fps")
    out.append(
        f"  Features    {_count(s.get('n_features')):<10}"
        f"Window {_count(window)} frames @ {_num(fps, 1)} fps"
    )
    subsampled = s.get("background_subsampled_to")
    if subsampled:
        out.append(f"  Background  subsampled to {_count(subsampled)} rows")
    return out


def _per_class(s: dict) -> list[str]:
    metrics = s.get("per_class_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return []

    name_width = min(max((len(str(k)) for k in metrics), default=5) + 1, 28)
    header = f"  {'class':<{name_width}}{'precision':>10}{'recall':>9}{'f1':>9}{'support':>10}"
    out = ["", "PER-CLASS", header, "  " + _rule("─", len(header) - 2)]

    precisions, recalls, f1s, supports = [], [], [], []
    for name, m in sorted(metrics.items()):
        if not isinstance(m, dict):
            continue
        p, r, f = m.get("precision"), m.get("recall"), m.get("f1")
        support = m.get("support")
        out.append(
            f"  {_clip(name, name_width - 1):<{name_width}}"
            f"{_num(p):>10}{_num(r):>9}{_num(f):>9}{_count(support):>10}"
        )
        for bucket, value in ((precisions, p), (recalls, r), (f1s, f)):
            if isinstance(value, (int, float)):
                bucket.append(float(value))
        if isinstance(support, (int, float)):
            supports.append(int(support))

    if precisions:
        out.append("  " + _rule("─", len(header) - 2))
        # Macro, not weighted: a rare class the model cannot do is exactly
        # what a support-weighted average hides.
        out.append(
            f"  {'macro avg':<{name_width}}"
            f"{_num(sum(precisions) / len(precisions)):>10}"
            f"{_num(sum(recalls) / len(recalls)):>9}"
            f"{_num(sum(f1s) / len(f1s)):>9}"
            f"{_count(sum(supports)):>10}"
        )
    return out


def _confusion(s: dict) -> list[str]:
    block = s.get("confusion_matrix")
    if not isinstance(block, dict):
        return []
    labels = block.get("labels") or []
    matrix = block.get("matrix") or []
    if not labels or not matrix:
        return []

    # Plain integers, not grouped: a confusion grid is read by comparing cell
    # magnitudes across a row, and thousands separators widen every column to
    # buy nothing.
    cells = [[_plain(v) for v in row] for row in matrix]
    cell_width = max((len(c) for row in cells for c in row), default=1)
    longest_label = max((len(str(lab)) for lab in labels), default=5)
    name_width = min(longest_label + 1, 22)

    # Spend whatever room is left on readable column headers rather than
    # abbreviating to a fixed four characters: with four classes in a 76-column
    # pane the names fit whole, and "gro…/inv…/loc…" is only necessary once
    # there are enough classes to make it unavoidable.
    available = WIDTH - name_width - 2
    n_cols = max(1, min(len(labels), available // (max(cell_width, 3) + 1)))
    col_width = max(cell_width + 1, min(longest_label + 1, available // n_cols))
    shown = list(range(n_cols))

    head = (
        "  "
        + " " * name_width
        + "".join(f"{_clip(str(labels[i]), col_width - 1):>{col_width}}" for i in shown)
    )
    out = ["", "CONFUSION  (rows = actual, cols = predicted)", head]
    for r, label in enumerate(labels):
        if r >= len(cells):
            break
        row = cells[r]
        out.append(
            f"  {_clip(str(label), name_width - 1):<{name_width}}"
            + "".join(f"{(row[i] if i < len(row) else _MISSING):>{col_width}}" for i in shown)
        )
    if len(labels) > len(shown):
        out.append(f"  (+{len(labels) - len(shown)} more column(s) not shown)")
    return out


def _labels(s: dict) -> list[str]:
    counts = s.get("kept_label_counts")
    if not isinstance(counts, dict) or not counts:
        return []
    total = sum(v for v in counts.values() if isinstance(v, (int, float))) or 1
    out = ["", "LABELLED ROWS"]
    for name, n in sorted(counts.items(), key=lambda kv: -_safe_int(kv[1])):
        share = 100.0 * _safe_int(n) / total
        out.append(f"  {_clip(name, 24):<26}{_count(n):>9}  {share:5.1f}%")
    return out


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _features(s: dict, limit: int = 8) -> list[str]:
    features = s.get("top_features")
    if not isinstance(features, list) or not features:
        return []
    out = ["", "TOP FEATURES"]
    for i, entry in enumerate(features[:limit], 1):
        if not isinstance(entry, dict):
            continue
        name = _clip(str(entry.get("feature", "?")), 40)
        out.append(f"  {i:>2}  {name:<42}{_importance(entry.get('importance')):>8}")
    return out


def _importance(value: Any) -> str:
    """Feature importance, at a precision that suits the backend that made it.

    RandomForest reports normalized fractions, where four decimals *are* the
    signal. LightGBM reports raw split counts, where the same format renders
    ``82.0000`` — four digits of noise implying a precision that is not there.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _MISSING
    return f"{number:,.0f}" if abs(number) >= 1 else f"{number:.4f}"


def _settings(s: dict) -> list[str]:
    bits = []
    weight = s.get("class_weight")
    bits.append(f"class_weight {weight}" if weight else "class_weight none")
    bits.append(f"mirror_augment {'on' if s.get('mirror_augment') else 'off'}")
    stats = s.get("stats")
    if isinstance(stats, list) and stats:
        bits.append(f"stats {', '.join(str(x) for x in stats)}")

    out = ["", "SETTINGS", "  " + _clip(" · ".join(bits), WIDTH - 2)]

    reg = s.get("lgbm_reg")
    if isinstance(reg, dict) and reg:
        knobs = ", ".join(f"{k} {v}" for k, v in reg.items())
        out.append("  " + _clip(f"lgbm: {knobs}", WIDTH - 2))
    return out
