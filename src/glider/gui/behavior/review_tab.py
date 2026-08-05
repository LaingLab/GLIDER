"""Review tab: read a training run instead of parsing it.

Sits between Train and Apply, and opens by itself when a run finishes, because
that is the moment the question "is this model good enough to score a cohort
with?" actually gets asked. Answering it used to mean reading a
seventy-six-column monospace report and knowing which of its numbers were
load-bearing.

What the layout encodes:

* One headline number, sized so it is the first thing read, with how it was
  measured directly under it. A run with no held-out set gets an amber number
  and a warning, because 0.998 on its own training rows is the single most
  misleading thing this pipeline can print.
* Warnings before tables. Class imbalance and a collapsed class are the two
  findings that change what the operator does next, and both were previously
  derivable only by comparing numbers across two different sections.
* Per-class rows sorted worst-F1 first, with the F1 drawn as a bar. The reason
  to open the table is to find the failure.
* The confusion matrix row-normalized and drawn as a heat grid, so "of the
  frames that really were rearing, where did they go" is one glance rather
  than a division per cell.

Everything computed lives in :mod:`glider.analysis.behavior.run_report`, which
is Qt-free and tested without a display. This module only draws.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from glider.analysis.behavior.run_report import RunReportError, TrainingRun
from glider.gui.behavior.embedding_view import EmbeddingView
from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import (
    CARD_GAP,
    GUTTER,
    Card,
    caption,
    data_font,
    hint,
    scroll_column,
    set_button_role,
    set_text_role,
)

logger = logging.getLogger(__name__)

#: The data face as a QSS font-family list. tool_ui.data_font() is the same
#: stack for QFont users; a style sheet cannot read that, so it is restated
#: here for the one place that must win against the theme's font-size rule.
_MONO_STACK = '"SF Mono", "JetBrains Mono", Menlo, Consolas, monospace'

#: Strength of evidence -> the colour the headline number is drawn in. Amber
#: is not decoration: it is the difference between a score to act on and one
#: measured on the training data.
_STRENGTH_COLOR = {
    "strong": colors.SUCCESS,
    "fair": colors.ACCENT,
    "weak": colors.WARNING,
}


class MetricTile(QFrame):
    """One big number with a label over it and a caption under it."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("MetricTile")
        self.setStyleSheet(
            f"QFrame#MetricTile {{ background: {colors.BASE}; "
            f"border: 1px solid {colors.BORDER}; border-radius: 8px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(2)

        self._label = set_text_role(QLabel(label.upper()), "hint")
        layout.addWidget(self._label)

        # Size and family come from the stylesheet, not setFont(). The tool
        # theme carries a `QWidget { font-size: 12px }` rule, and a style
        # sheet beats a programmatically-set QFont however specific the font
        # is -- setting 26px here through setFont() silently rendered at 12.
        self._value = QLabel("—")
        self._value.setObjectName("MetricValue")
        self._colorize(colors.TEXT_PRIMARY)
        layout.addWidget(self._value)

        self._caption = set_text_role(QLabel(""), "hint")
        self._caption.setWordWrap(True)
        self._caption.setVisible(False)
        layout.addWidget(self._caption)

    def _colorize(self, color: str) -> None:
        self._value.setStyleSheet(
            f"QLabel#MetricValue {{ color: {color}; font-size: 25px; font-weight: 700; "
            f"font-family: {_MONO_STACK}; }}"
        )

    def set_value(self, text: str, *, color: str = colors.TEXT_PRIMARY) -> None:
        self._value.setText(text)
        self._colorize(color)

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)
        self._caption.setVisible(bool(text))


class ConfusionMatrix(QWidget):
    """The confusion matrix as a heat grid, read row by row.

    Row-normalized: each row is "of the frames that really were this
    behavior, where did the model put them". Raw counts cannot be compared
    across rows when one behavior has fifty times another's support, which on
    real annotation is normal.

    Drawn rather than rendered through matplotlib because the report's PNGs
    are matplotlib's default light style and would sit in this dark window as
    a white rectangle.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: list[str] = []
        self._matrix: list[list[float]] = []
        self._counts: list[list[float]] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_matrix(self, labels, normalized, counts) -> None:
        self._labels = list(labels)
        self._matrix = [list(r) for r in normalized]
        self._counts = [list(r) for r in counts]
        n = len(self._labels)
        # Header row + one row per class, plus room for the rotated-free
        # column labels along the top.
        self.setMinimumHeight(0 if not n else 34 + n * 26)
        self.updateGeometry()
        self.update()

    def _gutter(self) -> int:
        """Width reserved for the row labels."""
        if not self._labels:
            return 0
        metrics = self.fontMetrics()
        widest = max(metrics.horizontalAdvance(_clip(lab)) for lab in self._labels)
        return min(max(widest + 12, 70), 180)

    def paintEvent(self, _event):
        if not self._labels:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self._labels)
        gutter = self._gutter()
        cell_w = max(1.0, (self.width() - gutter) / n)
        cell_h = 26.0
        top = 34.0

        painter.setFont(self.font())
        # Column headers, clipped to the cell so a long behavior name cannot
        # overrun its neighbour.
        painter.setPen(QPen(QColor(colors.TEXT_MUTED)))
        for j, label in enumerate(self._labels):
            rect = QRectF(gutter + j * cell_w, 8.0, cell_w, 22.0)
            painter.drawText(
                rect,
                int(Qt.AlignmentFlag.AlignCenter),
                _elide(label, painter, cell_w - 6),
            )

        accent = QColor(colors.ACCENT)
        for i, label in enumerate(self._labels):
            y = top + i * cell_h
            painter.setPen(QPen(QColor(colors.TEXT_SECONDARY)))
            painter.drawText(
                QRectF(0, y, gutter - 8, cell_h),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                _elide(label, painter, gutter - 12),
            )
            for j in range(n):
                value = self._matrix[i][j] if j < len(self._matrix[i]) else 0.0
                rect = QRectF(gutter + j * cell_w + 1, y + 1, cell_w - 2, cell_h - 2)
                # The diagonal is what the model got right, so it is the
                # accent; everything off it is an error and reads red. A
                # single hue for both would make a confident wrong answer
                # look like a confident right one.
                base = accent if i == j else QColor(colors.ERROR)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(colors.qcolor_with_alpha(base, 0.10 + 0.75 * value)))
                painter.drawRoundedRect(rect, 3, 3)
                if value >= 0.005:
                    painter.setPen(
                        QPen(QColor(colors.TEXT_PRIMARY if value > 0.45 else colors.TEXT_MUTED))
                    )
                    painter.setFont(data_font())
                    painter.drawText(
                        rect, int(Qt.AlignmentFlag.AlignCenter), f"{round(value * 100):d}"
                    )
                    painter.setFont(self.font())

    def cell_tooltip(self) -> str:
        return (
            "Each row is one true behavior, and sums to 100%. A cell says what "
            "share of that behavior's frames the model called the column's "
            "label — so the diagonal is correct and everything else is a "
            "specific confusion."
        )


class ScoreBar(QWidget):
    """A 0–1 score as a labelled bar. Used for per-class F1 and importances."""

    def __init__(self, value: float, color: str, parent=None):
        super().__init__(parent)
        self._value = max(0.0, min(1.0, float(value)))
        self._color = color
        self.setFixedHeight(16)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(0, 5, self.width(), 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(colors.BASE)))
        painter.drawRoundedRect(track, 3, 3)
        if self._value > 0:
            filled = QRectF(0, 5, self.width() * self._value, 6)
            painter.setBrush(QBrush(QColor(self._color)))
            painter.drawRoundedRect(filled, 3, 3)


class ReviewTab(QWidget):
    """Show one training run's report."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run: TrainingRun | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(GUTTER, GUTTER, GUTTER, GUTTER)
        outer.setSpacing(CARD_GAP)

        area, self._column = scroll_column()
        outer.addWidget(area, 1)

        self._empty = Card("No run loaded")
        self._empty.add(
            hint(
                "Fit a model on the Train tab and its report opens here "
                "automatically.\n\nReports are written next to the model "
                "bundle, so a run from any earlier session can be opened too."
            )
        )
        open_btn = QPushButton("Open a report folder…")
        open_btn.clicked.connect(self.open_report_dialog)
        self._empty.add(open_btn)
        self._column.addWidget(self._empty)
        self._column.addStretch(1)

        self._cards: list[QWidget] = []

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def open_report_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open a training report folder")
        if folder:
            self.load(Path(folder))

    def load(self, report_dir: Path) -> bool:
        """Show the run in *report_dir*. Returns whether it could be read."""
        try:
            run = TrainingRun.load(report_dir)
        except RunReportError as e:
            QMessageBox.warning(self, "Open training report", str(e))
            return False
        self.show_run(run)
        return True

    def show_run(self, run: TrainingRun) -> None:
        """Rebuild the tab for *run*."""
        self._run = run
        self._clear()
        self._empty.setVisible(False)
        for card in self._build_cards(run):
            self._column.insertWidget(self._column.count() - 1, card)
            self._cards.append(card)

    def _clear(self) -> None:
        for card in self._cards:
            self._column.removeWidget(card)
            card.deleteLater()
        self._cards = []

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_cards(self, run: TrainingRun) -> list[QWidget]:
        cards = [self._headline_card(run)]
        warnings = self._warnings_card(run)
        if warnings is not None:
            cards.append(warnings)
        per_class = self._per_class_card(run)
        if per_class is not None:
            cards.append(per_class)
        confusion = self._confusion_card(run)
        if confusion is not None:
            cards.append(confusion)
        embedding = self._embedding_card(run)
        if embedding is not None:
            cards.append(embedding)
        balance = self._balance_card(run)
        if balance is not None:
            cards.append(balance)
        features = self._features_card(run)
        if features is not None:
            cards.append(features)
        cards.append(self._settings_card(run))
        return cards

    def _headline_card(self, run: TrainingRun) -> Card:
        summary = run.summary
        card = Card("Result", str(summary.get("classifier_type") or ""))
        if run.path is not None:
            card.set_badge(run.path.name)

        head = run.headline
        tiles = QHBoxLayout()
        tiles.setSpacing(CARD_GAP)

        score = MetricTile(head.label)
        text = "—" if head.value is None else f"{head.value:.3f}"
        if head.spread is not None:
            text += f"  ± {head.spread:.3f}"
        score.set_value(text, color=_STRENGTH_COLOR.get(head.strength, colors.TEXT_PRIMARY))
        score.set_caption(head.caption)
        tiles.addWidget(score, 2)

        macro = MetricTile("Macro F1")
        value = run.macro_f1
        macro.set_value("—" if value is None else f"{value:.3f}")
        macro.set_caption("unweighted, so a rare class still counts")
        tiles.addWidget(macro, 1)

        weakest = run.weakest
        worst = MetricTile("Weakest class")
        if weakest is not None and weakest.f1 is not None:
            worst.set_value(
                f"{weakest.f1:.2f}",
                color=colors.ERROR if weakest.f1 < 0.5 else colors.TEXT_PRIMARY,
            )
            worst.set_caption(weakest.name)
        else:
            worst.set_caption("nothing measured")
        tiles.addWidget(worst, 1)

        rows = MetricTile("Rows trained")
        rows.set_value(f"{summary.get('n_rows_kept') or 0:,}")
        sessions = summary.get("n_sessions")
        holdout = summary.get("n_holdout_sessions") or 0
        detail = f"{sessions or 0} session{'' if sessions == 1 else 's'}"
        if holdout:
            detail += f" + {holdout} holdout"
        rows.set_caption(detail)
        tiles.addWidget(rows, 1)

        card.add(tiles)
        return card

    def _warnings_card(self, run: TrainingRun) -> Card | None:
        messages = run.warnings
        if not messages:
            return None
        card = Card("Worth knowing")
        card.set_badge(f"{len(messages)}")
        for message in messages:
            row = QHBoxLayout()
            row.setSpacing(8)
            marker = set_text_role(QLabel("▲"), "warning")
            marker.setAlignment(Qt.AlignmentFlag.AlignTop)
            row.addWidget(marker)
            text = QLabel(message)
            text.setWordWrap(True)
            set_text_role(text, "caption")
            row.addWidget(text, 1)
            card.add(row)
        return card

    def _per_class_card(self, run: TrainingRun) -> Card | None:
        rows = run.per_class
        if not rows:
            return None
        card = Card("Per behavior", "worst first")
        card.set_badge(f"{len(rows)}")

        header = QHBoxLayout()
        header.setSpacing(8)
        for text, width, stretch in (
            ("Behavior", 130, 0),
            ("F1", 0, 1),
            ("Precision", 70, 0),
            ("Recall", 60, 0),
            ("Rows", 70, 0),
        ):
            label = set_text_role(QLabel(text.upper()), "hint")
            if width:
                label.setFixedWidth(width)
            if text in ("Precision", "Recall", "Rows"):
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            header.addWidget(label, stretch)
        card.add(header)

        for metric in rows:
            card.add(self._per_class_row(metric))
        return card

    def _per_class_row(self, metric) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        name = QLabel(metric.name)
        name.setFixedWidth(130)
        set_text_role(name, "value")
        row.addWidget(name)

        f1 = metric.f1 or 0.0
        # Traffic-light on the score itself, so scanning the column finds the
        # failures without reading a number.
        color = colors.SUCCESS if f1 >= 0.8 else colors.WARNING if f1 >= 0.5 else colors.ERROR
        bar_holder = QHBoxLayout()
        bar_holder.setSpacing(8)
        bar_holder.addWidget(ScoreBar(f1, color), 1)
        value = QLabel(f"{f1:.2f}")
        value.setFont(data_font())
        value.setFixedWidth(38)
        value.setStyleSheet(f"color: {color};")
        bar_holder.addWidget(value)
        row.addLayout(bar_holder, 1)

        for text, width in (
            (_num(metric.precision), 70),
            (_num(metric.recall), 60),
            (f"{metric.support:,}" if metric.support is not None else "—", 70),
        ):
            cell = QLabel(text)
            cell.setFont(data_font())
            cell.setFixedWidth(width)
            cell.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            set_text_role(cell, "caption")
            row.addWidget(cell)
        return row

    def _embedding_card(self, run: TrainingRun) -> Card | None:
        """The learned feature space as a rotatable 3D scatter.

        Omitted entirely when the run has no embedding: most do not, and an
        empty panel inviting a drag that does nothing is worse than no panel.
        """
        points = run.embedding
        if points is None or len(getattr(points, "coords", ())) == 0:
            return None
        card = Card("Feature space", "what the classifier learned")
        view = EmbeddingView()
        view.set_artifact(points)
        card.add(view)
        card.add(
            hint(
                "Each point is a labelled window, reduced to three dimensions. "
                "Classes in their own regions are separable; classes sitting on "
                "top of each other are the confusion matrix, seen directly."
            )
        )
        return card

    def _confusion_card(self, run: TrainingRun) -> Card | None:
        labels, normalized = run.confusion_rows_normalized()
        if not labels:
            return None
        _labels, counts = run.confusion
        card = Card("Confusion", "row %, true behavior by row")
        grid = ConfusionMatrix()
        grid.set_matrix(labels, normalized, counts)
        grid.setToolTip(grid.cell_tooltip())
        card.add(grid)
        card.add(
            hint(
                "Each row sums to 100%. The diagonal is correct; a bright cell "
                "off it is a specific confusion worth labelling more of."
            )
        )
        return card

    def _balance_card(self, run: TrainingRun) -> Card | None:
        counts = run.label_counts
        if not counts:
            return None
        card = Card("Training rows", "per behavior")
        ratio = run.imbalance_ratio
        if ratio is not None:
            card.set_badge(f"{ratio:,.0f}:1")
        biggest = counts[0][1] or 1
        for label, n in counts:
            row = QHBoxLayout()
            row.setSpacing(8)
            name = QLabel(label)
            name.setFixedWidth(130)
            set_text_role(name, "value")
            row.addWidget(name)
            row.addWidget(ScoreBar(n / biggest, colors.ACCENT), 1)
            value = QLabel(f"{n:,}")
            value.setFont(data_font())
            value.setFixedWidth(80)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            set_text_role(value, "caption")
            row.addWidget(value)
            card.add(row)
        return card

    def _features_card(self, run: TrainingRun) -> Card | None:
        features = run.top_features[:10]
        if not features:
            return None
        card = Card("Top features", "what the model leans on")
        biggest = max(v for _f, v in features) or 1.0
        for name, value in features:
            row = QHBoxLayout()
            row.setSpacing(8)
            label = QLabel(name)
            label.setFixedWidth(220)
            label.setToolTip(name)
            set_text_role(label, "caption")
            row.addWidget(label)
            row.addWidget(ScoreBar(value / biggest, colors.INFO), 1)
            card.add(row)
        return card

    def _settings_card(self, run: TrainingRun) -> Card:
        summary = run.summary
        card = Card("Settings", "what produced this run")
        window = summary.get("window")
        fps = summary.get("fps")
        pairs = [
            ("Backend", summary.get("classifier_type")),
            ("Split", summary.get("split_strategy")),
            (
                "Window",
                (
                    f"{window} frames ({window / fps:g} s)"
                    if window and fps
                    else (f"{window} frames" if window else None)
                ),
            ),
            ("Features", summary.get("n_features")),
            ("Class weight", summary.get("class_weight") or "none"),
            ("Mirror augment", "yes" if summary.get("mirror_augment") else "no"),
        ]
        for label, value in pairs:
            if value in (None, ""):
                continue
            row = QHBoxLayout()
            row.setSpacing(8)
            key = caption(label)
            key.setFixedWidth(130)
            row.addWidget(key)
            shown = QLabel(str(value))
            set_text_role(shown, "value")
            row.addWidget(shown, 1)
            card.add(row)

        if run.path is not None:
            card.add_separator()
            where = hint(f"Report folder: {run.path}")
            where.setToolTip(str(run.path))
            card.add(where)
            reveal = QPushButton("Open a different report…")
            set_button_role(reveal, "ghost")
            reveal.clicked.connect(self.open_report_dialog)
            card.add(reveal)
        return card


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _num(value, places: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{places}f}"


def _clip(text: str, limit: int = 22) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _elide(text: str, painter: QPainter, width: float) -> str:
    """Trim *text* to fit *width*, marking that something was cut."""
    metrics = painter.fontMetrics()
    if metrics.horizontalAdvance(text) <= width:
        return text
    trimmed = text
    while trimmed and metrics.horizontalAdvance(trimmed + "…") > width:
        trimmed = trimmed[:-1]
    return (trimmed + "…") if trimmed else ""


__all__ = ["ConfusionMatrix", "MetricTile", "ReviewTab", "ScoreBar"]
