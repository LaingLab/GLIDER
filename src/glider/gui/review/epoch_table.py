"""The epoch table: sessions down, epochs across, a metric per column.

The timeline panel's second tab. Epochs are the cohort's range markers --
Baseline, Stim, Post -- plus the current selection, each applied to every
session on its own zero, so a row reads straight across as one animal's
protocol. The window computes the rows; this widget lays them out.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.analysis.epochs import (
    Epoch,
    Metric,
    default_metrics,
    metric_text,
    metric_value,
    missing_reason,
    summarize,
)
from glider.gui.review.timeline import swatch
from glider.gui.review.viewport import format_seconds
from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import caption, data_font, hint

__all__ = ["BAR_ROLE", "EpochSession", "EpochTable"]

#: A cell's value as a fraction of its column's largest.
BAR_ROLE = Qt.ItemDataRole.UserRole + 1

_T0 = {"flow_start": "flow start", "video_start": "the first video frame"}


@dataclass(frozen=True)
class EpochSession:
    """A session as the table lists it."""

    index: int  # into the window's cohort
    name: str
    group: str


class _DataBar(QStyledItemDelegate):
    """A faint bar behind a number, scaled to its column's largest value."""

    def paint(self, painter, option, index):
        fraction = index.data(BAR_ROLE)
        if isinstance(fraction, float) and fraction > 0:
            rect = QRectF(option.rect).adjusted(2, 5, -2, -5)
            rect.setWidth(rect.width() * min(1.0, fraction))
            painter.fillRect(rect, colors.qcolor_with_alpha(QColor(colors.ACCENT), 0.13))
        super().paint(painter, option, index)


def _epoch_icon(epoch: Epoch) -> QIcon:
    """A saved range's swatch; the current range as an outline, as on the timeline."""
    pixmap = QPixmap(10, 10)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if epoch.color:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(swatch(epoch.color))
        else:
            painter.setPen(QPen(QColor(colors.ACCENT), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(1, 1, 8, 8), 2, 2)
    finally:
        painter.end()
    return QIcon(pixmap)


def _tool(text: str) -> QToolButton:
    button = QToolButton()
    button.setObjectName("TimelineTool")
    button.setText(text)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def _label(text: str, font: QFont | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    if font is not None:
        item.setFont(font)
    return item


def _measure(text: str, font: QFont | None = None) -> QTableWidgetItem:
    """Monospace and right-aligned, so a column reads downward."""
    item = QTableWidgetItem(text)
    item.setFont(font if font is not None else data_font())
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


class EpochTable(QWidget):
    """Every session against every epoch; the window fills it."""

    session_picked = pyqtSignal(int)
    session_opened = pyqtSignal(int)
    export_requested = pyqtSignal(str)  # "tidy" | "wide"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("EpochTable")
        self._epochs: list[Epoch] = []
        self._sessions: list[EpochSession] = []
        self._rows: dict[str, list[dict]] = {}
        self._catalog: list[Metric] = []
        self._off: set[str] = set()  # epoch keys switched off
        self._chosen: list[str] | None = None  # metric keys; None = the defaults
        self._row_index: dict[int, int] = {}  # table row -> cohort index

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("EpochBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(4)
        row.addWidget(caption("Columns"))
        self._chip_row = QHBoxLayout()
        self._chip_row.setSpacing(4)
        row.addLayout(self._chip_row)
        self.chips: dict[str, QToolButton] = {}
        self.metrics_btn = _tool("Metrics ▾")
        self.metrics_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.metrics_btn.setMenu(QMenu(self.metrics_btn))
        row.addSpacing(8)
        row.addWidget(self.metrics_btn)
        row.addStretch(1)
        self.export_btn = _tool("Export ▾")
        self.export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export = QMenu(self.export_btn)
        export.addAction("Tidy CSV…", lambda: self.export_requested.emit("tidy"))
        export.addAction("Wide CSV…", lambda: self.export_requested.emit("wide"))
        self.export_btn.setMenu(export)
        row.addWidget(self.export_btn)
        column.addWidget(bar)

        self.table = QTableWidget()
        self.table.setObjectName("EpochGrid")
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        self.table.setItemDelegate(_DataBar(self.table))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellClicked.connect(self._clicked)
        self.table.cellDoubleClicked.connect(self._double_clicked)
        self.empty = hint(
            "Save a range marker as Whole cohort, or select a range, to fill this table."
        )
        self.empty.setContentsMargins(12, 12, 12, 12)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        column.addWidget(self.table, 1)
        column.addWidget(self.empty, 1)
        self._render()

    # ------------------------------------------------------------------
    # what is shown

    def set_data(self, epochs, sessions, rows, catalog) -> None:
        """Everything the table shows. ``rows[epoch.key]`` is one row per session."""
        self._epochs, self._sessions = list(epochs), list(sessions)
        self._rows, self._catalog = dict(rows), list(catalog)
        self._rebuild_chips()
        self._rebuild_metric_menu()
        self._render()

    def clear(self) -> None:
        self.set_data([], [], {}, [])

    def visible(self, epochs) -> list[Epoch]:
        """The given epochs, less those switched off with the chips."""
        return [e for e in epochs if e.key not in self._off]

    def epochs(self) -> list[Epoch]:
        return self.visible(self._epochs)

    def metric_keys(self, catalog) -> list[str]:
        """The chosen metrics that ``catalog`` offers, in its order.

        Falls back to the catalog's defaults when the choice offers nothing --
        a metric picked against one cohort's catalog can be entirely absent
        from the next refill's. The choice itself (``set_metrics``) is left
        untouched, so a metric that returns later is still the user's pick.
        """
        chosen = self._chosen if self._chosen is not None else default_metrics(catalog)
        keys = [m.key for m in catalog if m.key in chosen]
        if not keys and catalog:
            keys = [m.key for m in catalog if m.key in default_metrics(catalog)]
        return keys

    def metrics(self) -> list[Metric]:
        keys = set(self.metric_keys(self._catalog))
        return [m for m in self._catalog if m.key in keys]

    def set_metrics(self, keys) -> None:
        self._chosen = list(keys)
        self._rebuild_metric_menu()
        self._render()

    def session_count(self) -> int:
        return len(self._row_index)

    def row_of(self, index: int) -> int | None:
        """The table row showing cohort session ``index``."""
        return next((r for r, i in self._row_index.items() if i == index), None)

    def set_shown(self, index: int) -> None:
        """Highlight the session on screen, without reporting it as picked."""
        row = self.row_of(index)
        self.table.blockSignals(True)
        if row is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(row)
        self.table.blockSignals(False)

    # ------------------------------------------------------------------

    def _rebuild_chips(self) -> None:
        for chip in self.chips.values():
            self._chip_row.removeWidget(chip)
            chip.deleteLater()
        self.chips = {}
        for epoch in self._epochs:
            chip = _tool(epoch.name)
            chip.setIcon(_epoch_icon(epoch))
            chip.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            chip.setCheckable(True)
            chip.setChecked(epoch.key not in self._off)
            chip.toggled.connect(lambda on, key=epoch.key: self._toggle_epoch(key, on))
            self._chip_row.addWidget(chip)
            self.chips[epoch.key] = chip

    def _toggle_epoch(self, key: str, on: bool) -> None:
        if on:
            self._off.discard(key)
        else:
            self._off.add(key)
        self._render()

    def _rebuild_metric_menu(self) -> None:
        menu = self.metrics_btn.menu()
        menu.clear()
        chosen = set(self.metric_keys(self._catalog))
        for metric in self._catalog:
            action = menu.addAction(metric.label)
            action.setCheckable(True)
            action.setChecked(metric.key in chosen)
            action.toggled.connect(lambda on, key=metric.key: self._toggle_metric(key, on))

    def _toggle_metric(self, key: str, on: bool) -> None:
        keys = self.metric_keys(self._catalog)
        keys = [k for k in keys if k != key] + ([key] if on else [])
        self._chosen = keys
        self._render()

    def _by_group(self) -> list[tuple[str, list[tuple[int, EpochSession]]]]:
        groups: dict[str, list[tuple[int, EpochSession]]] = {}
        for pos, session in enumerate(self._sessions):
            groups.setdefault(session.group, []).append((pos, session))
        return sorted(groups.items(), key=lambda kv: (kv[0] == "", kv[0]))

    def _add_row(self) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)
        return row

    def _render(self) -> None:
        epochs, metrics = self.epochs(), self.metrics()
        table = self.table
        table.clearSpans()
        table.clear()
        table.setRowCount(0)
        self._row_index = {}
        ready = bool(epochs and metrics and self._sessions)
        table.setVisible(ready)
        self.empty.setVisible(not ready)
        if not ready:
            table.setColumnCount(0)
            return
        per = len(metrics)
        table.setColumnCount(1 + per * len(epochs))
        table.setHorizontalHeaderLabels(["Session"] + [m.label for _ in epochs for m in metrics])
        top = {
            (e.key, m.key): max(
                (v for v in (metric_value(r, m.key) for r in self._rows[e.key]) if v and v > 0),
                default=0.0,
            )
            for e in epochs
            for m in metrics
        }
        t0s = Counter(r.get("t0") for r in self._rows[epochs[0].key] if r.get("t0"))
        majority = t0s.most_common(1)[0][0] if t0s else None
        bold = QFont(table.font())
        bold.setBold(True)

        head = self._add_row()
        for i, epoch in enumerate(epochs):
            span = f"{format_seconds(epoch.start_s)}–{format_seconds(epoch.end_s)}"
            item = _label(f"{epoch.name}   {span}", bold)
            item.setIcon(_epoch_icon(epoch))
            table.setItem(head, 1 + i * per, item)
            if per > 1:
                table.setSpan(head, 1 + i * per, 1, per)

        groups = self._by_group()
        titled = len(groups) > 1 or any(name for name, _ in groups)
        for group, members in groups:
            if titled:
                r = self._add_row()
                table.setItem(r, 0, _label(group or "Ungrouped", bold))
                table.setSpan(r, 0, 1, table.columnCount())
            for pos, session in members:
                r = self._add_row()
                self._row_index[r] = session.index
                name = QTableWidgetItem(session.name)
                t0 = self._rows[epochs[0].key][pos].get("t0")
                if majority and t0 and t0 != majority:
                    name.setText(f"⚠ {session.name}")
                    name.setToolTip(
                        f"This session's zero is {_T0.get(t0, t0)}; most are "
                        f"{_T0.get(majority, majority)}. Every range is measured "
                        "from each session's own zero."
                    )
                table.setItem(r, 0, name)
                for i, epoch in enumerate(epochs):
                    row = self._rows[epoch.key][pos]
                    for j, metric in enumerate(metrics):
                        cell = self._cell(row, metric, top[(epoch.key, metric.key)])
                        table.setItem(r, 1 + i * per + j, cell)
            self._summary_rows(group, members, epochs, metrics, bold)

    def _cell(self, row: dict, metric: Metric, largest: float) -> QTableWidgetItem:
        value = metric_value(row, metric.key)
        item = _measure(metric_text(row, metric))
        if value is None:
            item.setToolTip(missing_reason(row, metric.key))
        elif largest > 0:
            item.setData(BAR_ROLE, max(0.0, value) / largest)
        return item

    def _summary_rows(self, group, members, epochs, metrics, bold) -> None:
        per = len(metrics)
        mean_row, sem_row = self._add_row(), self._add_row()
        self.table.setItem(mean_row, 0, _label(f"{group} mean" if group else "Mean", bold))
        self.table.setItem(sem_row, 0, _label("SEM"))
        for i, epoch in enumerate(epochs):
            block = self._rows[epoch.key]
            for j, metric in enumerate(metrics):
                mean, sem = summarize(metric_value(block[pos], metric.key) for pos, _ in members)
                column = 1 + i * per + j
                d = metric.decimals
                self.table.setItem(
                    mean_row, column, _measure("—" if mean is None else f"{mean:.{d}f}", bold)
                )
                self.table.setItem(
                    sem_row, column, _measure("" if sem is None else f"± {sem:.{d}f}")
                )

    def _clicked(self, row: int, _column: int) -> None:
        if row in self._row_index:
            self.session_picked.emit(self._row_index[row])

    def _double_clicked(self, row: int, _column: int) -> None:
        if row in self._row_index:
            self.session_opened.emit(self._row_index[row])
