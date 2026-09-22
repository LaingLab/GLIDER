"""The inspector: what the selected range contains, and what the session is.

Selecting a range *is* the analysis in Session Review, so the Range tab fills
the moment a drag ends. The window computes every number; this module only
lays them out.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractScrollArea,
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.gui.review.viewer import TRAIL_DEFAULT_S
from glider.gui.widgets.tool_ui import (
    data_font,
    hint,
    labelled_row,
    scroll_column,
    set_button_role,
    set_text_role,
)

__all__ = ["Inspector", "MarkerRow"]

_NO_RANGE = "No range selected"
_NO_RANGE_HINT = "Drag across the timeline to select a range."


def _section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("SectionTitle")
    return label


def _grow_with_rows(table: QTableWidget) -> QTableWidget:
    """As tall as its rows, inside a scrolling column.

    A table that scrolls inside a column that also scrolls traps the wheel;
    this one never scrolls, so the column does.
    """
    table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.verticalHeader().setVisible(False)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return table


def _dot(colour: QColor) -> QIcon:
    pixmap = QPixmap(8, 8)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(0, 0, 8, 8)
    finally:
        painter.end()
    return QIcon(pixmap)


@dataclass(frozen=True)
class MarkerRow:
    """One line of the Markers tab, already formatted by the window."""

    id: str
    kind: str  # "point" | "range"
    name: str
    colour: QColor
    scope: str  # "session" | "cohort"
    time: str  # "1:00.00 – 4:20.00", or "2:11.50" for a point
    duration: str  # "200.0 s", or "" for a point
    note: str = ""


class _Kpi(QFrame):
    """One big number with its unit underneath."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Kpi")
        column = QVBoxLayout(self)
        column.setContentsMargins(9, 7, 9, 7)
        column.setSpacing(1)
        self.value = QLabel("—")
        self.value.setObjectName("KpiValue")
        self.value.setFont(data_font(15, bold=True))
        label = QLabel(caption)
        set_text_role(label, "caption")
        column.addWidget(self.value)
        column.addWidget(label)


class Inspector(QTabWidget):
    """Range and Session tabs. The window fills them."""

    marker_picked = pyqtSignal(str)  # marker id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Inspector")
        self.setDocumentMode(True)
        self.setMinimumWidth(300)
        self.addTab(self._build_range(), "Range")
        self.addTab(self._build_session(), "Session")
        self.addTab(self._build_markers(), "Markers")

    def _build_range(self) -> QWidget:
        scroll, column = scroll_column()
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(8)

        self.range_card = QFrame()
        self.range_card.setObjectName("RangeCard")
        card = QVBoxLayout(self.range_card)
        card.setContentsMargins(11, 9, 11, 9)
        card.setSpacing(4)
        self.range_title = QLabel(_NO_RANGE)
        self.range_title.setFont(data_font(11))
        self.range_detail = hint(_NO_RANGE_HINT)
        self.inside = hint("")
        self.inside.setVisible(False)
        self.save_marker_btn = QPushButton("Save as range marker")
        self.save_marker_btn.setEnabled(False)
        self.save_marker_btn.setToolTip("Keep this range as a named range marker  (⇧M)")
        self.export_btn = QPushButton("Export…")
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Write the selected range's per-session numbers to a CSV")
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_marker_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.export_btn)
        card.addWidget(self.range_title)
        card.addWidget(self.range_detail)
        card.addWidget(self.inside)
        card.addLayout(buttons)
        column.addWidget(self.range_card)

        column.addWidget(_section("Movement"))
        kpis = QHBoxLayout()
        kpis.setSpacing(6)
        tiles = [_Kpi("cm travelled"), _Kpi("mean cm/s"), _Kpi("peak cm/s")]
        for tile in tiles:
            kpis.addWidget(tile)
        column.addLayout(kpis)
        self.distance, self.mean_speed, self.peak_speed = (t.value for t in tiles)
        self.range_text = hint("")
        column.addWidget(self.range_text)

        column.addWidget(_section("Behavior"))
        self.bouts = _grow_with_rows(QTableWidget(0, 6))
        self.bouts.setHorizontalHeaderLabels(
            ["Behavior", "Bouts", "Total (s)", "Fraction", "Mean (s)", "Median (s)"]
        )
        column.addWidget(self.bouts)

        column.addWidget(_section("Hardware in range"))
        self.hardware = _grow_with_rows(QTableWidget(0, 2))
        self.hardware.horizontalHeader().setVisible(False)
        header = self.hardware.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.hardware.setShowGrid(False)
        self.hardware.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        column.addWidget(self.hardware)

        self.zones_title = _section("Zones")
        column.addWidget(self.zones_title)
        self.zones = _grow_with_rows(QTableWidget(0, 6))
        self.zones.setHorizontalHeaderLabels(
            ["Zone", "Time (s)", "Fraction", "Entries", "Mean bout (s)", "Latency (s)"]
        )
        column.addWidget(self.zones)
        column.addStretch(1)
        return scroll

    def _build_session(self) -> QWidget:
        scroll, column = scroll_column()
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(10)
        self.summary = hint("No session loaded.")
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.summary)
        self.fixes = QHBoxLayout()
        self.fixes.setSpacing(6)
        column.addLayout(self.fixes)
        self.trail_s = QDoubleSpinBox()
        self.trail_s.setRange(0.5, 60.0)
        self.trail_s.setValue(TRAIL_DEFAULT_S)
        self.trail_s.setSuffix(" s")
        column.addLayout(labelled_row("Trail length", self.trail_s))
        column.addStretch(1)
        return scroll

    def _build_markers(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(8)
        chips = QHBoxLayout()
        chips.setSpacing(4)
        group = QButtonGroup(self)
        self.marker_filters: dict[str, QToolButton] = {}
        for key, text in (("all", "All"), ("range", "Ranges"), ("point", "Points")):
            button = QToolButton()
            button.setObjectName("TimelineTool")
            button.setText(text)
            button.setCheckable(True)
            button.setChecked(key == "all")
            button.toggled.connect(self._apply_marker_filter)
            group.addButton(button)
            chips.addWidget(button)
            self.marker_filters[key] = button
        chips.addStretch(1)
        column.addLayout(chips)
        self.marker_note = hint("")
        self.marker_note.setVisible(False)
        column.addWidget(self.marker_note)
        self.marker_list = QListWidget()
        self.marker_list.setObjectName("MarkerList")
        self.marker_list.itemClicked.connect(
            lambda item: self.marker_picked.emit(item.data(Qt.ItemDataRole.UserRole))
        )
        column.addWidget(self.marker_list, 1)
        footer = QHBoxLayout()
        self.epoch_btn = QPushButton("Epoch table ▸")
        set_button_role(self.epoch_btn, "primary")
        self.export_markers_btn = QPushButton("Export markers…")
        footer.addWidget(self.epoch_btn)
        footer.addWidget(self.export_markers_btn)
        footer.addStretch(1)
        column.addLayout(footer)
        return page

    # ------------------------------------------------------------------

    def set_range(self, title: str, detail: str) -> None:
        self.range_title.setText(title)
        self.range_detail.setText(detail)

    def clear_range(self) -> None:
        self.set_range(_NO_RANGE, _NO_RANGE_HINT)
        self.set_kpis("—", "—", "—")
        self.range_text.setText("")
        self.set_hardware([])
        for table in (self.bouts, self.zones):
            table.setRowCount(0)
            table.updateGeometry()
            table.setMaximumHeight(table.sizeHint().height())
        self.set_zone_count(0)
        self.set_inside([])
        self.save_marker_btn.setEnabled(False)

    def set_kpis(self, distance: str, mean: str, peak: str) -> None:
        self.distance.setText(distance)
        self.mean_speed.setText(mean)
        self.peak_speed.setText(peak)

    def set_hardware(self, rows: list[tuple[str, str, QColor]]) -> None:
        self.hardware.setRowCount(len(rows))
        for r, (name, text, colour) in enumerate(rows):
            item = QTableWidgetItem(name)
            item.setIcon(_dot(colour))
            value = QTableWidgetItem(text)
            value.setFont(data_font())
            value.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.hardware.setItem(r, 0, item)
            self.hardware.setItem(r, 1, value)
        self.hardware.updateGeometry()
        self.hardware.setMaximumHeight(self.hardware.sizeHint().height())

    def set_zone_count(self, n: int) -> None:
        self.zones_title.setText("ZONES" if n == 0 else f"ZONES ({n})")

    def set_markers(self, rows: list[MarkerRow]) -> None:
        """Ranges then points, as the window orders them; the filter survives."""
        self.marker_list.clear()
        for row in rows:
            scope = "cohort" if row.scope == "cohort" else "this session"
            name = row.name or ("Range" if row.kind == "range" else "Marker")
            detail = "  ·  ".join(part for part in (row.time, row.duration, row.note) if part)
            item = QListWidgetItem(_dot(row.colour), f"{name}   [{scope}]\n{detail}")
            item.setData(Qt.ItemDataRole.UserRole, row.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, row.kind)
            self.marker_list.addItem(item)
        self.setTabText(2, f"Markers ({len(rows)})" if rows else "Markers")
        self._apply_marker_filter()

    def _apply_marker_filter(self, *_args) -> None:
        kind = next((k for k, b in self.marker_filters.items() if b.isChecked()), "all")
        for i in range(self.marker_list.count()):
            item = self.marker_list.item(i)
            item.setHidden(kind != "all" and item.data(Qt.ItemDataRole.UserRole + 1) != kind)

    def set_marker_note(self, text: str) -> None:
        """Why markers cannot be edited here, or nothing."""
        self.marker_note.setText(text)
        self.marker_note.setVisible(bool(text))

    def set_inside(self, names: list[str]) -> None:
        """The range markers the selection falls inside."""
        self.inside.setText("inside " + ", ".join(names) if names else "")
        self.inside.setVisible(bool(names))
