"""The Retrieval tab: a retrieval session's setup and each pup's result.

It only lays out what the window hands it and says what was clicked; the
window owns the sidecars and does the scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from glider.gui.review.viewport import format_seconds
from glider.gui.widgets.tool_ui import scroll_column

if TYPE_CHECKING:
    from glider.analysis.behavior.retrieval_session import SessionRetrieval
    from glider.analysis.behavior.retrieval_setup import RetrievalProtocol, RetrievalSetup

__all__ = ["RetrievalPanel"]

# Two-line heads keep the units and let all five columns fit a 300 px inspector.
_COLUMNS = ("Pup", "Contact\n(s)", "Retrieved\n(s)", "Carries", "Drops")


def _clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _num(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.1f}"


class RetrievalPanel(QWidget):
    set_trial_start = pyqtSignal()
    redraw_nest = pyqtSignal()
    remove_nest = pyqtSignal(int)  # index into the nest periods
    jump_to = pyqtSignal(float)  # seconds from the video's first frame
    set_scale = pyqtSignal()
    edit_protocol = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll, column = scroll_column()
        # scroll_column never scrolls sideways; a pane narrower than a row would
        # cut it off with no way to reach the rest.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(8)
        outer.addWidget(scroll)

        column.addWidget(self._title("SETUP"))
        self.nest_list = QListWidget()
        self.nest_list.setMaximumHeight(72)
        self.nest_list.itemClicked.connect(self._on_nest_clicked)
        self.nest_button = QPushButton("Draw nest")
        self.nest_button.clicked.connect(self.redraw_nest)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._on_remove)
        # A form, not hand-built rows: where a label, value and button do not
        # fit side by side (Windows' wider text in a 300 px inspector), the
        # value and button wrap under the label instead of running off the pane.
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.addRow("Nest", self.nest_button)
        # Remove sits with the list it acts on: beside the draw button the row
        # was wider than the inspector.
        periods = QHBoxLayout()
        periods.addWidget(self.nest_list)
        periods.addWidget(self.remove_button, 0, Qt.AlignmentFlag.AlignTop)
        form.addRow(periods)
        self.trial_label = QLabel()
        self.trial_button = QPushButton("Set at playhead")
        self.trial_button.clicked.connect(self.set_trial_start)
        form.addRow("Trial start", self._pair(self.trial_label, self.trial_button))
        self.scale_label = QLabel()
        self.scale_button = QPushButton("Set scale…")
        self.scale_button.clicked.connect(self.set_scale)
        form.addRow("Scale", self._pair(self.scale_label, self.scale_button))
        self.protocol_label = QLabel()
        self.protocol_label.setWordWrap(True)
        self.protocol_button = QPushButton("Thresholds…")
        self.protocol_button.clicked.connect(self.edit_protocol)
        form.addRow("Thresholds", self._pair(self.protocol_label, self.protocol_button))
        column.addLayout(form)

        column.addWidget(self._title("PUPS"))
        self.status = QLabel()
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # Tighter than the app's 8 px and a point smaller: five numeric columns
        # are what the tab is for, and they must fit without scrolling sideways.
        self.table.setStyleSheet(
            "QHeaderView { font-size: 11px; }"
            "QHeaderView::section { padding: 6px 2px; }"
            "QTableView::item { padding: 5px 2px; }"
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self._on_row_clicked)
        column.addWidget(self.table)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        column.addWidget(self.summary)
        column.addStretch(1)
        self._result = None
        self._nests: list[float] = []
        self._read_only = ""

    @staticmethod
    def _title(text: str) -> QLabel:
        # The Inspector's section-title style (its _section helper).
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    @staticmethod
    def _pair(value: QLabel, button: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(value, 1)
        row.addWidget(button)
        return row

    def show_setup(
        self,
        setup: RetrievalSetup,
        protocol: RetrievalProtocol,
        *,
        writable: bool,
        protocol_writable: bool = True,
        error: str = "",
        n_pups: int = 0,
    ) -> None:
        """*writable* is the video's setup, *protocol_writable* the cohort's protocol.

        Two files, read and refused separately: a broken protocol must not
        lock the nest, nor a broken setup the protocol.
        """
        self._nests = [p.from_s for p in setup.nests]
        self.nest_list.clear()
        for i, period in enumerate(setup.nests, start=1):
            self.nest_list.addItem(QListWidgetItem(f"Nest {i} · from {_clock(period.from_s)}"))
        self.nest_button.setText("Redraw from here" if setup.nests else "Draw nest")
        self.trial_label.setText(
            "not set" if setup.trial_start_s is None else format_seconds(setup.trial_start_s)
        )
        self.scale_label.setText(
            "not set" if not setup.px_per_mm else f"{setup.px_per_mm:.2f} px/mm"
        )
        self.protocol_label.setText(
            f"cutoff {_clock(protocol.cutoff_s)} · nose '{protocol.nose_keypoint}' · litter {n_pups}"
        )
        for button in (self.nest_button, self.remove_button, self.trial_button, self.scale_button):
            button.setEnabled(writable)
        self.protocol_button.setEnabled(protocol_writable)
        self._read_only = f"Read-only: {error}" if error else ""
        self.status.setText(self._read_only)

    def show_result(self, result: SessionRetrieval | None, reason: str = "") -> None:
        self._result = result
        self.table.setRowCount(0)
        # Why the buttons are grey stays on screen under any reason not to score.
        self.status.setText("\n".join(t for t in (self._read_only, reason) if t))
        if result is None:
            self.summary.setText("")
            return
        for name, p in result.pups.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            cells = (
                name,
                _num(p.contact_latency_s),
                _num(p.retrieval_latency_s),
                str(p.carries),
                str(p.drops),
            )
            for col, text in enumerate(cells):
                self.table.setItem(row, col, QTableWidgetItem(text))
        litter = result.litter
        rank = litter["retrieval_order"]
        order = ", ".join(n.removeprefix("pup_") for n in sorted(rank, key=rank.get))
        parts = [f"{litter['pups_retrieved']} of {litter['pups_scattered']} retrieved"]
        if np.isfinite(litter["time_all_retrieved_s"]):
            parts.append(f"all by {litter['time_all_retrieved_s']:.1f} s")
        if order:
            parts.append(f"order {order}")
        if np.isfinite(result.dam_in_nest_frac):
            parts.append(f"dam in nest {result.dam_in_nest_frac:.0%}")
        self.summary.setText(" · ".join(parts))

    def highlight_carried(self, names: set[str]) -> None:
        """Bold the pups being carried on the frame under the playhead."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            font = item.font()
            font.setBold(item.text() in names)
            item.setFont(font)

    def _on_remove(self) -> None:
        if self.nest_list.currentRow() >= 0:
            self.remove_nest.emit(self.nest_list.currentRow())

    def _on_nest_clicked(self, item) -> None:
        self.jump_to.emit(float(self._nests[self.nest_list.row(item)]))

    def _on_row_clicked(self, row: int, _col: int) -> None:
        """To the pup's retrieval, or its first contact if it was never retrieved."""
        if self._result is None:
            return
        p = list(self._result.pups.values())[row]
        target = (
            p.retrieval_latency_s if np.isfinite(p.retrieval_latency_s) else p.contact_latency_s
        )
        if np.isfinite(target):
            # Latencies count from the trial start; the playhead from frame 0.
            self.jump_to.emit(float(target) + self._result.t0 / (self._result.fps or 1.0))
