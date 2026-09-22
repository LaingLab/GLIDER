"""The sessions panel: Resolve's media pool, one row per animal.

A cohort is the unit of analysis, so every loaded session stays visible,
grouped by the treatment recorded in ``glider_project.json``. Each row says
which evidence the session has -- video, poses, hardware -- before you click
it, and carries its ethogram in miniature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from glider.gui.review.timeline import behavior_qcolor
from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import ElidedLabel, set_text_role

__all__ = ["PoolEntry", "SessionPool", "ethogram_strip"]

_INDEX = Qt.ItemDataRole.UserRole
_GROUP = Qt.ItemDataRole.UserRole + 1


@dataclass(frozen=True)
class PoolEntry:
    name: str
    group: str
    duration_s: float
    has_video: bool
    has_poses: bool
    has_hardware: bool
    strip: QImage | None = None


def ethogram_strip(labels: list[str], order: list[str], width: int = 160) -> QImage:
    """The ethogram at ``width`` pixels, one majority behaviour per pixel."""
    image = QImage(width, 1, QImage.Format.Format_RGB32)
    image.fill(QColor(colors.BORDER))
    if not labels:
        return image
    slot = {name: i + 1 for i, name in enumerate(order)}
    codes = np.array([slot.get(v, 0) for v in labels], dtype=np.int64)
    palette = [QColor(colors.BORDER)] + [behavior_qcolor(name, order) for name in order]
    edges = np.linspace(0, len(codes), width + 1).astype(int)
    for x in range(width):
        lo = int(edges[x])
        hi = max(lo + 1, int(edges[x + 1]))
        code = int(np.bincount(codes[lo:hi], minlength=len(palette)).argmax())
        image.setPixelColor(x, 0, palette[code])
    return image


def _badge(text: str, present: bool) -> QLabel:
    label = QLabel(text)
    label.setObjectName("PoolBadge")
    label.setProperty("present", "true" if present else "false")
    return label


class _Row(QWidget):
    def __init__(self, entry: PoolEntry, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 5, 8, 5)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        # Name and length, then the badges on a row of their own: all three on
        # one line clipped the badges to "VI", "PO", "H" at the default width.
        name = ElidedLabel(entry.name)
        font = name.font()
        font.setBold(True)
        name.setFont(font)
        name.setToolTip(entry.name)
        seconds = int(round(entry.duration_s))
        length = QLabel(f"{seconds // 60}:{seconds % 60:02d}")
        set_text_role(length, "muted")
        badges = QHBoxLayout()
        badges.setSpacing(3)
        for text, present in (
            ("VID", entry.has_video),
            ("POSE", entry.has_poses),
            ("HW", entry.has_hardware),
        ):
            badges.addWidget(_badge(text, present))
        badges.addStretch(1)
        grid.addWidget(name, 0, 0)
        grid.addWidget(length, 0, 1)
        grid.addLayout(badges, 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        if entry.strip is not None:
            strip = QLabel()
            strip.setFixedHeight(5)
            strip.setScaledContents(True)
            strip.setPixmap(QPixmap.fromImage(entry.strip))
            grid.addWidget(strip, 2, 0, 1, 2)


class SessionPool(QFrame):
    """Every loaded session, grouped by treatment; clicking one shows it."""

    session_picked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SessionPool")
        self.setMinimumWidth(200)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(12, 8, 12, 4)
        title = QLabel("SESSIONS")
        title.setObjectName("SectionTitle")
        self._count = QLabel("0")
        set_text_role(self._count, "muted")
        header.addWidget(title)
        header.addWidget(self._count)
        header.addStretch(1)
        column.addLayout(header)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter sessions…")
        self.filter.setClearButtonEnabled(True)
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(8, 0, 8, 6)
        filter_row.addWidget(self.filter)
        column.addLayout(filter_row)
        self.tree = QTreeWidget()
        self.tree.setObjectName("SessionTree")
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(0)
        column.addWidget(self.tree, 1)
        self._items: list[QTreeWidgetItem] = []
        self._names: list[str] = []
        self._groups: list[QTreeWidgetItem] = []
        self.filter.textChanged.connect(self._apply_filter)
        self.tree.currentItemChanged.connect(self._on_current)

    def set_entries(self, entries: list[PoolEntry]) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        self._items, self._names, self._groups = [], [], []
        # PoolEntry holds a QImage, which is unhashable, so group names are
        # collected, not entries.
        grouped = bool(entries) and list(dict.fromkeys(e.group for e in entries)) != [""]
        headers: dict[str, QTreeWidgetItem] = {}
        for index, entry in enumerate(entries):
            parent = self.tree.invisibleRootItem()
            if grouped:
                name = entry.group or "Ungrouped"
                if name not in headers:
                    header = QTreeWidgetItem(self.tree.invisibleRootItem())
                    header.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    header.setData(0, _GROUP, name)
                    headers[name] = header
                    self._groups.append(header)
                parent = headers[name]
            item = QTreeWidgetItem(parent)
            item.setData(0, _INDEX, index)
            item.setSizeHint(0, QSize(0, 58 if entry.strip is not None else 40))
            self.tree.setItemWidget(item, 0, _Row(entry))
            self._items.append(item)
            self._names.append(entry.name)
        for header in self._groups:
            header.setText(0, f"{header.data(0, _GROUP).upper()}  ·  {header.childCount()}")
        self.tree.expandAll()
        self._count.setText(str(len(entries)))
        self.tree.blockSignals(False)
        self._apply_filter(self.filter.text())

    def count(self) -> int:
        return len(self._items)

    def select(self, index: int) -> None:
        if 0 <= index < len(self._items):
            self.tree.setCurrentItem(self._items[index])

    def current(self) -> int:
        item = self.tree.currentItem()
        index = None if item is None else item.data(0, _INDEX)
        return -1 if index is None else int(index)

    def _on_current(self, item, _previous) -> None:
        index = None if item is None else item.data(0, _INDEX)
        if index is not None:
            self.session_picked.emit(int(index))

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for item, name in zip(self._items, self._names, strict=True):
            item.setHidden(bool(needle) and needle not in name.lower())
        for header in self._groups:
            children = [header.child(i) for i in range(header.childCount())]
            header.setHidden(all(child.isHidden() for child in children))
