"""The Pi Run page's device status cards: two columns in a vertical scroll area.

Each card is a name, a type, and a state badge over a link word ("Ready",
"Disconnected"...). The badge repaints on a timer while a run is live
(``set_live``), and once more when the run ends. The two colours come from
``device_state_display`` / ``link_status_color`` at paint time, so they are the
one inline style left here; everything else is styled by the ``deviceCard``
property and the object names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QScroller,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from glider.core.config import get_config
from glider.gui.device_status import device_state_display, link_status_color, link_status_text
from glider.gui.styles import colors, radius

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

_COLUMNS = 2
_EMPTY_TEXT = "No devices yet. Add a board on the Dashboard › Hardware tab."


class DeviceCardGrid(QWidget):
    def __init__(self, core: GliderCore, parent=None):
        super().__init__(parent)
        self._core = core
        self._cards: dict[str, QWidget] = {}

        self._empty = QLabel(_EMPTY_TEXT)
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setProperty("textRole", "muted")
        self._grid = QGridLayout()
        self._grid.setSpacing(8)
        for col in range(_COLUMNS):
            self._grid.setColumnStretch(col, 1)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(self._empty)
        column.addLayout(self._grid)
        column.addStretch()

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(body)
        # Let the page behind show through, as the old runner list did.
        self._scroll.viewport().setAutoFillBackground(False)
        body.setAutoFillBackground(False)
        QScroller.grabGesture(
            self._scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self._poll = QTimer(self)
        self._poll.setInterval(get_config().timing.device_refresh_interval_ms)
        self._poll.timeout.connect(self.update_states)

        self.refresh_devices()

    # --- public API ---

    def cards(self) -> dict[str, QWidget]:
        return self._cards

    def refresh_devices(self) -> None:
        """Rebuild the cards from the hardware manager's devices."""
        for card in self._cards.values():
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        devices = self._core.hardware_manager.devices
        self._empty.setVisible(not devices)
        for i, (device_id, device) in enumerate(devices.items()):
            card = self._create_card(device_id, device)
            self._grid.addWidget(card, *divmod(i, _COLUMNS))
            self._cards[device_id] = card

    def update_states(self) -> None:
        """Repaint each card's state badge and link word."""
        for device_id, card in self._cards.items():
            device = self._core.hardware_manager.get_device(device_id)
            if device is not None:
                self._paint(card, device)

    def set_live(self, live: bool) -> None:
        """Poll the states while a run is live; repaint once when it ends."""
        if live:
            self._poll.start()
        else:
            self._poll.stop()
            self.update_states()

    # --- internals ---

    def _create_card(self, device_id: str, device) -> QWidget:
        card = QWidget()
        card.setProperty("deviceCard", True)
        card.setFixedHeight(80)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        info = QVBoxLayout()
        info.setSpacing(2)
        for text, name in (
            (device_id, "deviceCardName"),
            (getattr(device, "device_type", "Unknown"), "deviceCardType"),
        ):
            label = QLabel(text)
            label.setObjectName(name)
            label.setToolTip(text)
            # Clip a long name rather than let it widen the column.
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            info.addWidget(label)
        layout.addLayout(info, 1)

        # 80 for both: the link words ("Disconnected", "Reconnecting…") are
        # longer than the "Ready"/"---" a narrower box was sized for, and a
        # status that paints as "Disconnec" is not one anybody can act on.
        status = QWidget()
        status.setFixedSize(80, 50)
        status_layout = QVBoxLayout(status)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(2)
        card._state_label = QLabel()
        card._ready_label = QLabel()
        for label in (card._state_label, card._ready_label):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_layout.addWidget(label)
        layout.addWidget(status)

        self._paint(card, device)
        return card

    @staticmethod
    def _paint(card: QWidget, device) -> None:
        # link_state, not _initialized: "has this been set up" is a question a
        # peripheral that walked out of range never stops answering yes to, and
        # this card repaints on a timer, so it asserted a green Ready while the
        # status strip showed a red dot for the same device.
        link = getattr(device, "link_state", None)
        text, color, font_size = device_state_display(device)
        card._state_label.setText(text)
        card._state_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: {colors.TEXT_PRIMARY};
                font-size: {font_size};
                font-weight: bold;
                border-radius: {radius.MEDIUM}px;
                padding: 4px 8px;
                border: none;
                line-height: 1.2;
            }}
        """)
        card._ready_label.setText(link_status_text(link))
        card._ready_label.setStyleSheet(
            f"font-size: 10px; color: {link_status_color(link)};"
            " background: transparent; border: none;"
        )
