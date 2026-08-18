"""One plugin, as a row in the Plugins window.

The card owns a single plugin's identity, description, state pill, actions and
inline failure text. It holds **no** registry, installer or ``PluginManager``
reference -- the dialog wires its signals and drives its state. That is
deliberate: it keeps the card constructible from a plain dict, so every state
in the table below can be pinned by a test without a network or a subprocess.

Two things here carry meaning rather than decoration:

* **The package name and version are monospace**, and visually distinct from
  the human-readable display name. They are what you type into ``pip`` and
  what a bug report needs; the display name is not.
* **Failures render on the row, never in a modal**, and pip's own output is
  shown verbatim. pip's message is usually the actual answer ("no matching
  distribution for zmq>=26"), and summarising it destroys the useful part.

Colours live in :mod:`glider.gui.styles.colors`; the state pill carries none at
all, only an ``objectName`` and a ``state`` property for ``desktop.qss`` to
select on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import data_font

__all__ = ["PluginCard"]

#: The six states from the spec's error table, and the pill text for each.
PILL_TEXT: dict[str, str] = {
    "enabled": "Enabled",
    "disabled": "Disabled",
    "available": "Available",
    "installing": "Installing",
    "incompatible": "Not compatible",
    "failed": "Install failed",
}

#: Button labels per state, in the order they appear on the row. The order is
#: part of the contract -- ``buttons()`` is indexed by it.
STATE_ACTIONS: dict[str, tuple[str, ...]] = {
    "enabled": ("Disable", "Reload"),
    "disabled": ("Enable",),
    "available": ("Install",),
    "installing": ("Cancel",),
    "incompatible": ("Install",),
    "failed": ("Retry",),
}

#: Which signal a given label fires. ``Cancel`` fires nothing: an in-flight pip
#: run is not interruptible mid-resolve, so the button exists to say so and is
#: shown disabled rather than lying about what it can do.
_SIGNAL_FOR_LABEL: dict[str, str] = {
    "Install": "install_requested",
    "Retry": "install_requested",
    "Enable": "enable_requested",
    "Disable": "disable_requested",
    "Reload": "reload_requested",
}

#: Labels that are shown but cannot be pressed, keyed by the state that owns
#: them. Incompatible plugins keep a visible Install button so the reason
#: underneath it has something to point at.
_DISABLED_IN_STATE: dict[str, frozenset[str]] = {
    "incompatible": frozenset({"Install"}),
    "installing": frozenset({"Cancel"}),
}

#: The message colour per state. A token, never a literal.
_MESSAGE_COLOR: dict[str, str] = {
    "installing": colors.ACCENT,
    "incompatible": colors.STATE_WARN,
    "failed": colors.STATE_ERR,
}


def _restyle(widget: QWidget) -> None:
    """Make Qt re-evaluate *widget* after a dynamic property changed.

    Qt resolves property selectors at polish time, not when the property is
    set, so a ``[state="failed"]`` rule applied later never takes effect
    without this.
    """
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


class PluginCard(QFrame):
    """A single plugin row.

    Args:
        entry: A catalogue entry. ``name`` is required; ``display_name``,
            ``pypi``, ``version``, ``description``, ``author`` and ``provides``
            are used when present.
        state: One of the keys of :data:`PILL_TEXT`.
        message: Inline status text -- the version-gate refusal, or pip's exit
            code. Hidden when empty.
        output: Verbatim pip output. Hidden when empty.
    """

    install_requested = pyqtSignal(str)
    enable_requested = pyqtSignal(str)
    disable_requested = pyqtSignal(str)
    reload_requested = pyqtSignal(str)

    def __init__(
        self,
        entry: Mapping[str, Any],
        *,
        state: str = "available",
        message: str = "",
        output: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("PluginCard")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._entry = dict(entry)
        self.plugin_name: str = str(self._entry.get("name", ""))
        self.state: str = state

        self._buttons: list[QPushButton] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(6)

        outer.addLayout(self._build_header())
        outer.addWidget(self._build_description())
        outer.addWidget(self._build_meta())
        outer.addWidget(self._build_progress())
        outer.addWidget(self._build_message())
        outer.addWidget(self._build_output())

        self.set_state(state, message=message, output=output)

    # ---------------------------------------------------------------- build

    def _build_header(self) -> QHBoxLayout:
        """Identity line: display name, then the pip-facing name and version."""
        header = QHBoxLayout()
        header.setSpacing(10)

        self._name_label = QLabel(str(self._entry.get("display_name") or self.plugin_name), self)
        self._name_label.setStyleSheet(
            f"color: {colors.TEXT_PRIMARY}; font-size: 15px; font-weight: 600;"
        )
        header.addWidget(self._name_label)

        package = str(self._entry.get("pypi") or self.plugin_name)
        self._package_label = QLabel(package, self)
        self._package_label.setFont(data_font(12))
        self._package_label.setStyleSheet(f"color: {colors.TEXT_MUTED};")
        self._package_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self._package_label)

        self._version_label = QLabel(str(self._entry.get("version", "")), self)
        self._version_label.setFont(data_font(12))
        self._version_label.setStyleSheet(f"color: {colors.TEXT_TERTIARY};")
        self._version_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._version_label.setVisible(bool(self._version_label.text()))
        header.addWidget(self._version_label)

        # The pill deliberately carries no stylesheet of its own: desktop.qss
        # owns every state colour via QLabel#pluginStatePill[state="..."].
        self._pill = QLabel(self)
        self._pill.setObjectName("pluginStatePill")
        self._pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self._pill)

        header.addStretch(1)

        self._actions = QHBoxLayout()
        self._actions.setSpacing(8)
        header.addLayout(self._actions)
        return header

    def _build_description(self) -> QLabel:
        self._description = QLabel(str(self._entry.get("description", "")), self)
        self._description.setWordWrap(True)
        self._description.setStyleSheet(f"color: {colors.TEXT_TERTIARY}; font-size: 13px;")
        self._description.setVisible(bool(self._description.text()))
        return self._description

    def _build_meta(self) -> QLabel:
        """Author and what the plugin registers, on one muted line."""
        parts: list[str] = []
        author = str(self._entry.get("author", "")).strip()
        if author:
            parts.append(author)
        provides = self._entry.get("provides") or []
        if provides:
            parts.append("provides " + ", ".join(str(p) for p in provides))

        self._meta = QLabel(" · ".join(parts), self)
        self._meta.setStyleSheet(f"color: {colors.TEXT_MUTED}; font-size: 12px;")
        self._meta.setVisible(bool(parts))
        return self._meta

    def _build_progress(self) -> QProgressBar:
        self._progress = QProgressBar(self)
        self._progress.setObjectName("pluginCardProgress")
        # Indeterminate: pip reports no total, so a percentage would be a lie.
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        return self._progress

    def _build_message(self) -> QLabel:
        self._message = QLabel("", self)
        self._message.setObjectName("pluginCardMessage")
        self._message.setWordWrap(True)
        self._message.setVisible(False)
        return self._message

    def _build_output(self) -> QPlainTextEdit:
        self._output = QPlainTextEdit(self)
        self._output.setObjectName("pluginCardOutput")
        self._output.setReadOnly(True)
        self._output.setFont(data_font(11))
        self._output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._output.setMaximumHeight(96)
        self._output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._output.setVisible(False)
        return self._output

    # ----------------------------------------------------------------- state

    def set_state(self, state: str, *, message: str = "", output: str | None = None) -> None:
        """Move the card to *state*, rebuilding its actions.

        Passing ``output=None`` leaves whatever pip has already written in
        place; pass ``""`` to clear it.
        """
        self.state = state
        self._pill.setText(PILL_TEXT.get(state, state))
        self._pill.setProperty("state", state)
        _restyle(self._pill)

        self._rebuild_actions(state)
        self.set_message(message, state=state)
        if output is not None:
            self.set_output(output)
        self._progress.setVisible(state == "installing")

    def _rebuild_actions(self, state: str) -> None:
        while (item := self._actions.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._buttons = []

        disabled = _DISABLED_IN_STATE.get(state, frozenset())
        for label in STATE_ACTIONS.get(state, ()):
            button = QPushButton(label, self)
            if label in ("Install", "Retry"):
                button.setProperty("role", "primary")
            if label in disabled:
                button.setEnabled(False)
            else:
                signal_name = _SIGNAL_FOR_LABEL.get(label)
                if signal_name is not None:
                    signal = getattr(self, signal_name)
                    button.clicked.connect(
                        lambda _checked=False, s=signal: s.emit(self.plugin_name)
                    )
            _restyle(button)
            self._actions.addWidget(button)
            self._buttons.append(button)

    def set_message(self, message: str, *, state: str | None = None) -> None:
        """Show inline status text, or hide the line when *message* is empty."""
        colour = _MESSAGE_COLOR.get(state or self.state, colors.TEXT_TERTIARY)
        self._message.setStyleSheet(f"color: {colour}; font-size: 12px;")
        self._message.setText(message)
        self._message.setVisible(bool(message))

    def set_output(self, output: str) -> None:
        """Replace the pip transcript. Hidden while empty."""
        self._output.setPlainText(output)
        self._output.setVisible(bool(output))

    def append_output(self, chunk: str) -> None:
        """Append a line of pip output as it arrives, and keep the tail in view."""
        if not chunk:
            return
        self._output.appendPlainText(chunk.rstrip("\n"))
        self._output.setVisible(True)
        scrollbar = self._output.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    # ------------------------------------------------------------- accessors

    def buttons(self) -> list[QPushButton]:
        """The row's action buttons, in the order the spec's table gives them."""
        return list(self._buttons)

    def identity_text(self) -> str:
        """Display name, package name and version, as one string."""
        return " ".join(
            part
            for part in (
                self._name_label.text(),
                self._package_label.text(),
                self._version_label.text(),
            )
            if part
        )

    def message_text(self) -> str:
        return self._message.text()

    def output_text(self) -> str:
        return self._output.toPlainText()
