"""The Experiment tab's Plugins section: a browser that has to load first.

Every other section on that rail is a widget that exists the moment it is
constructed. This one is not: :class:`~glider.gui.dialogs.plugin_manager_dialog.PluginManagerDialog`
cannot be built until the plugin catalogue has been fetched over the network,
which takes seconds on a good connection and can fail outright on a rig with no
route to the internet -- which describes a lot of lab machines.

So the rail gets this instead: a host that is instantly constructible, says
what it is doing, and swaps the real browser in underneath itself when the
fetch lands. Three states, and the last two both matter:

* **Loading** -- the fetch is in flight. A blank section here would be
  indistinguishable from a broken one, and the fetch is slow enough that
  somebody will see this.
* **Ready** -- the browser, embedded.
* **Failed** -- the reason, on the section's own surface. Deliberately not a
  modal: the user is already looking at this page, and a warning box thrown
  over the tab they just opened tells them less than a line of text in it.

**The fetch is started once and never retried on its own.** A section that
re-fetched every time you clicked back onto it would hammer the index from a
tab people flick through; :meth:`PluginsSection.reload` is there for when the
user asks.

**No colour is set from Python here**: every part carries an ``objectName`` and
``desktop.qss`` owns the appearance.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.plugins.plugin_manager import PluginManager

logger = logging.getLogger(__name__)

__all__ = ["PluginsSection"]


class PluginsSection(QWidget):
    """Hosts the plugin browser, and stands in for it while it loads.

    Args:
        plugin_manager_fn: Called at fetch time rather than held, because the
            manager does not exist until plugin discovery has run and this
            section can be built before that.
        parent: Standard Qt parent.
    """

    def __init__(
        self,
        plugin_manager_fn: Callable[[], PluginManager | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("pluginsSection")
        self._plugin_manager_fn = plugin_manager_fn
        self._browser: QWidget | None = None
        # A strong reference to the in-flight fetch. asyncio holds only a weak
        # one, so a task dropped here is collected mid-request and the section
        # sits on "Loading" for ever with nothing in the log to say why.
        self._task: asyncio.Task[Any] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._stack = QStackedWidget(self)
        outer.addWidget(self._stack)

        self._status = QLabel("")
        self._status.setObjectName("pluginsSectionStatus")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)

        self._retry = QPushButton("Try Again")
        self._retry.setObjectName("pluginsSectionRetry")
        self._retry.setCursor(Qt.CursorShape.PointingHandCursor)
        self._retry.clicked.connect(self.reload)
        self._retry.hide()

        message_page = QWidget()
        message_layout = QVBoxLayout(message_page)
        message_layout.addStretch(1)
        message_layout.addWidget(self._status)
        message_layout.addSpacing(12)
        message_layout.addWidget(self._retry, 0, Qt.AlignmentFlag.AlignHCenter)
        message_layout.addStretch(2)
        self._stack.addWidget(message_page)

        self.reload()

    # ----------------------------------------------------------------- states

    def _show_message(self, text: str, *, retry: bool) -> None:
        self._status.setText(text)
        self._retry.setVisible(retry)
        self._stack.setCurrentIndex(0)

    def reload(self) -> None:
        """Fetch the catalogue and build the browser. Idempotent while in flight."""
        if self._task is not None and not self._task.done():
            return
        if self._browser is not None:
            return

        manager = self._plugin_manager_fn()
        if manager is None:
            # Discovery has not run. Saying so beats an empty catalogue, which
            # reads as "no plugins exist" rather than "not asked yet".
            self._show_message(
                "The plugin system has not started yet. Finish loading the "
                "session and try again.",
                retry=True,
            )
            return

        self._show_message("Loading the plugin catalogue…", retry=False)
        try:
            self._task = asyncio.ensure_future(self._load(manager))
        except RuntimeError:
            # No running loop. Reachable in a plain unit test and on the sync
            # fallback path in __main__, where qasync never started.
            logger.debug("No event loop for the plugin catalogue fetch", exc_info=True)
            self._show_message(
                "The plugin catalogue could not be loaded: no event loop.", retry=True
            )

    async def _load(self, manager: PluginManager) -> None:
        from glider.gui.dialogs.plugin_manager_dialog import PluginManagerDialog

        try:
            browser = await PluginManagerDialog.build_for(parent=self, plugin_manager=manager)
        except Exception as exc:
            # build_for raises rather than reporting, precisely so this can put
            # the reason on the section instead of in a modal over the tab.
            logger.exception("Plugins section could not load the catalogue")
            self._show_message(f"The plugin catalogue could not be read:\n\n{exc}", retry=True)
            return

        self._browser = browser.embed()
        self._stack.addWidget(self._browser)
        self._stack.setCurrentWidget(self._browser)

    # ----------------------------------------------------------------- probes

    def browser(self) -> QWidget | None:
        """The embedded plugin browser, or ``None`` while it is loading."""
        return self._browser

    def status_text(self) -> str:
        """Whatever the placeholder currently says. For tests."""
        return self._status.text()
