"""Browse the curated plugin catalogue, and install from it.

The window is **non-modal**. A pip resolve takes minutes, and a modal dialog
would hold the whole application -- including a rig that is mid-experiment --
still for the duration. That is the single most important decision in this file.

Two more are worth stating outright:

* **The constructor takes data, never services.** It is handed a resolved
  ``ResolvedIndex`` and a plain ``{name: version}`` map, so every row state in
  the spec's table can be pinned by a test with no network, no subprocess and
  no ``PluginManager``. :meth:`PluginManagerDialog.open_for` is the *only*
  place a network fetch happens.
* **The footer is not decoration.** Spec section 9 makes the curated index the
  entire security model here: installing runs arbitrary code with GLIDER's
  privileges on a machine driving lab hardware, so the user has to be able to
  see which index they are trusting and how old it is. Do not quietly drop it
  to save a line of vertical space.

Failures render on the row that caused them -- see
:mod:`glider.gui.widgets.plugin_card` -- never in a message box, and never on
a neighbouring plugin.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from glider._version import __version__ as GLIDER_VERSION
from glider.gui.widgets.plugin_card import PluginCard
from glider.plugins.installer import incompatibility_message, install, is_compatible
from glider.plugins.registry import PluginRegistry, ResolvedIndex

if TYPE_CHECKING:
    from glider.plugins.plugin_manager import PluginManager

logger = logging.getLogger(__name__)

__all__ = ["PluginManagerDialog"]

#: The filter buttons, in the order they appear. ``all`` is the default.
FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("installed", "Installed"),
    ("available", "Available"),
)

#: Row states that mean "this package is on disk".
_INSTALLED_STATES = frozenset({"enabled", "disabled"})

#: How each index source reads in the footer. Each phrase contains its own key
#: word, so the footer names the source whichever branch of `resolve()` won.
_SOURCE_TEXT: dict[str, str] = {
    "network": "downloaded over the network",
    "cache": "local cache",
    "bundled": "bundled with GLIDER",
}

#: The entry-point groups ``PluginManager`` scans. Used to map a distribution
#: back to the plugin names the manager knows it by -- the catalogue keys on
#: the *distribution* name, the manager keys on the *entry point* name, and
#: they are routinely different ("glider-harp" vs "harp").
_PLUGIN_GROUPS = ("glider.driver", "glider.device", "glider.node")

_RESTART_CAVEAT = (
    "Installing runs code from PyPI with GLIDER's privileges. A newly installed "
    "plugin loads immediately; upgrading one already in use needs a restart."
)


def _entry_point_names(pypi: str) -> list[str]:
    """The GLIDER entry-point names *pypi* registers, or ``[]`` if not installed."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution(pypi)
    except PackageNotFoundError:
        return []
    except Exception as exc:  # a corrupt dist-info must not take the window down
        logger.warning("Could not read entry points for %s: %s", pypi, exc)
        return []
    return [ep.name for ep in dist.entry_points if ep.group in _PLUGIN_GROUPS]


def installed_state(
    index: ResolvedIndex, plugin_manager: PluginManager | None
) -> tuple[dict[str, str], set[str]]:
    """Work out what is on disk, and what is on disk but switched off.

    "Installed" is answered by ``importlib.metadata``, not by the plugin
    manager: the manager only knows about packages whose entry points loaded,
    so a package that installed correctly but failed to import would otherwise
    read as "available" and invite a pointless second pip run.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    known = plugin_manager.plugins if plugin_manager is not None else {}
    installed: dict[str, str] = {}
    disabled: set[str] = set()

    for entry in index.plugins:
        name = str(entry.get("name", ""))
        pypi = str(entry.get("pypi") or name)
        if not pypi:
            continue
        try:
            installed[name] = dist_version(pypi)
        except PackageNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not read the installed version of %s: %s", pypi, exc)
            continue

        seen = [known[ep] for ep in _entry_point_names(pypi) if ep in known]
        # Only call it disabled when the manager knows the plugin and every
        # part of it is off. Silence from the manager is not a "no".
        if seen and not any(info.enabled for info in seen):
            disabled.add(name)

    return installed, disabled


class PluginManagerDialog(QDialog):
    """The Plugins window.

    Args:
        index: The resolved catalogue, including where it came from.
        installed: ``{catalogue name: installed version}`` for packages on disk.
        disabled: Names among *installed* the plugin manager has switched off.
        plugin_manager: Used for enable/disable/reload and post-install
            rediscovery. Optional so tests can build the window without one.
        glider_version: The running version, for the compatibility gate.
    """

    def __init__(
        self,
        index: ResolvedIndex,
        installed: Mapping[str, str],
        *,
        disabled: Iterable[str] = (),
        plugin_manager: PluginManager | None = None,
        glider_version: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("PluginManagerDialog")
        self.setWindowTitle("Plugins")
        # Non-modal on purpose: an install takes minutes and must not hold a
        # running experiment still.
        self.setModal(False)
        self.resize(760, 560)

        self._index = index
        self._installed: dict[str, str] = dict(installed)
        self._disabled: set[str] = set(disabled)
        self._plugin_manager = plugin_manager
        self._glider_version = glider_version or GLIDER_VERSION

        self._entries: dict[str, dict[str, Any]] = {}
        self._cards: list[PluginCard] = []
        self._cards_by_name: dict[str, PluginCard] = {}
        # Strong references to in-flight installs; asyncio only holds weak ones,
        # so a task dropped here would be collected mid-pip.
        self._tasks: set[asyncio.Task[Any]] = set()

        self._filter = "all"
        self._search = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(12)
        outer.addLayout(self._build_controls())
        outer.addWidget(self._build_list(), 1)
        outer.addWidget(self._build_footer())

        self._populate()

    # ---------------------------------------------------------------- build

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._search_box = QLineEdit(self)
        self._search_box.setObjectName("pluginSearch")
        self._search_box.setPlaceholderText("Search plugins…")
        self._search_box.setClearButtonEnabled(True)
        self._search_box.textChanged.connect(self.set_search)
        row.addWidget(self._search_box, 1)

        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)
        self._filter_buttons: dict[str, QPushButton] = {}
        for key, label in FILTERS:
            button = QPushButton(label, self)
            button.setObjectName("pluginFilter")
            button.setCheckable(True)
            button.setChecked(key == self._filter)
            button.clicked.connect(lambda _checked=False, k=key: self.set_filter(k))
            self._filter_group.addButton(button)
            self._filter_buttons[key] = button
            row.addWidget(button)

        return row

    def _build_list(self) -> QScrollArea:
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("pluginScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget(self._scroll)
        self._card_layout = QVBoxLayout(container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(8)
        self._card_layout.addStretch(1)

        self._empty = QLabel("No plugins match.", container)
        self._empty.setObjectName("pluginEmpty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setVisible(False)
        self._card_layout.insertWidget(self._card_layout.count() - 1, self._empty)

        self._scroll.setWidget(container)
        return self._scroll

    def _build_footer(self) -> QLabel:
        self._footer = QLabel(self)
        self._footer.setObjectName("pluginFooter")
        self._footer.setWordWrap(True)
        self._footer.setText(self._compose_footer())
        return self._footer

    def _compose_footer(self) -> str:
        source = _SOURCE_TEXT.get(self._index.source, self._index.source)
        provenance = f"Catalogue: {source}"
        if self._index.updated:
            provenance += f", updated {self._index.updated}"
        return f"{provenance}. {_RESTART_CAVEAT}"

    def _populate(self) -> None:
        for entry in self._index.plugins:
            name = str(entry.get("name", ""))
            if not name:
                continue
            state, message = self._state_for(entry)
            shown = dict(entry)
            # Show what is on disk rather than what the catalogue advertises;
            # they diverge the moment an upgrade is published.
            if name in self._installed:
                shown["version"] = self._installed[name]

            card = PluginCard(shown, state=state, message=message, parent=self)
            card.install_requested.connect(self._on_install_requested)
            card.enable_requested.connect(self._on_enable_requested)
            card.disable_requested.connect(self._on_disable_requested)
            card.reload_requested.connect(self._on_reload_requested)

            self._entries[name] = dict(entry)
            self._cards.append(card)
            self._cards_by_name[name] = card
            self._card_layout.insertWidget(self._card_layout.count() - 2, card)

        self._apply_filters()

    def _state_for(self, entry: Mapping[str, Any]) -> tuple[str, str]:
        name = str(entry.get("name", ""))
        if name in self._installed:
            return ("disabled" if name in self._disabled else "enabled", "")
        if not is_compatible(entry, self._glider_version):
            return "incompatible", incompatibility_message(entry, self._glider_version)
        return "available", ""

    # -------------------------------------------------------------- filtering

    def set_filter(self, name: str) -> None:
        """Show only ``all``, ``installed`` or ``available`` rows."""
        self._filter = name
        button = self._filter_buttons.get(name)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._apply_filters()

    def set_search(self, text: str) -> None:
        """Filter rows by a substring of the name, description or author."""
        self._search = text.strip().lower()
        if self._search_box.text() != text:
            self._search_box.setText(text)
        self._apply_filters()

    def _apply_filters(self) -> None:
        any_visible = False
        for card in self._cards:
            visible = self._matches(card)
            card.setVisible(visible)
            any_visible = any_visible or visible
        self._empty.setVisible(not any_visible)

    def _matches(self, card: PluginCard) -> bool:
        is_installed = card.state in _INSTALLED_STATES
        if self._filter == "installed" and not is_installed:
            return False
        if self._filter == "available" and is_installed:
            return False
        if not self._search:
            return True

        entry = self._entries.get(card.plugin_name, {})
        haystack = " ".join(
            str(entry.get(field, ""))
            for field in ("name", "display_name", "pypi", "description", "author")
        ).lower()
        return self._search in haystack

    # ---------------------------------------------------------------- actions

    def _on_install_requested(self, name: str) -> None:
        entry = self._entries.get(name)
        card = self._cards_by_name.get(name)
        if entry is None or card is None:
            return
        # Clear the previous transcript: a Retry that leaves the old failure
        # visible under a fresh progress bar reads as the new run failing too.
        card.set_state("installing", message=f"Installing {entry.get('pypi', name)}…", output="")
        self._spawn(self._install(entry, card))

    async def _install(self, entry: Mapping[str, Any], card: PluginCard) -> None:
        name = str(entry.get("name", ""))
        try:
            result = await install(entry, self._glider_version, on_output=card.append_output)
        except Exception as exc:
            logger.exception("Install of %s raised", name)
            self.show_install_failure(name, f"Install failed: {exc}", card.output_text())
            return

        if not result.ok:
            self.show_install_failure(name, result.message, result.output)
            return

        self._installed[name] = str(entry.get("version", ""))
        self._disabled.discard(name)
        card.set_state("enabled", message=result.message)
        await self._rediscover()
        self._refresh_row(name)
        self._apply_filters()

    async def _rediscover(self) -> None:
        """Let a freshly installed plugin register without a restart."""
        if self._plugin_manager is None:
            return
        try:
            await self._plugin_manager.discover_plugins()
            await self._plugin_manager.load_plugins()
        except Exception as exc:
            logger.warning("Rediscovery after install failed: %s", exc)

    def _refresh_row(self, name: str) -> None:
        """Recompute one row from disk. Deliberately not a full rebuild --
        another row may be mid-install with live pip output on it."""
        card = self._cards_by_name.get(name)
        entry = self._entries.get(name)
        if card is None or entry is None:
            return
        installed, disabled = installed_state(
            ResolvedIndex(plugins=[dict(entry)]), self._plugin_manager
        )
        if name in installed:
            self._installed[name] = installed[name]
        self._disabled = (self._disabled - {name}) | (disabled & {name})
        state, message = self._state_for(entry)
        card.set_state(state, message=message)

    def _on_enable_requested(self, name: str) -> None:
        self._toggle(name, enable=True)

    def _on_disable_requested(self, name: str) -> None:
        self._toggle(name, enable=False)

    def _toggle(self, name: str, *, enable: bool) -> None:
        card = self._cards_by_name.get(name)
        if card is None:
            return
        if self._plugin_manager is None:
            card.set_message("No plugin manager is running; nothing to switch.")
            return

        entry = self._entries.get(name, {})
        targets = _entry_point_names(str(entry.get("pypi") or name))
        for plugin in targets:
            if enable:
                self._plugin_manager.enable_plugin(plugin)
            else:
                self._plugin_manager.disable_plugin(plugin)

        if enable:
            self._disabled.discard(name)
            self._spawn(self._load_enabled())
        else:
            self._disabled.add(name)
        card.set_state("enabled" if enable else "disabled")
        self._apply_filters()

    async def _load_enabled(self) -> None:
        if self._plugin_manager is None:
            return
        try:
            await self._plugin_manager.load_plugins()
        except Exception as exc:
            logger.warning("Loading newly enabled plugins failed: %s", exc)

    def _on_reload_requested(self, name: str) -> None:
        card = self._cards_by_name.get(name)
        if card is None:
            return
        if self._plugin_manager is None:
            card.set_message("No plugin manager is running; nothing to reload.")
            return
        entry = self._entries.get(name, {})
        targets = _entry_point_names(str(entry.get("pypi") or name))
        self._spawn(self._reload(name, targets))

    async def _reload(self, name: str, targets: list[str]) -> None:
        card = self._cards_by_name.get(name)
        if card is None or self._plugin_manager is None:
            return
        failures = []
        for plugin in targets:
            try:
                if not await self._plugin_manager.reload_plugin(plugin):
                    failures.append(plugin)
            except Exception as exc:
                logger.warning("Reloading %s failed: %s", plugin, exc)
                failures.append(plugin)
        if failures:
            card.set_message(f"Could not reload: {', '.join(failures)}. A restart will.")
        else:
            card.set_message("Reloaded.")

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------- accessors

    def cards(self) -> list[PluginCard]:
        """Every row, in catalogue order."""
        return list(self._cards)

    def visible_cards(self) -> list[PluginCard]:
        """The rows the current filter and search leave showing.

        ``isHidden`` rather than ``isVisible``: the latter is false for every
        child of a window that has not been shown, which would make this empty
        under an offscreen test run.
        """
        return [card for card in self._cards if not card.isHidden()]

    def footer_text(self) -> str:
        return self._footer.text()

    def show_install_failure(self, name: str, message: str, output: str) -> None:
        """Put a failure on the row that caused it -- and on no other row.

        pip's own text is shown verbatim: it is usually the actual answer, and
        summarising it destroys the useful part.
        """
        card = self._cards_by_name.get(name)
        if card is None:
            logger.warning("Install failure for unknown plugin %s: %s", name, message)
            return
        card.set_state("failed", message=message, output=output)
        self._apply_filters()

    # ------------------------------------------------------------ entry point

    @classmethod
    async def open_for(
        cls, parent: QWidget | None, plugin_manager: PluginManager | None
    ) -> PluginManagerDialog:
        """Resolve the catalogue, then show the window.

        The only place in this module that touches the network.
        """
        from glider.core.config import get_config

        cache_dir = Path(get_config().paths.user_config_dir)
        index = await PluginRegistry(cache_dir=cache_dir).resolve()
        installed, disabled = installed_state(index, plugin_manager)

        dialog = cls(
            index=index,
            installed=installed,
            disabled=disabled,
            plugin_manager=plugin_manager,
            parent=parent,
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog
