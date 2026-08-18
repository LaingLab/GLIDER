"""The window: what it lists, how it filters, and where failures land."""

from __future__ import annotations

import asyncio
import importlib.metadata
import types

from PyQt6.QtWidgets import QMainWindow

from glider.gui.dialogs import plugin_manager_dialog as dialog_module
from glider.gui.dialogs.plugin_manager_dialog import PluginManagerDialog
from glider.plugins.installer import InstallResult
from glider.plugins.plugin_manager import PluginInfo
from glider.plugins.registry import PluginRegistry, ResolvedIndex

INDEX = ResolvedIndex(
    plugins=[
        {
            "name": "glider-harp",
            "display_name": "Harp Devices",
            "version": "0.1.0",
            "pypi": "glider-harp",
            "description": "Harp instruments.",
            "author": "Laing Lab",
            "glider_requires": ">=1.0,<2.0",
        },
        {
            "name": "glider-bpod",
            "display_name": "Bpod",
            "version": "0.2.0",
            "pypi": "glider-bpod",
            "description": "Bpod state machine.",
            "author": "Laing Lab",
            "glider_requires": ">=1.0,<2.0",
        },
    ],
    updated="2026-08-18",
    source="cache",
)


def test_every_catalogue_entry_gets_a_row(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    assert len(dialog.cards()) == 2


def test_an_installed_plugin_reads_as_enabled(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={"glider-harp": "0.1.0"})
    qtbot.addWidget(dialog)

    by_name = {c.plugin_name: c for c in dialog.cards()}
    assert by_name["glider-harp"].state == "enabled"
    assert by_name["glider-bpod"].state == "available"


def test_the_installed_filter_hides_the_rest(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={"glider-harp": "0.1.0"})
    qtbot.addWidget(dialog)

    dialog.set_filter("installed")

    assert [c.plugin_name for c in dialog.visible_cards()] == ["glider-harp"]


def test_search_matches_description_as_well_as_name(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    dialog.set_search("state machine")

    assert [c.plugin_name for c in dialog.visible_cards()] == ["glider-bpod"]


def test_the_footer_names_the_index_source_and_date(qtbot):
    """Spec section 9 makes this the whole security model, so it is not optional
    furniture."""
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    footer = dialog.footer_text()
    assert "cache" in footer.lower()
    assert "2026-08-18" in footer


def test_a_failure_lands_on_its_own_row_only(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    dialog.show_install_failure("glider-bpod", "pip exited with code 1.", "ERROR: no match")

    by_name = {c.plugin_name: c for c in dialog.cards()}
    assert by_name["glider-bpod"].state == "failed"
    assert by_name["glider-harp"].state == "available"
    assert by_name["glider-harp"].message_text() == ""


# --------------------------------------------------------------------- extras


def test_the_dialog_is_not_modal(qtbot):
    """A pip run takes minutes; a modal window would freeze a running rig."""
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    assert dialog.isModal() is False


def test_a_disabled_plugin_reads_as_disabled(qtbot):
    dialog = PluginManagerDialog(
        index=INDEX, installed={"glider-harp": "0.1.0"}, disabled={"glider-harp"}
    )
    qtbot.addWidget(dialog)

    by_name = {c.plugin_name: c for c in dialog.cards()}
    assert by_name["glider-harp"].state == "disabled"


def test_an_installed_row_shows_the_installed_version_not_the_catalogue_one(qtbot):
    """The catalogue advertises 0.1.0; the row must not claim that is what is on
    disk when 0.0.3 is."""
    dialog = PluginManagerDialog(index=INDEX, installed={"glider-harp": "0.0.3"})
    qtbot.addWidget(dialog)

    by_name = {c.plugin_name: c for c in dialog.cards()}
    assert "0.0.3" in by_name["glider-harp"].identity_text()


def test_the_version_gate_names_both_versions(qtbot):
    index = ResolvedIndex(
        plugins=[
            {
                "name": "glider-future",
                "version": "9.0.0",
                "pypi": "glider-future",
                "description": "Needs a GLIDER that does not exist yet.",
                "glider_requires": ">=9.0",
            }
        ],
        updated="2026-08-18",
        source="bundled",
    )
    dialog = PluginManagerDialog(index=index, installed={}, glider_version="1.0.0")
    qtbot.addWidget(dialog)

    card = dialog.cards()[0]
    assert card.state == "incompatible"
    assert ">=9.0" in card.message_text()
    assert "1.0.0" in card.message_text()


def test_the_footer_carries_the_restart_caveat(qtbot):
    """A fresh install loads live; upgrading an already-imported plugin does not."""
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    assert "restart" in dialog.footer_text().lower()


def test_the_available_filter_hides_what_is_installed(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={"glider-harp": "0.1.0"})
    qtbot.addWidget(dialog)

    dialog.set_filter("available")

    assert [c.plugin_name for c in dialog.visible_cards()] == ["glider-bpod"]


def test_clearing_the_search_brings_every_row_back(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    dialog.set_search("bpod")
    dialog.set_search("")

    assert len(dialog.visible_cards()) == 2


# ------------------------------------------------------- through the real path
#
# Everything above enters through the constructor with a hand-built `installed`
# map, which is exactly why `_install`, `_rediscover`, `_refresh_row`, `_reload`
# and `open_for` could all be wrong at once and stay green. These drive the
# buttons and the entry point instead, and fake only the two things a unit test
# genuinely cannot have: pip, and the contents of site-packages.


class _FakeManager:
    """A ``PluginManager`` stand-in that reports what a load actually did.

    The point of the fake is the distinction the window kept losing: a plugin
    can be *enabled* (nobody switched it off), *not loaded*, and carrying an
    error, all at once. That is what a package that pip-installs but raises on
    import looks like.
    """

    def __init__(self, infos: list[PluginInfo]) -> None:
        self._infos = {info.name: info for info in infos}
        self.discovered = 0
        self.loads = 0

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        return dict(self._infos)

    async def discover_plugins(self) -> list[PluginInfo]:
        self.discovered += 1
        return list(self._infos.values())

    async def load_plugins(self) -> dict[str, bool]:
        self.loads += 1
        return {name: info.loaded for name, info in self._infos.items()}


async def _drain(dialog: PluginManagerDialog) -> None:
    """Await whatever the last button press spawned."""
    tasks = list(dialog._tasks)
    if tasks:
        await asyncio.gather(*tasks)


def _fake_pip(monkeypatch, result: InstallResult, *, output: str = "Collecting glider-harp"):
    async def fake_install(entry, glider_version, on_output=None):
        if on_output:
            on_output(output)
        return result

    monkeypatch.setattr(dialog_module, "install", fake_install)


def _fake_site_packages(monkeypatch, *, versions: dict[str, str], entry_points: list[str]):
    """Pin what ``importlib.metadata`` would say about the installed world."""

    def fake_version(name: str) -> str:
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    monkeypatch.setattr(dialog_module, "_entry_point_names", lambda pypi: list(entry_points))


def _card(dialog: PluginManagerDialog, name: str):
    return {c.plugin_name: c for c in dialog.cards()}[name]


async def test_a_plugin_that_installs_but_will_not_import_does_not_read_as_enabled(
    qtbot, monkeypatch
):
    """pip succeeded, the import raised. ``enabled`` defaults to True and no
    failure path clears it, so reading the pill off ``enabled`` alone put a green
    "Enabled" on a plugin that does not work."""
    broken = PluginInfo(name="harp", enabled=True, loaded=False, error="Module not found: serial")
    manager = _FakeManager([broken])
    _fake_pip(monkeypatch, InstallResult(ok=True, message="Installed glider-harp.", output="done"))
    _fake_site_packages(monkeypatch, versions={"glider-harp": "0.1.0"}, entry_points=["harp"])

    dialog = PluginManagerDialog(index=INDEX, installed={}, plugin_manager=manager)
    qtbot.addWidget(dialog)
    card = _card(dialog, "glider-harp")

    card.buttons()[0].click()  # Install
    await _drain(dialog)

    assert card.state != "enabled"
    assert manager.loads == 1


async def test_the_load_error_is_shown_on_the_row_that_owns_it(qtbot, monkeypatch):
    """`load_plugin` writes the diagnosis onto `PluginInfo.error` and the window
    threw it away, so it died in `logger.error` where nobody was looking."""
    broken = PluginInfo(name="harp", enabled=True, loaded=False, error="Module not found: serial")
    manager = _FakeManager([broken])
    _fake_pip(monkeypatch, InstallResult(ok=True, message="Installed glider-harp.", output="done"))
    _fake_site_packages(monkeypatch, versions={"glider-harp": "0.1.0"}, entry_points=["harp"])

    dialog = PluginManagerDialog(index=INDEX, installed={}, plugin_manager=manager)
    qtbot.addWidget(dialog)
    card = _card(dialog, "glider-harp")

    card.buttons()[0].click()
    await _drain(dialog)

    assert "serial" in card.message_text()
    assert _card(dialog, "glider-bpod").message_text() == ""


async def test_a_plugin_that_loads_cleanly_still_reads_as_enabled(qtbot, monkeypatch):
    """The other half of the same rule: reading `loaded` must not make every
    freshly installed plugin look broken."""
    good = PluginInfo(name="harp", enabled=True, loaded=True)
    manager = _FakeManager([good])
    _fake_pip(monkeypatch, InstallResult(ok=True, message="Installed glider-harp.", output="done"))
    _fake_site_packages(monkeypatch, versions={"glider-harp": "0.1.0"}, entry_points=["harp"])

    dialog = PluginManagerDialog(index=INDEX, installed={}, plugin_manager=manager)
    qtbot.addWidget(dialog)
    card = _card(dialog, "glider-harp")

    card.buttons()[0].click()
    await _drain(dialog)

    assert card.state == "enabled"


async def test_the_row_shows_the_version_that_landed_not_the_advertised_one(qtbot, monkeypatch):
    """The catalogue says 0.1.0; pip resolved 0.2.5. The row kept claiming the
    catalogue's number, which is the one thing on the row nobody can check."""
    manager = _FakeManager([PluginInfo(name="harp", loaded=True)])
    _fake_pip(monkeypatch, InstallResult(ok=True, message="Installed glider-harp.", output="done"))
    _fake_site_packages(monkeypatch, versions={"glider-harp": "0.2.5"}, entry_points=["harp"])

    dialog = PluginManagerDialog(index=INDEX, installed={}, plugin_manager=manager)
    qtbot.addWidget(dialog)
    card = _card(dialog, "glider-harp")

    card.buttons()[0].click()
    await _drain(dialog)

    assert "0.2.5" in card.identity_text()
    assert "0.1.0" not in card.identity_text()


async def test_a_finished_install_is_not_filtered_out_from_under_the_user(qtbot, monkeypatch):
    """With **Available** selected, re-filtering on completion hid the row at the
    exact moment it had something to say -- taking the pip transcript with it."""
    manager = _FakeManager([PluginInfo(name="harp", loaded=True)])
    _fake_pip(monkeypatch, InstallResult(ok=True, message="Installed glider-harp.", output="done"))
    _fake_site_packages(monkeypatch, versions={"glider-harp": "0.1.0"}, entry_points=["harp"])

    dialog = PluginManagerDialog(index=INDEX, installed={}, plugin_manager=manager)
    qtbot.addWidget(dialog)
    dialog.set_filter("available")
    card = _card(dialog, "glider-harp")

    card.buttons()[0].click()
    await _drain(dialog)

    assert card in dialog.visible_cards()
    assert "Collecting glider-harp" in card.output_text()


async def test_a_failed_install_is_not_filtered_out_either(qtbot, monkeypatch):
    _fake_pip(monkeypatch, InstallResult(ok=False, message="pip exited with code 1.", output="err"))

    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)
    dialog.set_filter("installed")
    card = _card(dialog, "glider-harp")
    assert card not in dialog.visible_cards()

    card.buttons()[0].click()
    await _drain(dialog)

    assert card in dialog.visible_cards()
    assert card.state == "failed"


async def test_touching_the_filter_again_applies_it_in_full(qtbot, monkeypatch):
    """The exemption is for the moment of completion only. Once the user works
    the controls again the filter means what it says."""
    manager = _FakeManager([PluginInfo(name="harp", loaded=True)])
    _fake_pip(monkeypatch, InstallResult(ok=True, message="Installed glider-harp.", output="done"))
    _fake_site_packages(monkeypatch, versions={"glider-harp": "0.1.0"}, entry_points=["harp"])

    dialog = PluginManagerDialog(index=INDEX, installed={}, plugin_manager=manager)
    qtbot.addWidget(dialog)
    dialog.set_filter("available")
    card = _card(dialog, "glider-harp")
    card.buttons()[0].click()
    await _drain(dialog)

    dialog.set_filter("available")

    assert card not in dialog.visible_cards()


async def test_reload_says_when_there_was_nothing_to_reload(qtbot):
    """`glider-bpod` is not installed here, so `_entry_point_names` returns [],
    the loop over targets never runs, and the row used to report "Reloaded." """
    manager = _FakeManager([])
    dialog = PluginManagerDialog(
        index=INDEX, installed={"glider-bpod": "0.2.0"}, plugin_manager=manager
    )
    qtbot.addWidget(dialog)
    card = _card(dialog, "glider-bpod")
    assert [b.text() for b in card.buttons()] == ["Disable", "Reload"]

    card.buttons()[1].click()  # Reload
    await _drain(dialog)

    assert card.message_text() != "Reloaded."
    assert "entry point" in card.message_text().lower()


# ------------------------------------------------------------------- open_for


async def test_open_for_renders_an_unreadable_entry_instead_of_taking_the_window_down(
    qtbot, monkeypatch
):
    """`"1.0"` where `">=1.0"` was meant is the natural authoring mistake, and
    the index arrives over the network. It must render as a bad row, not as an
    exception thrown out of the constructor into a discarded task."""
    bad = ResolvedIndex(
        plugins=[
            {
                "name": "glider-oops",
                "pypi": "glider-oops",
                "version": "1.0.0",
                "glider_requires": "1.0",
            }
        ],
        updated="2026-08-18",
        source="network",
    )

    async def fake_resolve(self):
        return bad

    monkeypatch.setattr(PluginRegistry, "resolve", fake_resolve)

    dialog = await PluginManagerDialog.open_for(parent=None, plugin_manager=None)
    assert dialog is not None
    qtbot.addWidget(dialog)

    card = dialog.cards()[0]
    assert card.state == "incompatible"
    assert "unreadable" in card.message_text().lower()
    assert card.buttons()[0].isEnabled() is False


async def test_open_for_tells_the_user_when_the_catalogue_cannot_be_read(qtbot, monkeypatch):
    """A malformed *bundled* index is sanctioned to raise, and nothing caught it.
    `ensure_future` then turned it into a GC-time warning and a window that never
    appeared."""

    async def boom(self):
        raise ValueError("bundled plugin index is malformed: index.json")

    monkeypatch.setattr(PluginRegistry, "resolve", boom)
    shown: list[tuple] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox, "warning", lambda *args, **kwargs: shown.append(args)
    )

    result = await PluginManagerDialog.open_for(parent=None, plugin_manager=None)

    assert result is None
    assert len(shown) == 1
    assert "malformed" in " ".join(str(part) for part in shown[0])


# ---------------------------------------------------------------- menu wiring


def _menu_only_window():
    """A MainWindow with only ``_setup_menu`` run — the pattern established in
    ``test_main_window_tools_menu.py``, where the rationale is written out."""
    from glider.gui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(win)
    win._view_manager = types.SimpleNamespace(is_runner_mode=False)
    win._setup_menu()
    return win


def _tools_menu(win):
    for action in win.menuBar().actions():
        if action.text().replace("&", "") == "Tools":
            return action.menu()
    return None


def test_the_tools_menu_offers_plugins(qtbot):
    win = _menu_only_window()
    try:
        labels = [a.text().replace("&", "") for a in _tools_menu(win).actions()]
        assert any("Plugins" in label for label in labels)
    finally:
        win.deleteLater()


def test_a_cold_start_says_so_instead_of_showing_an_empty_catalogue(qtbot, monkeypatch):
    """``plugin_manager`` is None until discovery has run. Opening a window that
    lists nothing would read as a broken index rather than an uninitialised one."""
    from glider.gui import main_window as main_window_module

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, text, *a, **k: shown.append((title, text)),
    )

    win = _menu_only_window()
    try:
        win._core = types.SimpleNamespace(plugin_manager=None)
        win._on_open_plugins()

        assert len(shown) == 1
        assert "plugin" in shown[0][1].lower()
    finally:
        win.deleteLater()


def _window_with_a_plugin_manager():
    win = _menu_only_window()
    win._core = types.SimpleNamespace(plugin_manager=_FakeManager([]))
    win._plugins_dialog = None
    win._plugins_task = None
    return win


async def test_the_menu_does_not_open_a_second_window_or_fire_a_second_fetch(qtbot, monkeypatch):
    """Three clicks used to mean three windows and three network fetches -- the
    first two while the first fetch was still in flight."""
    opened: list[object] = []

    async def fake_open_for(parent, plugin_manager):
        opened.append(plugin_manager)
        dialog = PluginManagerDialog(index=INDEX, installed={}, parent=parent)
        dialog.show()
        return dialog

    monkeypatch.setattr(PluginManagerDialog, "open_for", fake_open_for)

    win = _window_with_a_plugin_manager()
    try:
        win._on_open_plugins()
        task = win._plugins_task
        win._on_open_plugins()  # while the first fetch is still in flight
        await task
        await asyncio.sleep(0)

        assert len(opened) == 1
        assert win._plugins_dialog is not None

        win._on_open_plugins()  # with the window already up
        assert len(opened) == 1
    finally:
        win.deleteLater()


async def test_a_plugins_window_that_fails_to_open_reaches_the_user(qtbot, monkeypatch):
    """`ensure_future` with no done callback is the exact hazard the dialog's own
    `_spawn` documents; the menu did it anyway, so a raise became silence."""
    from glider.gui import main_window as main_window_module

    async def boom(parent, plugin_manager):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(PluginManagerDialog, "open_for", boom)
    shown: list[tuple] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "warning", lambda *args, **kwargs: shown.append(args)
    )

    win = _window_with_a_plugin_manager()
    try:
        win._on_open_plugins()
        await asyncio.gather(win._plugins_task, return_exceptions=True)
        await asyncio.sleep(0)

        assert len(shown) == 1
        assert "index exploded" in " ".join(str(part) for part in shown[0])
    finally:
        win.deleteLater()


def test_glider_core_exposes_the_plugin_manager():
    """The GUI must not reach through ``_plugin_manager``."""
    from glider.core.glider_core import GliderCore

    assert isinstance(GliderCore.plugin_manager, property)
