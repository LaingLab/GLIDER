"""The window: what it lists, how it filters, and where failures land."""

from __future__ import annotations

import types

from PyQt6.QtWidgets import QMainWindow

from glider.gui.dialogs.plugin_manager_dialog import PluginManagerDialog
from glider.plugins.registry import ResolvedIndex

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


def test_glider_core_exposes_the_plugin_manager():
    """The GUI must not reach through ``_plugin_manager``."""
    from glider.core.glider_core import GliderCore

    assert isinstance(GliderCore.plugin_manager, property)
