"""The Experiment tab: lazy sections, and the dialogs that learned to embed.

The lazy-build and reset behaviour is the part worth pinning down. A section
built against the previous session keeps editing an object nothing reads any
more -- edits that appear to work and are silently discarded, which is the
worst failure mode this page could have.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QWidget

from glider.gui.main_window import PAGE_EXPERIMENT
from glider.gui.panels.experiment_page import SECTION_KEYS, ExperimentPage


@pytest.fixture
def counting_builders():
    """Three builders that record how often each was called."""
    calls: dict[str, int] = dict.fromkeys(SECTION_KEYS, 0)

    def make(key: str):
        def _build() -> QWidget:
            calls[key] += 1
            return QLabel(f"section:{key}")

        return _build

    return calls, {key: make(key) for key in SECTION_KEYS}


def test_no_section_is_built_until_it_is_visited(qtbot, counting_builders):
    """Constructing the zone editor grabs a camera frame. Doing that at window
    startup, for a tab that may never be opened, costs every launch a camera
    round-trip."""
    calls, builders = counting_builders
    page = ExperimentPage(builders)
    qtbot.addWidget(page)

    assert calls == dict.fromkeys(SECTION_KEYS, 0)


def test_a_section_is_built_once_and_then_reused(qtbot, counting_builders):
    calls, builders = counting_builders
    page = ExperimentPage(builders)
    qtbot.addWidget(page)

    page.show_section("zones")
    page.show_section("metadata")
    page.show_section("zones")

    assert calls["zones"] == 1
    assert calls["metadata"] == 1
    assert calls["vocabulary"] == 0


def test_reset_drops_built_sections_so_they_rebuild(qtbot, counting_builders):
    calls, builders = counting_builders
    page = ExperimentPage(builders)
    qtbot.addWidget(page)
    page.show_section("metadata")

    page.reset()
    assert page.built_sections() == {}

    page.refresh()
    assert calls["metadata"] == 2


def test_a_builder_that_raises_costs_only_its_own_section(qtbot, counting_builders):
    """A rig whose camera has gone away should lose the Zones section, not the
    Experiment tab and everything on it."""
    calls, builders = counting_builders

    def _boom() -> QWidget:
        raise RuntimeError("camera went away")

    builders["zones"] = _boom
    page = ExperimentPage(builders)
    qtbot.addWidget(page)

    page.show_section("zones")
    assert page.built_sections() == {}

    page.show_section("metadata")
    assert "metadata" in page.built_sections()


def test_the_rail_switches_sections(qtbot, counting_builders):
    calls, builders = counting_builders
    page = ExperimentPage(builders)
    qtbot.addWidget(page)

    page.rail_buttons()["vocabulary"].click()

    assert page.current_section() == "vocabulary"
    assert calls["vocabulary"] == 1


# ------------------------------------------------------- embedded dialogs


def test_experiment_dialog_splits_into_two_pages(qtbot, main_window_factory):
    """Metadata and Mice are separate pages backed by one editor -- so a
    subject added on one and a protocol typed on the other reach the session
    through a single object that knows about both."""
    from glider.gui.dialogs.experiment_dialog import ExperimentDialog

    window = main_window_factory(desktop_mode=True)
    dialog = ExperimentDialog(session=window._core.session)
    qtbot.addWidget(dialog)

    metadata, mice = dialog.detach_sections()
    qtbot.addWidget(metadata)
    qtbot.addWidget(mice)

    assert dialog.windowFlags() & Qt.WindowType.Widget == Qt.WindowType.Widget
    assert not dialog._button_box.isVisible()
    # Each group box actually moved into its own page, rather than being
    # copied or left behind in the dialog's own scroll area.
    assert dialog._info_group.window() is metadata.window()
    assert dialog._subjects_group.window() is mice.window()
    assert metadata is not mice


def test_zone_editor_embeds_without_ok_or_cancel(qtbot):
    """Cancel is the sharper of the two to remove: the editor mutates the
    caller's ZoneConfiguration as you draw, so it never rolled anything back."""
    from glider.gui.dialogs.zone_dialog import ZoneDialog
    from glider.vision.zones import ZoneConfiguration

    editor = ZoneDialog(camera_manager=None, zone_config=ZoneConfiguration()).embed()
    qtbot.addWidget(editor)

    assert not editor._ok_btn.isVisible()
    assert not editor._cancel_dialog_btn.isVisible()


def test_lab_setup_embedded_keeps_done_and_survives_accept(qtbot):
    """Done stays because it is the only thing that writes the file, and
    ``accept()`` must not hide the page out from under a successful save."""
    from glider.gui.dialogs.lab_setup_dialog import LabSetupDialog

    form = LabSetupDialog().embed()
    qtbot.addWidget(form)
    form.show()

    assert not form.skip_button.isVisible()
    assert form.done_button.isVisible()

    form.accept()
    assert form.isVisible()


def test_the_window_builds_the_metadata_section_on_first_visit(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()

    window._tab_bar.buttons()["experiment"].click()

    assert window._stack.currentIndex() == PAGE_EXPERIMENT
    assert "metadata" in window._experiment_page.built_sections()


def test_metadata_and_mice_share_one_editor(qtbot, main_window_factory):
    """Two rail entries, one ExperimentDialog underneath."""
    window = main_window_factory(desktop_mode=True)
    window.show()
    window._show_experiment_tab()
    window._experiment_page.show_section("mice")

    built = window._experiment_page.built_sections()
    assert {"metadata", "mice"} <= set(built)
    assert built["metadata"] is not built["mice"]
    assert window._experiment_details is not None


def test_new_experiment_drops_the_shared_editor_too(qtbot, main_window_factory):
    """Not just the pages: a surviving dialog would keep writing subjects into
    the experiment that was just replaced."""
    window = main_window_factory(desktop_mode=True)
    window.show()
    window._show_experiment_tab()
    assert window._experiment_details is not None

    window._on_new()

    assert window._experiment_details is None
    assert window._experiment_detail_widgets is None


def test_every_rail_entry_has_its_own_glyph(qtbot):
    """Four distinguishable icons, painted rather than shipped -- a blank or a
    duplicate here is invisible until someone looks at the rail."""
    page = ExperimentPage({k: (lambda key=k: QLabel(key)) for k in SECTION_KEYS})
    qtbot.addWidget(page)

    seen = []
    for key, button in page.rail_buttons().items():
        assert not button.icon().isNull(), key
        image = button.icon().pixmap(16, 16).toImage()
        assert any(
            image.pixelColor(x, y).alpha() > 0
            for x in range(image.width())
            for y in range(image.height())
        ), f"{key} glyph painted nothing"
        seen.append(image)

    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert seen[i] != seen[j], "two rail glyphs are identical"


def test_new_experiment_resets_the_tab(qtbot, main_window_factory):
    """The sections hold the session and ZoneConfiguration that New replaces."""
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()
    window._show_experiment_tab()
    assert window._experiment_page.built_sections()

    window._on_new()

    assert window._experiment_page.built_sections() == {}


# ------------------------------------------------------------ plugins section


def test_plugins_section_says_so_before_discovery_has_run(qtbot):
    """An empty catalogue reads as 'no plugins exist' rather than 'not asked
    yet' -- which is why this says something instead of showing a blank list."""
    from glider.gui.panels.plugins_section import PluginsSection

    section = PluginsSection(lambda: None)
    qtbot.addWidget(section)

    assert section.browser() is None
    assert "has not started yet" in section.status_text()


def test_plugins_section_reports_a_catalogue_it_cannot_read(qtbot, monkeypatch):
    """build_for raises rather than reporting, precisely so the reason lands on
    the section instead of in a modal thrown over the tab."""
    import asyncio

    from glider.gui.dialogs.plugin_manager_dialog import PluginManagerDialog
    from glider.gui.panels.plugins_section import PluginsSection

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("no route to host")

    monkeypatch.setattr(PluginManagerDialog, "build_for", _boom)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        section = PluginsSection(lambda: object())
        qtbot.addWidget(section)
        loop.run_until_complete(section._task)
    finally:
        loop.close()

    assert section.browser() is None
    assert "no route to host" in section.status_text()


def test_the_rail_offers_plugins(qtbot, counting_builders):
    calls, builders = counting_builders
    page = ExperimentPage(builders)
    qtbot.addWidget(page)

    assert "plugins" in page.rail_buttons()
    page.rail_buttons()["plugins"].click()
    assert calls["plugins"] == 1


def test_every_rail_entry_is_the_full_width_of_the_rail(qtbot, counting_builders):
    """The reported miss: a QToolButton sizes itself to its label, so each entry
    was a different width and most of the rail looked clickable but was inert.
    You aim at "Mice" and hit nothing, because the button stops with the text.
    """
    _calls, builders = counting_builders
    page = ExperimentPage(builders)
    qtbot.addWidget(page)
    page.resize(600, 400)
    page.show()
    qtbot.waitExposed(page)

    widths = {key: b.width() for key, b in page.rail_buttons().items()}
    assert len(set(widths.values())) == 1, f"inconsistent hit targets: {widths}"

    # And that one width is the rail, not the widest label.
    rail = next(iter(page.rail_buttons().values())).parentWidget()
    margins = rail.layout().contentsMargins()
    expected = rail.width() - margins.left() - margins.right()
    assert next(iter(widths.values())) == expected
