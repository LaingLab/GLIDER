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
    page.show_section("details")
    page.show_section("zones")

    assert calls["zones"] == 1
    assert calls["details"] == 1
    assert calls["vocabulary"] == 0


def test_reset_drops_built_sections_so_they_rebuild(qtbot, counting_builders):
    calls, builders = counting_builders
    page = ExperimentPage(builders)
    qtbot.addWidget(page)
    page.show_section("details")

    page.reset()
    assert page.built_sections() == {}

    page.refresh()
    assert calls["details"] == 2


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

    page.show_section("details")
    assert "details" in page.built_sections()


def test_the_rail_switches_sections(qtbot, counting_builders):
    calls, builders = counting_builders
    page = ExperimentPage(builders)
    qtbot.addWidget(page)

    page.rail_buttons()["vocabulary"].click()

    assert page.current_section() == "vocabulary"
    assert calls["vocabulary"] == 1


# ------------------------------------------------------- embedded dialogs


def test_experiment_dialog_embeds_as_a_plain_widget(qtbot, main_window_factory):
    """``embed()`` has to do both halves: without the Widget flag the
    'embedded' form opens as a floating window over the tab meant to hold it."""
    from glider.gui.dialogs.experiment_dialog import ExperimentDialog

    window = main_window_factory(desktop_mode=True)
    panel = ExperimentDialog(session=window._core.session).embed()
    qtbot.addWidget(panel)

    assert panel.windowFlags() & Qt.WindowType.Widget == Qt.WindowType.Widget
    assert not panel._button_box.isVisible()


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


def test_the_window_builds_the_details_section_on_first_visit(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()

    window._tab_bar.buttons()["experiment"].click()

    assert window._stack.currentIndex() == PAGE_EXPERIMENT
    assert "details" in window._experiment_page.built_sections()


def test_new_experiment_resets_the_tab(qtbot, main_window_factory):
    """The sections hold the session and ZoneConfiguration that New replaces."""
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()
    window._show_experiment_tab()
    assert window._experiment_page.built_sections()

    window._on_new()

    assert window._experiment_page.built_sections() == {}
