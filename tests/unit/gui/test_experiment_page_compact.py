"""The Experiment tab on the Pi's 480 px screen: chips on top, sections full width."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QScrollArea, QWidget

from glider.gui.dialogs.zone_dialog import ZoneDialog
from glider.gui.panels.experiment_page import SECTION_KEYS, ExperimentPage
from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape


def _width_needed(section: QWidget) -> int:
    """What a section needs, counting what its own scroll areas hold.

    A section that wraps itself in a scroll area reports a tiny
    ``minimumSizeHint``, so the widgets inside are measured too; and an explicit
    ``setMinimumSize`` counts, which ``minimumSizeHint`` ignores.
    """
    inside = [s.widget() for s in section.findChildren(QScrollArea) if s.widget() is not None]
    return max(max(w.minimumSizeHint().width(), w.minimumWidth()) for w in [section, *inside])


def test_compact_page_has_chips_above_the_sections(qtbot):
    page = ExperimentPage({k: (lambda k=k: QLabel(k)) for k in SECTION_KEYS}, compact=True)
    qtbot.addWidget(page)
    assert page.findChild(QFrame, "experimentRail") is None
    chips = page.findChild(QFrame, "experimentChips")
    assert chips is not None
    assert list(page.rail_buttons()) == list(SECTION_KEYS)
    page.rail_buttons()["zones"].click()
    assert page.current_section() == "zones"


def test_compact_sections_scroll_vertically_only(qtbot):
    page = ExperimentPage({k: (lambda k=k: QLabel(k)) for k in SECTION_KEYS}, compact=True)
    qtbot.addWidget(page)
    page.show_section("mice")
    section = page.built_sections()["mice"]  # the section itself, not its wrapper
    assert isinstance(section, QLabel)
    wrapper = section.parentWidget().parentWidget()
    assert isinstance(wrapper, QScrollArea)
    assert wrapper.widgetResizable()
    assert wrapper.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_every_pi_section_fits_480(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=False)
    window.resize(480, 800)
    window.show()
    window._show_experiment_tab()
    page = window._experiment_page
    assert page.minimumSizeHint().width() <= 480  # the chips
    for key in SECTION_KEYS:
        page.show_section(key)
        QApplication.processEvents()
        built = page.built_sections().get(key)
        if built is None:  # a builder that legitimately fails offscreen (no camera)
            continue
        assert built.minimumSizeHint().width() <= 480, key
        assert _width_needed(built) <= 480, key


def test_detached_dialog_shell_stays_hidden_on_the_pi(qtbot, main_window_factory):
    """The emptied ExperimentDialog is a child of the page; shown, it covers both sections."""
    window = main_window_factory(desktop_mode=False)
    window.resize(480, 800)
    window.show()
    window._show_experiment_tab()
    window._experiment_page.show_section("metadata")
    QApplication.processEvents()
    assert window._experiment_details.isHidden()


def _one_zone() -> ZoneConfiguration:
    config = ZoneConfiguration()
    zone = Zone(
        id="a",
        name="Nest",
        shape=ZoneShape.RECTANGLE,
        vertices=[(0.1, 0.1), (0.5, 0.5)],
        color=(0, 255, 0),
    )
    config.add_zone(zone)
    return config


def test_compact_zone_editor_is_a_column_and_keeps_its_delete_label(qtbot):
    editor = ZoneDialog(camera_manager=None, zone_config=_one_zone()).embed(compact=True)
    qtbot.addWidget(editor)
    assert _width_needed(editor) <= 480
    # Fixed at 40px it is narrower than the touch button's own padding.
    assert editor._table.cellWidget(0, 3).maximumWidth() > 40


def test_desktop_zone_editor_embed_is_unchanged(qtbot):
    editor = ZoneDialog(camera_manager=None, zone_config=_one_zone()).embed()
    qtbot.addWidget(editor)
    assert editor.minimumWidth() == 1000
    assert editor._table.cellWidget(0, 3).maximumWidth() == 40
