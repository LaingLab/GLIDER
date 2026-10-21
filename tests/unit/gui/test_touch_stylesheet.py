"""Fit and finish on the Pi: what touch.qss and the runner-only touch fixes
have to get right, measured on real widgets."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QScrollArea, QScroller, QTreeView

from glider.gui.styles import colors, load_stylesheet

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "box_pi_side.glider"


def _pi(qtbot, main_window_factory):
    w = main_window_factory(desktop_mode=False)
    w.resize(480, 800)
    w.show()
    qtbot.waitUntil(w._pi_header.isVisible)
    return w


def test_the_graph_lock_is_a_44px_target_and_shows_all_its_text(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w.switch_to_builder()
    lock = w._graph_view._overlay.lock_btn
    qtbot.waitUntil(lock.isVisible)
    # The sheet lands after the button was first sized; it must re-fit.
    assert lock.height() >= 44
    assert lock.width() >= lock.sizeHint().width()


def test_the_notice_text_takes_the_colour_of_a_new_level(qtbot):
    from glider.gui.runner.notice_bar import NoticeBar

    bar = NoticeBar()
    qtbot.addWidget(bar)
    bar.setStyleSheet(load_stylesheet("touch"))
    bar.show_notice("Saved", "box_pi_side.glider", "info")
    bar.show_notice("UNSAFE STATE", "Hardware may still be active", "critical")
    text = bar._label.palette().color(QPalette.ColorRole.WindowText)
    assert text.name() == colors.ERROR


def test_the_styled_strip_still_fits_five_boards_beside_the_gear(qtbot):
    from glider.gui.shell.status_strip import StatusStrip

    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.setStyleSheet(load_stylesheet("touch"))
    strip.set_touch_mode(True)
    strip.set_experiment("x" * 60, dirty=True)
    strip.set_run_state("recording", "12:41")
    strip.set_devices([(f"board_{i}", "error", "") for i in range(5)])
    strip.ensurePolished()
    for child in strip.findChildren(type(strip.pill())):
        child.ensurePolished()
    assert strip.minimumSizeHint().width() <= 480 - 44


def test_the_pi_run_box_wraps_its_board_line(qtbot, mock_core):
    from glider.gui.dashboard.panels.run_control_panel import RunControlPanel

    touch = RunControlPanel(mock_core, touch=True)
    desktop = RunControlPanel(mock_core)
    qtbot.addWidget(touch)
    qtbot.addWidget(desktop)
    assert touch._sub.wordWrap()
    assert not desktop._sub.wordWrap()


def test_vocabulary_rows_and_remove_buttons_are_44px_on_the_pi(qtbot):
    from glider.core.vocabulary import Vocabulary
    from glider.gui.dialogs.lab_setup_dialog import LabSetupDialog

    vocabulary = Vocabulary(groups=["control", "CHR2"])
    pi = LabSetupDialog(vocabulary=vocabulary, is_touch_mode=True).embed(compact=True)
    desk = LabSetupDialog(vocabulary=vocabulary)
    qtbot.addWidget(pi)
    qtbot.addWidget(desk)
    editor = pi.editors["groups"]
    for row in range(editor.row_count()):
        assert editor.list_widget.item(row).sizeHint().height() >= 44
    button = editor.remove_button("control")
    assert button.minimumSize().width() >= 44 and button.minimumSize().height() >= 44
    assert desk.editors["groups"].remove_button("control").minimumSize().height() < 44


def _finger_scrolled(w) -> dict[str, list[bool]]:
    """Whether each scrollable viewport of the Pi's panels has a scroller."""
    page = w._experiment_page
    page.show_section("metadata")
    page.show_section("vocabulary")
    page.show_section("zones")
    panels = {
        "nodes": w._node_library_panel,
        "hardware": w._hardware_panel,
        "camera": w._camera_panel,
        "metadata": page.built_sections()["metadata"],
        "vocabulary": page.built_sections()["vocabulary"],
    }
    found = {
        key: [
            QScroller.hasScroller(area.viewport())
            for area in panel.findChildren((QScrollArea, QTreeView))
        ]
        for key, panel in panels.items()
    }
    holders = page._holders
    for key in ("metadata", "zones"):
        if holders[key] is not page.built_sections()[key]:  # compact wrapper
            found[f"{key}_wrapper"] = [QScroller.hasScroller(holders[key].viewport())]
    return found


def test_the_pi_scrolls_its_panels_by_finger_and_desktop_does_not(qtbot, main_window_factory):
    pi = _finger_scrolled(_pi(qtbot, main_window_factory))
    for key in ("nodes", "hardware", "camera", "vocabulary"):
        assert pi[key] and all(pi[key]), key
    assert pi["metadata_wrapper"] == [True]
    # Zones is drawn on by dragging: a scroller would take the drags.
    assert pi["zones_wrapper"] == [False]

    desk = _finger_scrolled(main_window_factory(desktop_mode=True))
    assert not any(any(flags) for flags in desk.values())


def test_an_opened_flow_is_fitted_clear_of_the_lock(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w.switch_to_builder()
    graph = w._graph_view
    qtbot.waitUntil(graph.isVisible)
    graph.scale(0.2, 0.2)  # somewhere the flow is not framed
    assert w._load_experiment_path(str(_EXAMPLE))
    view = graph.viewport().rect()
    lock = graph._overlay.lock_btn.geometry()
    assert graph.nodes
    for node in graph.nodes.values():
        box = graph.mapFromScene(node.sceneBoundingRect()).boundingRect()
        assert view.contains(box), node
        assert not box.intersects(lock), node


def test_the_first_dashboard_visit_fits_the_flow(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    assert w._load_experiment_path(str(_EXAMPLE))  # loaded while on Run
    calls = []
    graph = w._graph_view
    original = graph.fit_to_screen
    graph.fit_to_screen = lambda: (calls.append(1), original())
    w.switch_to_builder()
    qtbot.waitUntil(lambda: bool(calls))


def test_taps_still_reach_controls_inside_the_finger_scrolled_areas(
    qtbot, main_window_factory, monkeypatch
):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QPushButton

    w = _pi(qtbot, main_window_factory)
    assert w._load_experiment_path(str(_EXAMPLE))
    w.switch_to_builder()
    dash = w._pi_dashboard
    graph = w._graph_view
    graph.set_locked(False)

    # Nodes: a tap on a library button adds the node.
    before = len(graph.nodes)
    button = next(
        b for b in w._node_library_panel.findChildren(QPushButton) if b.property("nodeButton")
    )
    qtbot.waitUntil(button.isVisible)
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(graph.nodes) == before + 1)

    # Hardware: a tap on a row selects it.
    dash.set_current("hardware")
    tree = w._hardware_panel.findChild(QTreeView)
    qtbot.waitUntil(tree.isVisible)
    item = tree.model().index(0, 0)
    qtbot.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualRect(item).center())
    qtbot.waitUntil(lambda: tree.currentIndex() == item)

    # Camera: a tap on one of the panel's buttons lands.
    dash.set_current("camera")
    camera_button = next(
        b for b in w._camera_panel.findChildren(QPushButton) if b.isVisibleTo(w._camera_panel)
    )
    tapped = []
    camera_button.clicked.connect(lambda: tapped.append(1))
    qtbot.waitUntil(camera_button.isVisible)
    camera_button.setEnabled(True)  # some need a camera; the tap is the point
    # On screen, as a finger would find it: the scroller replays the press
    # at that spot.
    w._camera_panel.findChild(QScrollArea).ensureWidgetVisible(camera_button)
    qtbot.mouseClick(camera_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: bool(tapped))

    # Experiment: a chip tap switches the section.
    w._show_experiment_tab()
    chip = w._experiment_page._buttons["mice"]
    qtbot.waitUntil(chip.isVisible)
    qtbot.mouseClick(chip, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: w._experiment_page.current_section() == "mice")
