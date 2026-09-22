"""The review timeline: lanes on one axis, painted honestly."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor  # noqa: E402

from glider.analysis.timeline import (  # noqa: E402
    BehaviorLane,
    FrameMap,
    Lane,
    Marker,
    Segment,
    Timeline,
    build_timeline,
)
from glider.gui.review.timeline import HEADER_W, TimelineView, behavior_qcolor  # noqa: E402
from glider.gui.styles import colors  # noqa: E402

_ORDER = ["groom", "locomote"]


def _led_lane() -> Lane:
    return Lane(
        key="led1",
        label="led1",
        board_id="board0",
        segments=[Segment(0.0, 2000.0, 1.0, 1.0), Segment(2000.0, 10000.0, 0.0, 0.0)],
        markers=[Marker(5000.0, "tag-A7")],
        pin_type="DIGITAL",
    )


def _timeline(*, hardware=True, behavior=True, start_ms=0.0, lanes=None, beh=None) -> Timeline:
    """A 10 s, 30 fps session: LED on for the first 2 s, one RFID read."""
    frames = np.arange(0, 301, dtype=float)
    if lanes is None:
        lanes = [_led_lane()] if hardware else []
    if beh is None:
        labels = ["groom"] * 150 + ["locomote"] * 151
        beh = [BehaviorLane("tracking", labels, np.arange(0, 301))] if behavior else []
    return Timeline(
        lanes=lanes,
        behavior=beh,
        frame_map=FrameMap(frames, frames / 30.0 * 1000.0, "tracking"),
        flow_start_ms=0.0,
        flow_end_ms=10000.0,
        start_ms=start_ms,
        end_ms=10000.0,
    )


@pytest.fixture
def view(qtbot):
    widget = TimelineView()
    qtbot.addWidget(widget)
    widget.resize(HEADER_W + 400, 200)
    return widget


def _pixel(view, x, y) -> QColor:
    return view.grab().toImage().pixelColor(int(x), int(y))


def _band_y(view, key) -> float:
    """Inside a clip, above where its label is drawn."""
    return view.lane_rect(key).top() + 5


def test_selection_is_emitted_in_frames(view):
    view.set_timeline(_timeline())
    seen = []
    view.selection_changed.connect(lambda a, b: seen.append((a, b)))
    view.set_selection(200, 50)
    assert view.selection() == (50, 200)
    assert seen == [(50, 200)]


def test_frame_bounds_come_from_the_frame_map(view):
    view.set_timeline(_timeline())
    assert view.frame_bounds() == (0, 300)


def test_empty_timeline_has_zero_bounds(view):
    view.set_timeline(None)
    assert view.frame_bounds() == (0, 0)


def test_selection_survives_scrubbing(view):
    view.set_timeline(_timeline())
    view.set_selection(50, 200)
    view.set_frame(120)
    assert view.selection() == (50, 200)


def test_clearing_the_selection_says_so(view):
    view.set_timeline(_timeline())
    view.set_selection(50, 200)
    cleared = []
    view.selection_cleared.connect(lambda: cleared.append(True))
    view.clear_selection()
    assert view.selection() is None and cleared == [True]


def test_paints_without_a_timeline(view, qtbot):
    view.set_timeline(None)
    view.show()
    qtbot.waitExposed(view)
    view.repaint()


def test_paints_a_loaded_timeline(view, qtbot):
    view.set_timeline(_timeline())
    view.show()
    qtbot.waitExposed(view)
    view.repaint()


def test_the_lanes_start_after_the_header_column(view):
    view.set_timeline(_timeline())
    assert view.lanes_rect().left() == HEADER_W
    assert view.x_of_frame(0) == pytest.approx(HEADER_W)
    assert view.x_of_frame(300) == pytest.approx(HEADER_W + 400, abs=1.5)


def test_behavior_is_drawn_in_its_own_lane(view):
    view.set_timeline(_timeline())
    y = _band_y(view, "tracking")
    assert _pixel(view, view.x_of_frame(75), y) == behavior_qcolor("groom", _ORDER)
    assert _pixel(view, view.x_of_frame(225), y) == behavior_qcolor("locomote", _ORDER)


def test_a_hardware_lane_is_drawn_in_its_role_colour(view):
    view.set_timeline(_timeline())
    lane = view.lane_rect("led1")
    assert _pixel(view, view.x_of_frame(20), lane.bottom() - 5) == QColor(colors.LANE_OUTPUT)
    assert _pixel(view, view.x_of_frame(200), lane.bottom() - 5) == QColor(colors.CANVAS)


def test_behavior_sits_on_the_time_axis_with_everything_else(view):
    """Pre-flow device writes stretch the axis; behaviour must not stretch with it."""
    view.set_timeline(_timeline(start_ms=-30000.0))
    y = _band_y(view, "tracking")
    boundary = view.x_of_frame(150)
    assert boundary == pytest.approx(HEADER_W + 400 * 35000 / 40000, abs=0.5)
    assert _pixel(view, boundary - 2, y) == behavior_qcolor("groom", _ORDER)
    assert _pixel(view, boundary + 2, y) == behavior_qcolor("locomote", _ORDER)


def test_an_ethogram_only_timeline_paints_without_a_view(view):
    lanes = SimpleNamespace(labels=["groom"] * 150 + ["locomote"] * 151, frames=np.arange(301))
    view.set_timeline(build_timeline(None, lanes))
    assert view.frame_bounds() == (0, 300)
    y = _band_y(view, "ethogram")
    assert _pixel(view, view.x_of_frame(75), y) == behavior_qcolor("groom", _ORDER)
    assert _pixel(view, view.x_of_frame(250), y) == behavior_qcolor("locomote", _ORDER)


def test_a_lane_paints_nothing_outside_its_own_frames(view):
    frames = np.arange(0, 601, dtype=float)
    view.set_timeline(
        Timeline(
            lanes=[],
            behavior=[BehaviorLane("ethogram", ["groom"] * 201, np.arange(200, 401))],
            frame_map=FrameMap(frames, frames / 30.0 * 1000.0, "tracking"),
            flow_start_ms=0.0,
            flow_end_ms=20000.0,
            start_ms=0.0,
            end_ms=20000.0,
        )
    )
    y = _band_y(view, "ethogram")
    assert _pixel(view, view.x_of_frame(50), y) == QColor(colors.CANVAS)
    assert _pixel(view, view.x_of_frame(550), y) == QColor(colors.CANVAS)
    assert _pixel(view, view.x_of_frame(300), y) == behavior_qcolor("groom", ["groom"])


def test_many_lanes_compress_rather_than_vanish(view):
    many = [BehaviorLane(f"tracking[{i}]", ["groom"] * 301, np.arange(301)) for i in range(7)]
    view.set_timeline(_timeline(beh=many))
    view.resize(HEADER_W + 400, view.sizeHint().height())
    heights = [row.height for row in view._rows() if row.kind == "hardware"]
    assert heights and min(heights) > 0


def test_the_selection_shades_what_it_excludes_and_nothing_else(view):
    """Tint the inside and every behaviour drifts toward one hue. Shade the outside."""
    view.set_timeline(_timeline())
    y = _band_y(view, "tracking")
    inside, outside = view.x_of_frame(125), view.x_of_frame(250)
    inside_before, outside_before = _pixel(view, inside, y), _pixel(view, outside, y)
    view.set_selection(100, 199)
    assert _pixel(view, inside, y) == inside_before
    after = _pixel(view, outside, y)
    assert after != outside_before
    assert after != QColor(0, 0, 0)


def test_selecting_everything_changes_nothing_inside(view):
    view.set_timeline(_timeline())
    y = _band_y(view, "tracking")
    xs = [view.x_of_frame(f) for f in (20, 75, 280)]
    before = [_pixel(view, x, y) for x in xs]
    view.set_selection(0, 300)
    assert [_pixel(view, x, y) for x in xs] == before


def test_zoomed_in_bouts_become_clips(view):
    labels = ["groom", "locomote"] * 1500
    view.set_timeline(build_timeline(None, SimpleNamespace(labels=labels, frames=np.arange(3000))))
    lane = view._behavior_lanes()[0]
    assert view._uses_clips(lane) is False
    view.viewport.show(1000, 1060)
    view.refresh_viewport()
    assert view._uses_clips(lane) is True


def test_a_fast_pulse_train_is_drawn_at_its_duty_cycle(view):
    """Solid would draw a 100 Hz, 50 % train exactly like a lamp held on."""
    segments = []
    for i in range(1000):
        segments.append(Segment(i * 10.0, i * 10.0 + 5.0, 1.0, 1.0))
        segments.append(Segment(i * 10.0 + 5.0, (i + 1) * 10.0, 0.0, 0.0))
    train = Lane("led1", "led1", "board0", segments, [], pin_type="DIGITAL")
    view.set_timeline(_timeline(lanes=[train], behavior=False))
    colour = _pixel(view, view.x_of_frame(150), view.lane_rect("led1").bottom() - 6)
    assert colour != QColor(colors.LANE_OUTPUT)
    assert colour != QColor(colors.CANVAS)


def test_the_cached_image_follows_the_device_pixel_ratio(view):
    view.set_timeline(_timeline())
    rows = view._rows()
    assert view._static_pixmap(rows).devicePixelRatio() == pytest.approx(view.devicePixelRatioF())
    view.devicePixelRatioF = lambda: 2.0  # a move to a Retina screen, no resize
    assert view._static_pixmap(rows).devicePixelRatio() == pytest.approx(2.0)


from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QContextMenuEvent, QMouseEvent, QWheelEvent  # noqa: E402

_HANDLERS = {
    "press": (QEvent.Type.MouseButtonPress, "mousePressEvent"),
    "move": (QEvent.Type.MouseMove, "mouseMoveEvent"),
    "release": (QEvent.Type.MouseButtonRelease, "mouseReleaseEvent"),
}


def _mouse(widget, kind, x, y):
    event_type, handler = _HANDLERS[kind]
    button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
    buttons = Qt.MouseButton.NoButton if kind == "release" else Qt.MouseButton.LeftButton
    event = QMouseEvent(event_type, QPointF(x, y), button, buttons, Qt.KeyboardModifier.NoModifier)
    getattr(widget, handler)(event)


def _drag(widget, x0, x1, y):
    _mouse(widget, "press", x0, y)
    _mouse(widget, "move", (x0 + x1) / 2, y)
    _mouse(widget, "move", x1, y)
    _mouse(widget, "release", x1, y)


def _wheel(widget, x, dy, modifiers=Qt.KeyboardModifier.NoModifier):
    widget.wheelEvent(
        QWheelEvent(
            QPointF(x, 100),
            QPointF(x, 100),
            QPoint(0, 0),
            QPoint(0, dy),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
    )


def _lane_y(view):
    return view.lane_rect("tracking").center().y()


def test_the_ruler_scrubs_on_press(view):
    view.set_timeline(_timeline())
    seen = []
    view.scrubbed.connect(seen.append)
    _mouse(view, "press", view.x_of_frame(150), 10)
    assert seen and abs(seen[0] - 150) <= 1


def test_a_click_in_the_lanes_moves_the_playhead_without_selecting(view):
    view.set_timeline(_timeline())
    seen = []
    view.scrubbed.connect(seen.append)
    x = view.x_of_frame(90)
    _mouse(view, "press", x, _lane_y(view))
    assert seen == []
    _mouse(view, "release", x, _lane_y(view))
    assert seen and abs(seen[0] - 90) <= 1
    assert view.selection() is None


def test_dragging_in_the_lanes_selects_frames(view):
    view.set_timeline(_timeline())
    view.set_snap(False)
    _drag(view, view.x_of_frame(50), view.x_of_frame(200), _lane_y(view))
    start, end = view.selection()
    assert abs(start - 50) <= 1 and abs(end - 200) <= 1


def test_snap_lands_the_end_on_a_bout_edge(view):
    """groom → locomote at frame 150, so a drag ending at 153 means 'to the edge'."""
    view.set_timeline(_timeline())
    _drag(view, view.x_of_frame(20), view.x_of_frame(153), _lane_y(view))
    assert view.selection()[1] == 149


def test_without_snap_the_end_lands_where_dropped(view):
    view.set_timeline(_timeline())
    view.set_snap(False)
    _drag(view, view.x_of_frame(20), view.x_of_frame(153), _lane_y(view))
    assert abs(view.selection()[1] - 153) <= 1


def test_dragging_an_edge_trims_the_range(view):
    view.set_timeline(_timeline())
    view.set_snap(False)
    view.set_selection(50, 200)
    y = _lane_y(view)
    _mouse(view, "press", view.x_of_frame(50), y)
    _mouse(view, "move", view.x_of_frame(80), y)
    _mouse(view, "release", view.x_of_frame(80), y)
    start, end = view.selection()
    assert abs(start - 80) <= 1 and end == 200


def test_ctrl_scroll_zooms_about_the_cursor(view):
    view.set_timeline(_timeline())
    x = view.x_of_frame(150)
    before = view.viewport.t_at(x - HEADER_W, 400)
    _wheel(view, x, 120, Qt.KeyboardModifier.ControlModifier)
    assert view.viewport.span < 10000.0
    assert view.viewport.t_at(x - HEADER_W, 400) == pytest.approx(before)


def test_plain_scroll_pans(view):
    view.set_timeline(_timeline())
    view.viewport.show(0.0, 2000.0)
    view.refresh_viewport()
    _wheel(view, view.x_of_frame(30), -120)
    assert view.viewport.start > 0.0


def test_a_group_header_collapses_its_lanes(view):
    view.set_timeline(_timeline())
    group = next(r for r in view._rows() if r.key == "board:board0")
    _mouse(view, "press", 20, group.top + 5)
    assert not [r for r in view._rows() if r.kind == "hardware"]


def test_hidden_lanes_are_reported_and_dropped(view):
    view.set_timeline(_timeline())
    seen = []
    view.hidden_changed.connect(seen.append)
    view.set_hidden({"led1"})
    assert seen == [["led1"]]
    assert view.lane_rect("led1") is None


def test_right_click_asks_for_the_range_menu(view):
    view.set_timeline(_timeline())
    seen = []
    view.context_menu_requested.connect(lambda _pos, frame: seen.append(frame))
    x, y = int(view.x_of_frame(120)), int(_lane_y(view))
    view.contextMenuEvent(
        QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(x, y), QPoint(x, y))
    )
    assert seen and abs(seen[0] - 120) <= 1


def test_a_backward_drag_snaps_like_a_forward_one(view):
    view.set_timeline(_timeline())
    _drag(view, view.x_of_frame(153), view.x_of_frame(20), _lane_y(view))
    assert view.selection() == (20, 149)


def test_trimming_past_the_other_edge_keeps_that_edge(view):
    view.set_timeline(_timeline())
    view.set_snap(False)
    view.set_selection(50, 200)
    y = _lane_y(view)
    _mouse(view, "press", view.x_of_frame(50), y)
    for frame in (250, 280):
        _mouse(view, "move", view.x_of_frame(frame), y)
    _mouse(view, "release", view.x_of_frame(280), y)
    start, end = view.selection()
    assert start == 200 and abs(end - 280) <= 1


def test_a_move_without_the_button_ends_a_stale_drag(view):
    view.set_timeline(_timeline())
    y = _lane_y(view)
    _mouse(view, "press", view.x_of_frame(50), y)
    stale = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(view.x_of_frame(150), y),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mouseMoveEvent(stale)
    assert view.selection() is None
    assert view._drag is None


def test_a_right_release_does_not_end_a_left_drag(view):
    view.set_timeline(_timeline())
    view.set_snap(False)
    y = _lane_y(view)
    _mouse(view, "press", view.x_of_frame(50), y)
    _mouse(view, "move", view.x_of_frame(100), y)
    right = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(view.x_of_frame(100), y),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mouseReleaseEvent(right)
    _mouse(view, "move", view.x_of_frame(150), y)
    assert abs(view.selection()[1] - 150) <= 1


from PyQt6.QtWidgets import QTableWidget  # noqa: E402

from glider.gui.review.timeline import TimelinePanel  # noqa: E402


@pytest.fixture
def panel(qtbot):
    widget = TimelinePanel()
    qtbot.addWidget(widget)
    widget.resize(HEADER_W + 400, 300)
    widget.show()
    qtbot.waitExposed(widget)
    widget.view.set_timeline(_timeline())
    return widget


def _nav_x(nav, t):
    lane_w = nav.width() - HEADER_W
    return HEADER_W + t / 10000.0 * lane_w


def test_the_box_is_the_viewport(panel):
    panel.view.viewport.show(2000.0, 4000.0)
    panel.view.refresh_viewport()
    x0, x1 = panel.navigator.box()
    assert x0 == pytest.approx(_nav_x(panel.navigator, 2000.0))
    assert x1 == pytest.approx(_nav_x(panel.navigator, 4000.0))


def test_dragging_the_box_pans(panel):
    nav = panel.navigator
    panel.view.viewport.show(2000.0, 4000.0)
    panel.view.refresh_viewport()
    mid = _nav_x(nav, 3000.0)
    to = _nav_x(nav, 4000.0)
    _mouse(nav, "press", mid, 15)
    _mouse(nav, "move", to, 15)
    _mouse(nav, "release", to, 15)
    assert panel.view.viewport.start == pytest.approx(3000.0, abs=40.0)
    assert panel.view.viewport.span == pytest.approx(2000.0)


def test_clicking_outside_the_box_jumps_there(panel):
    panel.view.viewport.show(0.0, 2000.0)
    panel.view.refresh_viewport()
    _mouse(panel.navigator, "press", _nav_x(panel.navigator, 8000.0), 15)
    _mouse(panel.navigator, "release", _nav_x(panel.navigator, 8000.0), 15)
    assert panel.view.viewport.start == pytest.approx(7000.0, abs=40.0)


def test_dragging_an_edge_zooms(panel):
    nav = panel.navigator
    panel.view.viewport.show(2000.0, 4000.0)
    panel.view.refresh_viewport()
    _mouse(nav, "press", _nav_x(nav, 4000.0), 15)
    _mouse(nav, "move", _nav_x(nav, 6000.0), 15)
    _mouse(nav, "release", _nav_x(nav, 6000.0), 15)
    assert panel.view.viewport.start == pytest.approx(2000.0, abs=40.0)
    assert panel.view.viewport.end == pytest.approx(6000.0, abs=40.0)


def test_a_stale_navigator_drag_does_not_pan(panel):
    nav = panel.navigator
    panel.view.viewport.show(2000.0, 4000.0)
    panel.view.refresh_viewport()
    _mouse(nav, "press", _nav_x(nav, 3000.0), 15)
    hover = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(_nav_x(nav, 6000.0), 15),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    nav.mouseMoveEvent(hover)
    assert panel.view.viewport.start == pytest.approx(2000.0)
    assert nav._drag is None


def test_the_toolbar_reports_the_range(panel):
    panel.view.set_selection(30, 59)
    assert panel.in_label.text() == "In 00:00:01:00"
    assert panel.out_label.text() == "Out 00:00:02:00"
    assert panel.dur_label.text() == "1.00 s"


def test_fit_shows_the_whole_session(panel):
    panel.view.viewport.show(2000.0, 3000.0)
    panel.fit_btn.click()
    assert (panel.view.viewport.start, panel.view.viewport.end) == (0.0, 10000.0)


def test_snap_toggle_reaches_the_view(panel):
    panel.snap.setChecked(False)
    assert panel.view._snap_on is False


def test_the_toolbar_belongs_to_the_timeline_tab(panel):
    panel.addTab(QTableWidget(), "Cohort")
    panel.setCurrentIndex(1)
    assert panel.cornerWidget(Qt.Corner.TopRightCorner).isHidden()
    panel.setCurrentIndex(0)
    assert not panel.cornerWidget(Qt.Corner.TopRightCorner).isHidden()


def test_the_navigator_strip_follows_the_device_pixel_ratio(panel):
    """Built at 1x it is blurry on a Retina screen."""
    nav = panel.navigator
    nav.devicePixelRatioF = lambda: 2.0
    assert nav._strip_pixmap().devicePixelRatio() == pytest.approx(2.0)


def test_a_whole_session_pulse_train_repaints_quickly(qtbot):
    """Zoomed out, every pan or zoom repaints every lane: this cost ~60 ms a lane."""
    import time

    frames = np.arange(0, 45001, dtype=float)  # 1500 s at 30 fps
    segments = []
    for i in range(15000):  # 10 Hz, 50 % duty, the whole session
        segments.append(Segment(i * 100.0, i * 100.0 + 50.0, 1.0, 1.0))
        segments.append(Segment(i * 100.0 + 50.0, (i + 1) * 100.0, 0.0, 0.0))
    train = Lane("led1", "led1", "board0", segments, [], pin_type="DIGITAL")
    view = TimelineView()
    qtbot.addWidget(view)
    view.resize(HEADER_W + 1400, 200)
    view.set_timeline(
        Timeline(
            lanes=[train],
            behavior=[],
            frame_map=FrameMap(frames, frames / 30.0 * 1000.0, "tracking"),
            flow_start_ms=0.0,
            flow_end_ms=1_500_000.0,
            start_ms=0.0,
            end_ms=1_500_000.0,
        )
    )
    rows = view._rows()
    view._static_pixmap(rows)  # warm: the lane's cached arrays
    view._static = None
    t0 = time.perf_counter()
    view._static_pixmap(rows)
    assert time.perf_counter() - t0 < 0.05


def test_column_coverage_is_exact():
    from glider.gui.review.timeline import _column_coverage

    a, b = np.array([0.2, 1.5, 3.9]), np.array([0.7, 3.25, 9.0])
    assert _column_coverage(a, b, 4).tolist() == pytest.approx([0.5, 0.5, 1.0, 0.35])


def test_a_behaviour_lane_is_tall_enough_to_read(view):
    view.set_timeline(_timeline())
    assert view.lane_rect("tracking").height() >= 40


def test_the_navigator_strip_is_tall_enough_to_read(panel):
    """The whole-session ethogram in the navigator is a band, not a hairline."""
    image = panel.navigator.grab().toImage()
    x = HEADER_W + 20
    groom = behavior_qcolor("groom", _ORDER)
    # The old 30 px navigator drew an 18 px strip; y=38 was off the widget.
    assert image.pixelColor(x, 9) == groom
    assert image.pixelColor(x, 38) == groom


def test_a_long_value_at_the_playhead_stays_clear_of_the_lane_name(qtbot):
    """The chip sits right of the title's column; a long label elides into it."""
    from PyQt6.QtGui import QFontMetrics

    from glider.gui.review.timeline import _VALUE_CHIP_LEFT, _fit_value
    from glider.gui.widgets.tool_ui import data_font

    metrics = QFontMetrics(data_font(8))
    text = _fit_value(metrics, "grooming_bilateral_face_wash", True)
    assert text != "grooming_bilateral_face_wash"
    width = metrics.horizontalAdvance(text) + 12 + 10
    assert HEADER_W - 8 - width >= _VALUE_CHIP_LEFT
    assert _fit_value(metrics, "rest", True) == "rest"
