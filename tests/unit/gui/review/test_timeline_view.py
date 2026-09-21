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
