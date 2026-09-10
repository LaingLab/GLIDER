"""The timeline bar must stay a drop-in for the ethogram bar."""

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
from glider.gui.styles import colors  # noqa: E402
from glider.gui.widgets.timeline_bar import TimelineBar, behavior_qcolor  # noqa: E402

_ORDER = ["groom", "locomote"]


def _timeline(*, hardware: bool = True, behavior: bool = True) -> Timeline:
    """A 10s, 30fps session: LED on for the first 2s, one RFID read."""
    frames = np.arange(0, 301, dtype=float)
    frame_map = FrameMap(frames=frames, ms=frames / 30.0 * 1000.0, source="tracking")
    lanes = (
        [
            Lane(
                key="led1",
                label="led1",
                board_id="board0",
                segments=[
                    Segment(start_ms=0.0, end_ms=2000.0, value=1.0, level=1.0),
                    Segment(start_ms=2000.0, end_ms=10000.0, value=0.0, level=0.0),
                ],
                markers=[Marker(at_ms=5000.0, label="tag-A7")],
            )
        ]
        if hardware
        else []
    )
    beh = (
        [
            BehaviorLane(
                source="tracking",
                labels=["groom"] * 150 + ["locomote"] * 151,
                frames=np.arange(0, 301),
            )
        ]
        if behavior
        else []
    )
    return Timeline(
        lanes=lanes,
        behavior=beh,
        frame_map=frame_map,
        flow_start_ms=0.0,
        flow_end_ms=10000.0,
        start_ms=0.0,
        end_ms=10000.0,
    )


@pytest.fixture
def bar(qtbot):
    widget = TimelineBar()
    qtbot.addWidget(widget)
    widget.resize(300, 120)
    return widget


def test_selection_is_emitted_in_frames(bar, qtbot):
    """The stats tables consume frames. If this ever emits milliseconds,
    every window statistic below the bar is silently computed over the
    wrong range — and nothing would raise."""
    bar.set_timeline(_timeline())
    with qtbot.waitSignal(bar.selection_changed) as blocker:
        bar.set_selection(10, 40)
    assert blocker.args == [10, 40]
    assert all(isinstance(v, int) for v in blocker.args)


def test_frame_bounds_come_from_the_frame_map(bar):
    bar.set_timeline(_timeline())
    assert bar.frame_bounds() == (0, 300)


def test_empty_timeline_has_zero_bounds(bar):
    bar.set_timeline(None)
    assert bar.frame_bounds() == (0, 0)


def test_selection_survives_scrubbing(bar):
    bar.set_timeline(_timeline())
    bar.set_selection(50, 200)
    bar.set_frame(120)
    assert bar.selection() == (50, 200)


def test_paints_without_a_timeline(bar, qtbot):
    """A bar with nothing loaded draws its empty state rather than raising."""
    bar.set_timeline(None)
    bar.show()
    qtbot.waitExposed(bar)
    bar.repaint()


def test_paints_a_loaded_timeline(bar, qtbot):
    bar.set_timeline(_timeline())
    bar.show()
    qtbot.waitExposed(bar)
    bar.repaint()


def test_behavior_fills_the_bar_when_there_is_no_hardware(bar, qtbot):
    """Protects test_analysis_window.py:488 — an ethogram-only session has
    no hardware lanes, so its single behaviour lane must still occupy the
    full height exactly as the old ethogram bar did."""
    bar.resize(300, 46)
    bar.set_timeline(_timeline(hardware=False))
    image = bar.grab().toImage()
    # Frame 75 of 300 sits near x=75 and is grooming top to bottom. Assert
    # the colour, not just that the two pixels agree: two untouched
    # background pixels agree too, so the weaker form passed against a
    # painter that drew nothing at all.
    groom = behavior_qcolor("groom", _ORDER)
    assert image.pixelColor(75, 8) == groom
    assert image.pixelColor(75, 40) == groom


def test_hardware_lane_is_drawn_below_behavior(bar, qtbot):
    """With hardware present the bar is no longer one flat band.

    Against the accent, not merely against the behaviour row: an unpainted
    hardware row is background, which differs from the behaviour band too,
    so the inequality alone held even with ``_paint_hardware`` stubbed out.
    """
    bar.set_timeline(_timeline())
    image = bar.grab().toImage()
    # x=20 is ~666ms, inside the LED's first-2s segment at level 1.0, and
    # y=110 is in the hardware row (behaviour takes the top 46px).
    assert image.pixelColor(20, 8) == behavior_qcolor("groom", _ORDER)
    assert image.pixelColor(20, 110) == QColor(colors.ACCENT)


def _pre_flow_timeline() -> Timeline:
    """The same 10s of camera, with 30s of device writes ahead of it.

    Which is the ordinary shape of a hardware session: ``build_timeline``
    deliberately extends the axis back over pre-flow device initialisation,
    so the camera occupies only the last quarter of the bar.
    """
    timeline = _timeline()
    return Timeline(
        lanes=timeline.lanes,
        behavior=timeline.behavior,
        frame_map=timeline.frame_map,
        flow_start_ms=0.0,
        flow_end_ms=10000.0,
        start_ms=-30000.0,
        end_ms=10000.0,
    )


def test_behavior_bands_sit_on_the_same_axis_as_everything_else(bar, qtbot):
    """Behaviour resolves on the TIME axis, like the playhead and the lanes.

    Slicing the frame axis into columns instead spread the ethogram across
    the full width even though the camera only covered the last quarter of
    it, so the playhead sat on one behaviour while the band under it showed
    another.
    """
    bar.resize(400, 46)
    bar.set_timeline(_pre_flow_timeline())

    # Frame 150 (5000ms) is where groom becomes locomote.
    boundary_x = bar._x_of_ms(bar._frame_map().ms_of(150))
    assert boundary_x == pytest.approx(350.0)

    image = bar.grab().toImage()
    assert image.pixelColor(int(boundary_x) - 1, 20) == behavior_qcolor("groom", _ORDER)
    assert image.pixelColor(int(boundary_x), 20) == behavior_qcolor("locomote", _ORDER)


def test_ethogram_only_timeline_paints_without_a_view(bar, qtbot):
    """``set_timeline`` alone is enough — no companion ``set_view`` call.

    ``build_timeline(None, view)`` has no frame map and no ms bounds, so the
    bar has to take its frame range from the behaviour lane itself. Taking
    it from ``self._view`` instead reported ``(0, 0)`` here, which collapsed
    every column onto frame 0 and painted the whole ethogram one colour.
    """
    view = SimpleNamespace(
        labels=["groom"] * 150 + ["locomote"] * 151,
        frames=np.arange(0, 301),
    )
    bar.resize(300, 46)
    bar.set_timeline(build_timeline(None, view))

    assert bar.frame_bounds() == (0, 300)
    image = bar.grab().toImage()
    assert image.pixelColor(75, 20) == behavior_qcolor("groom", _ORDER)
    assert image.pixelColor(250, 20) == behavior_qcolor("locomote", _ORDER)


def _windowed_timeline() -> Timeline:
    """A 20s recording whose ethogram scores only frames 200-400.

    The ordinary shape of a windowed apply run: the recording is long, the
    scored stretch is a slice out of the middle of it.
    """
    frames = np.arange(0, 601, dtype=float)
    return Timeline(
        lanes=[],
        behavior=[
            BehaviorLane(
                source="ethogram",
                labels=["groom"] * 201,
                frames=np.arange(200, 401),
            )
        ],
        frame_map=FrameMap(frames=frames, ms=frames / 30.0 * 1000.0, source="tracking"),
        flow_start_ms=0.0,
        flow_end_ms=20000.0,
        start_ms=0.0,
        end_ms=20000.0,
    )


def test_a_lane_paints_nothing_outside_its_own_frames(bar, qtbot):
    """Unscored stretches read as background, not as the nearest label.

    The ``hi <= lo`` fallback clamps a column to the nearest row, which is
    right for an interior column falling between two rows and wrong for a
    column past either end: 201 scored frames inside a 600-frame recording
    painted behaviour colour across the whole 600px bar, two thirds of it
    invented.
    """
    bar.resize(600, 46)
    bar.set_timeline(_windowed_timeline())
    image = bar.grab().toImage()

    background = QColor(colors.BASE)
    assert image.pixelColor(50, 20) == background  # before the scored window
    assert image.pixelColor(550, 20) == background  # after it
    # And the scored stretch is still drawn, so this is not just a blank bar.
    assert image.pixelColor(300, 20) == behavior_qcolor("groom", ["groom"])


def test_many_behavior_lanes_still_leave_the_hardware_a_row(bar):
    """``sizeHint`` has to allow for the gap ``_rows`` advances by.

    It summed the lane heights alone while ``_rows`` steps by height + gap,
    so the hardware rows were short by two pixels per behaviour lane. The
    vertical size policy is Fixed, so at seven behaviour lanes to one device
    — a small social assay — the raster was allotted zero height and simply
    vanished, with nothing raised anywhere.
    """
    base = _timeline()
    bar.set_timeline(
        Timeline(
            lanes=base.lanes,
            behavior=[
                BehaviorLane(
                    source=f"tracking[{i}]",
                    labels=["groom"] * 301,
                    frames=np.arange(0, 301),
                )
                for i in range(7)
            ],
            frame_map=base.frame_map,
            flow_start_ms=0.0,
            flow_end_ms=10000.0,
            start_ms=0.0,
            end_ms=10000.0,
        )
    )
    bar.resize(300, bar.sizeHint().height())
    heights = [height for kind, _lane, _top, height in bar._rows() if kind == "hardware"]
    assert heights and min(heights) > 0
