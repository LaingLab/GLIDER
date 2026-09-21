"""The viewer: the right video frame, the HUD over it, and the transport."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor  # noqa: E402

from glider.analysis.behavior.session_view import SessionView  # noqa: E402
from glider.gui.review.viewer import KeypointCanvas, Transport  # noqa: E402
from glider.gui.styles import colors  # noqa: E402


class _Reader:
    def __init__(self):
        self.asked = []

    def read_frame(self, index):
        self.asked.append(index)
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        pass


def _view(**overrides) -> SessionView:
    xy = np.full((300, 1, 2), np.nan)
    xy[:, 0, 0] = np.arange(300)
    xy[:, 0, 1] = 100.0
    fields = {
        "labels": ["groom"] * 300,
        "frames": np.arange(300),
        "fps": 30.0,
        "xy": xy,
        "keypoint_names": ["centroid"],
        "resolution": (640, 480),
    }
    fields.update(overrides)
    return SessionView(**fields)


@pytest.fixture
def canvas(qtbot):
    widget = KeypointCanvas()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    return widget


def test_the_video_is_read_at_the_session_frame_minus_the_offset(canvas):
    canvas.set_view(_view(video_path=Path("fake.mp4"), first_video_frame=1))
    reader = _Reader()
    canvas._reader = reader
    canvas._frame_image(5)
    assert reader.asked == [4]


def test_a_frame_before_the_video_starts_reads_nothing(canvas):
    canvas.set_view(_view(video_path=Path("fake.mp4"), first_video_frame=1))
    reader = _Reader()
    canvas._reader = reader
    assert canvas._frame_image(0) is None
    assert reader.asked == []


def test_the_hud_draws_the_behaviour_chip(canvas):
    canvas.set_view(_view())
    colour = QColor(colors.LANE_MOTOR)
    canvas.set_hud(("groom  1.20 s in", colour), [])
    assert canvas.grab().toImage().pixelColor(25, 21) == colour
    canvas.set_show_hud(False)
    assert canvas.grab().toImage().pixelColor(25, 21) != colour


def test_hardware_chips_sit_top_right(canvas):
    canvas.set_view(_view())
    colour = QColor(colors.LANE_OUTPUT)
    canvas.set_hud(None, [("LED 470 nm", colour)])
    image = canvas.grab().toImage()
    assert any(image.pixelColor(x, 21) == colour for x in range(200, 400))


def test_poses_can_be_hidden(canvas):
    canvas.set_view(_view())
    canvas.set_trail(5.0, False)
    canvas.set_frame(100)
    scale, dx, dy = canvas._transform()
    x, y = int(100 * scale + dx), int(100 * scale + dy)
    shown = canvas.grab().toImage().pixelColor(x, y)
    canvas.set_show_poses(False)
    assert canvas.grab().toImage().pixelColor(x, y) != shown


def test_the_transport_has_every_toggle(qtbot):
    transport = Transport()
    qtbot.addWidget(transport)
    toggles = [
        transport.video_on,
        transport.poses_on,
        transport.trail_on,
        transport.heatmap_on,
        transport.zones_on,
        transport.hud_on,
    ]
    assert [t.text() for t in toggles] == ["Video", "Poses", "Trail", "Heatmap", "Zones", "HUD"]
    assert transport.heatmap_on.isChecked() is False
    assert transport.video_on.isChecked() is True
    assert "Play" in transport.play.text()


def test_step_buttons_do_not_look_like_play(qtbot):
    transport = Transport()
    qtbot.addWidget(transport)
    assert transport.forward.text() != "▶"
    assert "▶" not in {transport.back.text(), transport.forward.text()} - {"▕▶"}
    assert (transport.back.text(), transport.forward.text()) == ("◀▏", "▕▶")
