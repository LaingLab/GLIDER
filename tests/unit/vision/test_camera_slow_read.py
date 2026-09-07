"""A camera too slow to stream should say so, and say which one it is.

Two cameras on an eight-camera rig blocked for 1000 ms per frame. They opened
cleanly, reported the right resolution and pixel format, and then failed to
stream -- and all the log said was "Failed to retrieve frame (attempt 1)", with
no camera named and no measurement of the read that had already given the
answer away. Working out which two took reading timestamps out of the log by
hand.

1000 ms is a one-second exposure, which is where auto-exposure caps on those
cameras when the arena is dark. The point of these tests is that the warning
carries enough to reach that conclusion: which camera, how slow, and what
usually causes it.
"""

from __future__ import annotations

import logging

import pytest

from glider.vision.camera_manager import CameraManager


@pytest.fixture
def manager():
    camera = CameraManager()
    camera._settings.camera_index = 6
    camera._settings.fps = 30
    return camera


def _warnings(caplog) -> list[str]:
    return [r.message for r in caplog.records if r.levelno >= logging.WARNING]


class TestSlowRead:
    def test_a_one_second_read_is_reported(self, manager, caplog):
        with caplog.at_level(logging.WARNING):
            manager._warn_if_slow(1000.0)
        assert len(_warnings(caplog)) == 1

    def test_it_names_the_camera(self, manager, caplog):
        # The whole gap: with eight cameras running, an unattributed warning
        # says one of eight is broken and nothing more.
        with caplog.at_level(logging.WARNING):
            manager._warn_if_slow(1000.0)
        assert "Camera 6" in _warnings(caplog)[0]

    def test_it_gives_the_measurement_and_the_target(self, manager, caplog):
        with caplog.at_level(logging.WARNING):
            manager._warn_if_slow(1000.0)
        message = _warnings(caplog)[0]
        assert "1000 ms" in message
        assert "1.0 fps" in message  # the rate that implies
        assert "30 fps" in message  # what was asked for

    def test_it_names_the_usual_cause(self, manager, caplog):
        # "This camera is slow" is a restatement of the symptom. Naming
        # exposure is what turns it into something to go and check.
        with caplog.at_level(logging.WARNING):
            manager._warn_if_slow(1000.0)
        assert "exposure" in _warnings(caplog)[0].lower()

    def test_a_healthy_read_says_nothing(self, manager, caplog):
        # A healthy camera returns a buffered frame in about a millisecond.
        with caplog.at_level(logging.WARNING):
            manager._warn_if_slow(2.0)
        assert _warnings(caplog) == []

    def test_a_merely_sluggish_read_says_nothing(self, manager, caplog):
        # The first frame after opening a device is legitimately slow, so the
        # threshold sits where a camera can no longer hold a usable rate --
        # not where it is briefly slower than ideal.
        with caplog.at_level(logging.WARNING):
            manager._warn_if_slow(manager._SLOW_READ_MS - 1)
        assert _warnings(caplog) == []

    def test_a_zero_length_read_does_not_divide_by_zero(self, manager, caplog):
        with caplog.at_level(logging.WARNING):
            manager._warn_if_slow(0.0)
        assert _warnings(caplog) == []
