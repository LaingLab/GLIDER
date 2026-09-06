"""Dropped frames have to be visible while recording, not only afterwards.

MultiVideoRecorder already counts drops per camera and logs a total when the
run stops. At sixteen cameras that is too late: a single starved writer is the
failure that quietly shortens one animal's recording, and the operator needs to
see it while there is still time to stop and fix it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glider.vision.multi_video_recorder import MultiVideoRecorder


@pytest.fixture
def recorder():
    return MultiVideoRecorder(MagicMock())


class TestDropVisibility:
    def test_a_fresh_recorder_reports_no_drops(self, recorder):
        assert recorder.frames_dropped == {}

    def test_drops_are_readable_per_camera_mid_run(self, recorder):
        recorder._frames_dropped = {"cam_0": 3, "cam_7": 0}
        assert recorder.frames_dropped == {"cam_0": 3, "cam_7": 0}

    def test_the_total_is_available_without_summing_by_hand(self, recorder):
        recorder._frames_dropped = {"cam_0": 3, "cam_1": 4, "cam_2": 0}
        assert recorder.total_frames_dropped == 7

    def test_the_caller_cannot_corrupt_the_counters(self, recorder):
        recorder._frames_dropped = {"cam_0": 3}
        recorder.frames_dropped["cam_0"] = 999
        assert recorder._frames_dropped["cam_0"] == 3
