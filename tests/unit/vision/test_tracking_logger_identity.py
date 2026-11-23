"""Identity columns in the live tracking CSV."""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.cv_processor import TrackedObject
from glider.vision.tracking_logger import TrackingDataLogger


def _obj(tid, animal, visible=True, flag="", kp=False):
    return TrackedObject(
        track_id=tid,
        class_name="mouse",
        bbox=(10, 10, 20, 20),
        confidence=0.9,
        centroid=(20, 20),
        keypoints=np.array([[5.0, 5.0, 0.9]]) if kp else None,
        animal=animal,
        visible=visible,
        identity_flag=flag,
    )


def _rows(path):
    return [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]


@pytest.mark.asyncio
async def test_identity_columns_and_hidden_rows(tmp_path):
    lg = TrackingDataLogger(output_dir=tmp_path)
    lg.set_identity_columns(True)
    await lg.start("ident")
    lg.log_frame(1.0, [_obj(0, "A", flag="close", kp=True), _obj(1, "B", visible=False, kp=True)])
    path, kp_path = lg.file_path, lg.keypoints_file_path
    await lg.stop()
    rows = _rows(path)
    assert rows[0].endswith(",animal,identity_flag")
    assert len(rows) == 2 and rows[1].endswith(",A,close")
    assert len(_rows(kp_path)) == 2  # header + the visible animal's one keypoint


@pytest.mark.asyncio
async def test_default_header_unchanged(tmp_path):
    lg = TrackingDataLogger(output_dir=tmp_path)
    await lg.start("plain")
    path = lg.file_path
    await lg.stop()
    assert _rows(path)[0].endswith(",velocity_px_frame")
