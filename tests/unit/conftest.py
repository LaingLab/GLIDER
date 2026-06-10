"""Shared fixtures for vision unit tests: synthesize a tiny deterministic clip."""

from pathlib import Path

import cv2
import numpy as np
import pytest

CLIP_FRAMES = 12
CLIP_W = 64
CLIP_H = 48
CLIP_FPS = 10.0


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A short MJPG/AVI clip: a white square sliding left-to-right on black.

    MJPG in an .avi container gives exact, portable frame counts across
    OpenCV builds. Skips the test if this build cannot open the writer.
    """
    path = tmp_path / "clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, CLIP_FPS, (CLIP_W, CLIP_H))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot open an MJPG writer")
    for i in range(CLIP_FRAMES):
        frame = np.zeros((CLIP_H, CLIP_W, 3), dtype=np.uint8)
        x = 4 + i * 4
        cv2.rectangle(frame, (x, 18), (x + 8, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path
