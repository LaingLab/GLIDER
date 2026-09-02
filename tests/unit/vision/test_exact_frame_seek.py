"""Exact frame access by index, regardless of how badly the codec seeks.

``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, n)`` is only accurate on
all-keyframe codecs. On the long-GOP mp4s this project records, it lands
somewhere near ``n`` — measured at -5 to +8 frames on a 30 fps session — and
``get(CAP_PROP_POS_FRAMES)`` afterwards still reports ``n``, so the decoder
cannot be asked where it actually is. Grabbing forward from a seeked position
inherits the same error.

That matters because pose is paired with frames by index. A 2-frame error is
invisible while the animal is still and throws the skeleton clean off it while
the animal runs, which is exactly the artifact that motivated this module.

The fake capture below reproduces that behaviour so the guarantee can be
tested without shipping a long-GOP fixture. ``synthetic_clip`` is MJPG/AVI —
every frame a keyframe — so it cannot fail the way real footage does, and is
used here only to prove the reader behaves against a genuine cv2 capture.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from glider.vision.video_source import ExactFrameReader


class _MisSeekingCapture:
    """A capture whose non-zero seeks land ``error`` frames off target.

    Frame ``i`` is ``[[i]]``, so a test can name the frame it actually got.
    Seeking to 0 is exact — that is the one thing real captures get right, and
    the reader is allowed to rely on it.
    """

    def __init__(self, n_frames: int = 200, error: int = 5):
        self.n_frames = n_frames
        self.error = error
        self._true_pos = 0  # index the next grab() will decode
        self._held: int | None = None
        self.seeks: list[int] = []

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        target = int(value)
        self.seeks.append(target)
        self._true_pos = 0 if target == 0 else min(target + self.error, self.n_frames)
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            # Deliberately a lie, as real captures are: it reports what was
            # asked for, not where the decoder landed.
            return float(self.seeks[-1] if self.seeks else self._true_pos)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.n_frames)
        return 0.0

    def grab(self):
        if self._true_pos >= self.n_frames:
            self._held = None
            return False
        self._held = self._true_pos
        self._true_pos += 1
        return True

    def retrieve(self):
        if self._held is None:
            return False, None
        return True, np.array([[self._held]], dtype=np.int32)

    def read(self):
        return self.retrieve() if self.grab() else (False, None)


def _index_of(frame) -> int:
    return int(frame[0][0])


def test_reads_the_frame_asked_for_when_seeking_is_inexact():
    """The whole point: index in, that exact frame out."""
    reader = ExactFrameReader(_MisSeekingCapture(error=5))
    assert _index_of(reader.read(120)) == 120


def test_naive_seek_would_have_been_wrong():
    """Guards the fake: without correction this lands 5 frames late, so the
    test above is failing for the right reason rather than by accident."""
    cap = _MisSeekingCapture(error=5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
    ok, frame = cap.read()
    assert ok and _index_of(frame) == 125


def test_walking_forward_stays_exact():
    reader = ExactFrameReader(_MisSeekingCapture(error=7))
    assert [_index_of(reader.read(n)) for n in (10, 11, 12, 40)] == [10, 11, 12, 40]


def test_jumping_backwards_stays_exact():
    reader = ExactFrameReader(_MisSeekingCapture(error=7))
    assert _index_of(reader.read(150)) == 150
    assert _index_of(reader.read(3)) == 3
    assert _index_of(reader.read(90)) == 90


def test_walking_forward_does_not_rewind():
    """Sequential reads must not re-decode from 0 each time, or playback of a
    20,000-frame session becomes quadratic."""
    cap = _MisSeekingCapture(error=5)
    reader = ExactFrameReader(cap)
    for n in range(30, 40):
        reader.read(n)
    assert cap.seeks.count(0) == 1


def test_returns_none_past_the_end():
    reader = ExactFrameReader(_MisSeekingCapture(n_frames=50))
    assert reader.read(60) is None


def test_negative_index_is_refused():
    reader = ExactFrameReader(_MisSeekingCapture())
    with pytest.raises(ValueError):
        reader.read(-1)


def test_matches_sequential_decode_on_a_real_capture(synthetic_clip: Path):
    """Against a genuine cv2 capture, every index returns the same pixels a
    straight sequential decode gives for that index."""
    cap = cv2.VideoCapture(str(synthetic_clip))
    expected = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        expected.append(frame)
    cap.release()

    cap = cv2.VideoCapture(str(synthetic_clip))
    reader = ExactFrameReader(cap)
    try:
        for n in (7, 2, 11, 0, 5):
            got = reader.read(n)
            assert got is not None
            assert np.array_equal(got, expected[n]), f"frame {n} differs"
    finally:
        cap.release()
