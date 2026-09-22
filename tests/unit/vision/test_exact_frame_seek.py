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


# ---------------------------------------------------------------------------
# Seeking near and counting the rest, by the frame's own timestamp.
#
# Counting from 0 on every backward jump made Session Review crawl: stepping
# back one frame at 40,000 re-decoded 40,000 frames. A decoded frame carries
# its own timestamp, which says where a seek really landed even when the seek
# itself was inexact, so the reader seeks near, reads the landing, and walks
# the last few frames -- falling back to counting when a file's timestamps
# do not name its frames.


class _StampedCapture(_MisSeekingCapture):
    """Mis-seeks like the capture above, but stamps each frame as FFmpeg does."""

    def __init__(self, n_frames: int = 2000, error: int = 5, fps: float = 30.0, stamp=None):
        super().__init__(n_frames, error)
        self.fps = fps
        self.grabs = 0
        self._stamp = stamp or (lambda i: i * 1000.0 / fps)

    def grab(self):
        self.grabs += 1
        return super().grab()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return 0.0 if self._held is None else self._stamp(self._held)
        return super().get(prop)


def test_a_backward_jump_seeks_instead_of_decoding_from_the_start():
    cap = _StampedCapture(error=5)
    reader = ExactFrameReader(cap)
    assert _index_of(reader.read(1500)) == 1500
    cap.grabs = 0
    assert _index_of(reader.read(1200)) == 1200
    assert cap.grabs < 40


def test_a_far_jump_ahead_seeks_too():
    cap = _StampedCapture(error=5)
    reader = ExactFrameReader(cap)
    reader.read(10)
    cap.grabs = 0
    assert _index_of(reader.read(1900)) == 1900
    assert cap.grabs < 40


def test_stepping_back_one_frame_is_cheap():
    cap = _StampedCapture(error=5)
    reader = ExactFrameReader(cap)
    reader.read(1000)
    cap.grabs = 0
    assert _index_of(reader.read(999)) == 999
    assert cap.grabs < 40


def test_a_seek_that_lands_past_the_target_aims_earlier():
    cap = _StampedCapture(error=40)
    reader = ExactFrameReader(cap)
    assert _index_of(reader.read(900)) == 900
    assert _index_of(reader.read(500)) == 500


def test_the_same_frame_twice_decodes_once():
    cap = _StampedCapture()
    reader = ExactFrameReader(cap)
    reader.read(700)
    cap.grabs = 0
    assert _index_of(reader.read(700)) == 700
    assert cap.grabs == 0


def test_timestamps_that_do_not_count_frames_fall_back_to_counting():
    """A variable-rate file: a frame's time no longer names its index."""
    cap = _StampedCapture(error=5, stamp=lambda i: i * 1000.0 / 30.0 * 1.5)
    reader = ExactFrameReader(cap)
    assert [_index_of(reader.read(n)) for n in (1500, 1200, 10, 1300)] == [1500, 1200, 10, 1300]


@pytest.mark.parametrize("stamp", [lambda i: 0.0, lambda i: i * 1000.0 / 30.0 * 0.5])
def test_timestamps_that_land_nowhere_near_the_seek_are_not_trusted(stamp):
    """No timestamps at all, or ones running slow: fall back to counting."""
    cap = _StampedCapture(error=5, stamp=stamp)
    reader = ExactFrameReader(cap)
    assert [_index_of(reader.read(n)) for n in (1500, 1200, 1900)] == [1500, 1200, 1900]


def _numbered_clip(path: Path, n: int, fourcc: str) -> bool:
    """A long-GOP clip whose frames carry their index as 12 bright/dark blocks."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 30.0, (320, 240))
    if not writer.isOpened():
        return False
    for i in range(n):
        image = np.full((240, 320, 3), 40, np.uint8)
        for bit in range(12):
            if i >> bit & 1:
                row, col = divmod(bit, 6)
                image[20 + row * 50 : 60 + row * 50, 10 + col * 50 : 50 + col * 50] = 230
        image[150:190, (i * 7) % 280 : (i * 7) % 280 + 40] = (0, 180, 255)  # motion
        writer.write(image)
    writer.release()
    return True


def _number_of(image) -> int:
    value = 0
    for bit in range(12):
        row, col = divmod(bit, 6)
        if image[28 + row * 50 : 52 + row * 50, 18 + col * 50 : 42 + col * 50].mean() > 135:
            value |= 1 << bit
    return value


@pytest.mark.parametrize("fourcc", ["mp4v", "avc1"])
def test_exact_on_a_real_long_gop_file(tmp_path: Path, fourcc: str):
    path = tmp_path / f"clip_{fourcc}.mp4"
    if not _numbered_clip(path, 900, fourcc):
        pytest.skip(f"this OpenCV cannot write {fourcc}")
    cap = cv2.VideoCapture(str(path))
    reader = ExactFrameReader(cap)
    try:
        for n in (600, 30, 899, 450, 449, 0, 777):
            assert _number_of(reader.read(n)) == n
    finally:
        cap.release()
