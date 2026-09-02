"""Tests for VideoFileSource — offline video frame access (scrub + sequential)."""

from pathlib import Path

import numpy as np

from glider.vision.video_source import VideoFileSource, video_resolution


def test_video_resolution_reads_the_header(synthetic_clip: Path):
    assert video_resolution(synthetic_clip) == (64, 48)


def test_video_resolution_is_none_when_unreadable(tmp_path: Path):
    """None is meaningful: callers size an arena with it, and a wrong size is
    worse than an absent one."""
    assert video_resolution(tmp_path / "nope.avi") is None


def test_load_reports_metadata(synthetic_clip: Path):
    src = VideoFileSource()
    assert src.load(synthetic_clip) is True
    assert src.frame_count == 12
    assert src.fps == 10.0
    assert src.resolution == (64, 48)
    src.release()


def test_load_rejects_missing_file(tmp_path: Path):
    src = VideoFileSource()
    assert src.load(tmp_path / "nope.avi") is False


def test_read_frame_returns_array(synthetic_clip: Path):
    src = VideoFileSource()
    src.load(synthetic_clip)
    frame = src.read_frame(5)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (48, 64, 3)
    src.release()


def test_frames_yields_every_frame_in_order(synthetic_clip: Path):
    src = VideoFileSource()
    src.load(synthetic_clip)
    indices = [n for n, _frame in src.frames()]
    assert indices == list(range(12))
    src.release()


def test_release_resets_state(synthetic_clip: Path):
    src = VideoFileSource()
    src.load(synthetic_clip)
    src.release()
    assert src.is_loaded is False
    assert src.frame_count == 0
    assert src.resolution == (0, 0)
    assert src.path is None


class _SeekRecorder:
    """Delegates to a real VideoCapture, noting every seek.

    cv2.VideoCapture attributes are read-only, so counting seeks means
    standing in front of the object rather than patching it.
    """

    def __init__(self, cap):
        self._cap = cap
        self.seeks: list[float] = []

    def set(self, prop, value):
        self.seeks.append(value)
        return self._cap.set(prop, value)

    def __getattr__(self, name):
        return getattr(self._cap, name)


def _record_seeks(src: VideoFileSource) -> _SeekRecorder:
    """Put a recorder in front of the capture the reader actually uses.

    Swapping only ``src._cap`` records nothing: the ExactFrameReader holds its
    own reference, so the seeks that matter go straight past the recorder.
    """
    recorder = _SeekRecorder(src._cap)
    src._cap = recorder
    src._reader._cap = recorder
    return recorder


def test_reading_forward_does_not_reseek(synthetic_clip: Path):
    """Playback is a run of consecutive reads; on a long-GOP codec a seek per
    frame re-decodes from the previous keyframe."""
    src = VideoFileSource()
    src.load(synthetic_clip)
    recorder = _record_seeks(src)

    for n in range(5):
        assert src.read_frame(n) is not None
    # One rewind to frame 0 to establish an exact position, then nothing:
    # every later frame is reached by grabbing onward.
    assert recorder.seeks == [0]
    src._cap = recorder._cap
    src.release()


def test_jumping_backwards_rewinds_rather_than_trusting_a_seek(synthetic_clip: Path):
    """Only a seek to frame 0 is exact on a long-GOP codec, so a backwards
    jump re-counts from there instead of seeking to the target."""
    src = VideoFileSource()
    src.load(synthetic_clip)
    src.read_frame(5)
    recorder = _record_seeks(src)

    assert src.read_frame(2) is not None
    assert recorder.seeks == [0]
    src._cap = recorder._cap
    src.release()


def test_the_fast_path_returns_the_same_frames_as_seeking(synthetic_clip: Path):
    """The optimisation must be invisible in the output."""
    sequential = VideoFileSource()
    sequential.load(synthetic_clip)
    walked = [sequential.read_frame(n) for n in range(6)]
    sequential.release()

    sought = VideoFileSource()
    sought.load(synthetic_clip)
    jumped = []
    for n in range(6):
        sought.read_frame(11)  # force a seek away every time
        jumped.append(sought.read_frame(n))
    sought.release()

    for a, b in zip(walked, jumped, strict=True):
        assert np.array_equal(a, b)


def test_read_frame_matches_a_sequential_decode(synthetic_clip: Path):
    """Scrubbing must return the same pixels as decoding straight through.

    Frame access is how pose is paired with video; an index that lands
    approximately puts the skeleton on the wrong frame.
    """
    import cv2

    cap = cv2.VideoCapture(str(synthetic_clip))
    expected = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        expected.append(frame)
    cap.release()

    src = VideoFileSource()
    src.load(synthetic_clip)
    try:
        for n in (9, 1, 6, 0, 11):
            assert np.array_equal(src.read_frame(n), expected[n]), f"frame {n}"
    finally:
        src.release()


def test_read_frame_is_still_exact_after_iterating(synthetic_clip: Path):
    """frames() drives the same decoder, so scrubbing afterwards must not
    trust a position the iterator moved out from under it."""
    import cv2

    cap = cv2.VideoCapture(str(synthetic_clip))
    expected = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        expected.append(frame)
    cap.release()

    src = VideoFileSource()
    src.load(synthetic_clip)
    try:
        assert np.array_equal(src.read_frame(8), expected[8])
        for _n, _frame in src.frames():
            pass
        assert np.array_equal(src.read_frame(3), expected[3])
        assert np.array_equal(src.read_frame(8), expected[8])
    finally:
        src.release()
