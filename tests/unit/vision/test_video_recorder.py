"""Tests for VideoFormat defaults and the codec-fallback writer helper."""

from unittest.mock import MagicMock, patch

import pytest

from glider.vision.video_recorder import (
    FALLBACK_CODEC,
    VideoFormat,
    open_video_writer,
)


def _fake_writer(opened: bool) -> MagicMock:
    """Build a mock cv2.VideoWriter with the given isOpened() result."""
    writer = MagicMock()
    writer.isOpened.return_value = opened
    return writer


def test_default_codec_is_h264():
    """New recordings default to H.264 (avc1), the modern successor to mp4v."""
    assert VideoFormat().codec == "avc1"
    assert VideoFormat().extension == ".mp4"


def test_from_dict_defaults_to_h264():
    """Loading a format dict without a codec key uses the new default."""
    assert VideoFormat.from_dict({}).codec == "avc1"
    # An explicit legacy codec is still honored.
    assert VideoFormat.from_dict({"codec": "mp4v"}).codec == "mp4v"


def test_open_video_writer_uses_primary_when_available():
    """When the preferred codec opens, no fallback is attempted."""
    writer = _fake_writer(opened=True)
    with patch("glider.vision.video_recorder.cv2.VideoWriter", return_value=writer) as ctor:
        result, codec_used = open_video_writer("out.mp4", "avc1", 30.0, (640, 480))

    assert result is writer
    assert codec_used == "avc1"
    # Only one writer constructed: no fallback path taken.
    assert ctor.call_count == 1


def test_open_video_writer_falls_back_when_primary_unavailable():
    """If the preferred codec won't open, retry once with the fallback codec."""
    failed = _fake_writer(opened=False)
    succeeded = _fake_writer(opened=True)
    with patch(
        "glider.vision.video_recorder.cv2.VideoWriter",
        side_effect=[failed, succeeded],
    ) as ctor:
        result, codec_used = open_video_writer("out.mp4", "avc1", 30.0, (640, 480))

    assert result is succeeded
    assert codec_used == FALLBACK_CODEC
    assert ctor.call_count == 2
    # The failed handle is released rather than leaked.
    failed.release.assert_called_once()


def test_open_video_writer_returns_none_when_all_fail():
    """If neither the preferred nor fallback codec opens, return (None, None)."""
    with patch(
        "glider.vision.video_recorder.cv2.VideoWriter",
        side_effect=[_fake_writer(opened=False), _fake_writer(opened=False)],
    ):
        result, codec_used = open_video_writer("out.mp4", "avc1", 30.0, (640, 480))

    assert result is None
    assert codec_used is None


def test_open_video_writer_no_duplicate_attempt_when_codec_is_fallback():
    """When the requested codec already equals the fallback, only try once."""
    failed = _fake_writer(opened=False)
    with patch(
        "glider.vision.video_recorder.cv2.VideoWriter",
        return_value=failed,
    ) as ctor:
        result, codec_used = open_video_writer("out.mp4", "mp4v", 30.0, (640, 480))

    assert result is None
    assert codec_used is None
    # No redundant second attempt with the same codec.
    assert ctor.call_count == 1


@pytest.mark.parametrize("codec", ["avc1", "AVC1", "mp4v"])
def test_open_video_writer_fallback_case_insensitive(codec):
    """Fallback de-duplication is case-insensitive against the requested codec."""
    succeeded = _fake_writer(opened=True)
    with patch(
        "glider.vision.video_recorder.cv2.VideoWriter",
        return_value=succeeded,
    ) as ctor:
        result, codec_used = open_video_writer(
            "out.mp4", codec, 30.0, (640, 480), fallback_codec="MP4V"
        )

    assert result is succeeded
    assert codec_used == codec
    # First attempt succeeds, so exactly one construction regardless of case.
    assert ctor.call_count == 1
