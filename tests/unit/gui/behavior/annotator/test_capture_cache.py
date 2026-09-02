"""Tests for the LRU `VideoCaptureCache`.

The cache is a tiny convenience: keep a small fixed number of cv2.VideoCapture
handles open so jumping between videos in the multi-video annotator session
doesn't re-open + seek on every clip. The test uses a fake-capture stub so it
runs without OpenCV present.
"""

from __future__ import annotations


class _FakeCap:
    """Stand-in for cv2.VideoCapture. Tracks open/close state."""

    instances: list[_FakeCap] = []

    def __init__(self, path):
        self.path = path
        self.opened = True
        _FakeCap.instances.append(self)

    def isOpened(self):
        return self.opened

    def release(self):
        self.opened = False

    # Minimal cv2.VideoCapture API surface so ClipPlayer can drive a fake
    # without crashing; we don't care about returned frames in cache tests.
    def set(self, *args, **kwargs):  # noqa: A003 - mirrors cv2 API
        return True

    def read(self):
        return False, None

    # ExactFrameReader walks with grab()/retrieve() rather than seeking, so a
    # stand-in has to model that part of the VideoCapture contract too.
    def grab(self):
        return False

    def retrieve(self):
        return False, None


def test_capture_cache_returns_same_handle_for_repeated_path(monkeypatch):
    from glider.gui.behavior.annotator import capture_cache

    _FakeCap.instances.clear()
    monkeypatch.setattr(capture_cache, "_open_capture", lambda p: _FakeCap(p))

    cache = capture_cache.VideoCaptureCache(max_open=3)
    a = cache.get("/tmp/x.mp4")
    b = cache.get("/tmp/x.mp4")
    assert a is b
    assert len(_FakeCap.instances) == 1


def test_capture_cache_evicts_least_recently_used(monkeypatch):
    from glider.gui.behavior.annotator import capture_cache

    _FakeCap.instances.clear()
    monkeypatch.setattr(capture_cache, "_open_capture", lambda p: _FakeCap(p))

    cache = capture_cache.VideoCaptureCache(max_open=2)
    a = cache.get("/tmp/a.mp4")
    b = cache.get("/tmp/b.mp4")
    cache.get("/tmp/a.mp4")  # mark a as most-recent
    c = cache.get("/tmp/c.mp4")  # this should evict b, not a
    assert b.opened is False
    assert a.opened is True
    assert c.opened is True


def test_capture_cache_close_all_releases_handles(monkeypatch):
    from glider.gui.behavior.annotator import capture_cache

    _FakeCap.instances.clear()
    monkeypatch.setattr(capture_cache, "_open_capture", lambda p: _FakeCap(p))

    cache = capture_cache.VideoCaptureCache(max_open=3)
    cache.get("/tmp/a.mp4")
    cache.get("/tmp/b.mp4")
    cache.close_all()
    assert all(not inst.opened for inst in _FakeCap.instances)


def test_clip_player_uses_injected_cache(monkeypatch, tmp_path):
    """When a VideoCaptureCache is passed to ClipPlayer, the player gets its
    cv2.VideoCapture from the cache instead of opening one privately.

    Skips if PyQt6 / OpenCV aren't available.
    """
    pytest = __import__("pytest")
    try:
        import cv2  # noqa: F401
        from PyQt6.QtWidgets import QApplication  # noqa: F401
    except ImportError:
        pytest.skip("PyQt6 + OpenCV required")
    from glider.gui.behavior.annotator import capture_cache
    from glider.gui.behavior.annotator.clip_player import ClipPlayer

    _FakeCap.instances.clear()
    monkeypatch.setattr(capture_cache, "_open_capture", lambda p: _FakeCap(p))

    # Need a real path so the existence check in set_clip passes.
    video = tmp_path / "fake.mp4"
    video.write_bytes(b"")

    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    cache = capture_cache.VideoCaptureCache(max_open=2)
    player = ClipPlayer(capture_cache=cache)
    # set_clip will short-circuit on the fake cap (we don't have a real video
    # to decode); we just need to confirm it asked the cache.
    player.set_clip(video, 0, 1, fps=30.0)
    assert len(_FakeCap.instances) == 1
    assert _FakeCap.instances[0].path == str(video)
