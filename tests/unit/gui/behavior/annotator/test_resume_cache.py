"""Tests for the resume-cache.

The annotator writes a JSON sidecar (`.annotate_queue.json`) recording the
sampled clip list plus a hash of the sampling inputs. On the next launch with
the same args we can pick up at the cached next-unlabeled index instead of
re-sampling. Changing any sampling input invalidates the cache.
"""

from __future__ import annotations


def test_resume_cache_round_trip(tmp_path):
    from glider.gui.behavior.annotator.resume_cache import ResumeCache

    inputs = {
        "videos": ["a.mp4", "b.mp4"],
        "n_clips": 50,
        "window": 30,
        "fps": 30.0,
        "random_state": 42,
        "spatial_weight": 1.0,
        "min_frame_gap": None,
    }
    cache = ResumeCache(tmp_path)
    cache.save(inputs=inputs, clip_payload=[{"video": "a.mp4", "start": 0, "end": 30}])
    loaded = cache.load(inputs=inputs)
    assert loaded is not None
    assert loaded["clips"] == [{"video": "a.mp4", "start": 0, "end": 30}]


def test_resume_cache_invalidates_on_input_change(tmp_path):
    from glider.gui.behavior.annotator.resume_cache import ResumeCache

    base = {
        "videos": ["a.mp4"],
        "n_clips": 50,
        "window": 30,
        "fps": 30.0,
        "random_state": 42,
        "spatial_weight": 1.0,
        "min_frame_gap": None,
    }
    cache = ResumeCache(tmp_path)
    cache.save(inputs=base, clip_payload=[{"video": "a.mp4"}])

    changed = dict(base, n_clips=51)
    assert cache.load(inputs=changed) is None


def test_resume_cache_path_resolution_with_common_parent(tmp_path):
    """When no --videos-dir is set, the cache lives in the common parent of
    the positional videos."""
    from glider.gui.behavior.annotator.resume_cache import resolve_cache_dir

    (tmp_path / "sub").mkdir()
    a = tmp_path / "sub" / "a.mp4"
    b = tmp_path / "sub" / "b.mp4"
    for f in (a, b):
        f.write_bytes(b"")
    assert resolve_cache_dir(videos=[a, b], videos_dir=None) == tmp_path / "sub"


def test_resume_cache_path_resolution_falls_back_to_first_parent(tmp_path, monkeypatch):
    """If `os.path.commonpath` raises (cross-drive paths on Windows), fall
    back to the parent of the first video."""
    from glider.gui.behavior.annotator import resume_cache as rc

    a = tmp_path / "ax" / "a.mp4"
    b = tmp_path / "bx" / "b.mp4"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"")
    b.write_bytes(b"")

    # Force the commonpath branch to fail to exercise the fallback.
    def _boom(_paths):
        raise ValueError("simulated cross-drive failure")

    monkeypatch.setattr(rc.os.path, "commonpath", _boom)
    result = rc.resolve_cache_dir(videos=[a, b], videos_dir=None)
    assert result == a.parent


def test_resume_cache_uses_videos_dir_when_provided(tmp_path):
    from glider.gui.behavior.annotator.resume_cache import resolve_cache_dir

    a = tmp_path / "a.mp4"
    a.write_bytes(b"")
    assert resolve_cache_dir(videos=[a], videos_dir=tmp_path) == tmp_path


def test_resume_cache_round_trip_with_proposedclip(tmp_path):
    """ProposedClip dataclasses serialize cleanly through the JSON cache."""
    from glider.gui.behavior.annotator.resume_cache import ResumeCache
    from glider.gui.behavior.annotator.sampler import ProposedClip

    clip = ProposedClip(
        window_index=42,
        center_frame=120,
        start_frame=100,
        end_frame=140,
        clip_seconds=0.6,
        video_path=str(tmp_path / "x.mp4"),
    )
    inputs = {"videos": [str(tmp_path / "x.mp4")], "n_clips": 1}
    cache = ResumeCache(tmp_path)
    cache.save(inputs=inputs, clip_payload=[clip.__dict__])
    loaded = cache.load(inputs=inputs)
    assert loaded is not None
    restored = [ProposedClip(**c) for c in loaded["clips"]]
    assert restored[0].center_frame == 120
    assert restored[0].video_path == str(tmp_path / "x.mp4")
