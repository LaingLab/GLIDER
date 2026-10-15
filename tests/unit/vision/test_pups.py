"""Pups from a semantic mask: splitting huddles and keeping identities.

Synthetic masks of ellipse "pups" at known positions, so every slot's true
pup is known by construction.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from glider.vision.pose.pups import (
    PupTracks,
    contours_from_mask,
    load_pups,
    pups_path,
    save_pups,
    segment_video,
    split_blobs,
    track_pups,
    track_video_pups,
    tracks_from_result,
    unit_area,
)

H, W = 240, 320


def _mask(centres):
    m = np.zeros((H, W), np.uint8)
    for x, y in centres:
        if np.isfinite(x):
            cv2.ellipse(m, (round(x), round(y)), (14, 8), 0, 0, 360, 1, -1)
    return m


def _frames(paths):
    """Per-frame contours for pups following ``paths`` (pups, frames, 2)."""
    return [contours_from_mask(_mask(paths[:, f])) for f in range(paths.shape[1])]


def _slot_of(result, paths):
    """Which slot holds each true pup, judged by mean distance."""
    tracks = np.stack([result.tracks[s].xy[:, 0, :] for s in result.tracks])
    owner = []
    for p in range(len(paths)):
        d = np.nanmean(np.linalg.norm(tracks - paths[p][None], axis=2), axis=1)
        owner.append(int(np.argmin(d)))
    return owner, tracks


def test_specks_dropped_and_touching_pups_split():
    m = _mask([(60, 60), (200, 120)])
    m[10:13, 10:13] = 1  # a 9 px speck
    assert len(contours_from_mask(m)) == 2

    unit = unit_area([contours_from_mask(_mask([(60, 60)]))])
    huddle = contours_from_mask(_mask([(100, 100), (126, 100)]))
    assert len(huddle) == 1  # semantic: one blob
    parts = split_blobs(huddle, unit)
    assert len(parts) == 2 and all(p.merged == 2 for p in parts)
    xs = sorted(p.x for p in parts)
    assert abs(xs[0] - 100) < 4 and abs(xs[1] - 126) < 4


def test_identities_hold_through_occlusion_huddle_and_carry():
    n = 120
    start = np.array([[40, 40], [40, 200], [160, 40], [160, 200], [280, 120]], float)
    paths = np.repeat(start[:, None, :], n, axis=1)
    paths[1, :, 0] += np.linspace(0, 20, n)  # crawls a little
    paths[0, 30:40] = np.nan  # under the dam for 10 frames
    paths[3, 50:70] = np.nan  # carried: hidden, then put down far away
    paths[3, 70:] = [240, 60]
    paths[2, 70:80] = np.linspace([160, 40], [216, 60], 10)  # nudged over to pup 3...
    paths[2, 80:] = [216, 60]  # ...and left touching it: one blob
    frames = _frames(paths)

    result, blobs = track_pups(frames, n_pups=5, fps=15.0)
    owner, tracks = _slot_of(result, paths)
    assert sorted(owner) == [0, 1, 2, 3, 4]  # one slot per pup, no sharing

    for p, s in enumerate(owner):
        seen = np.isfinite(paths[p][:, 0])
        err = np.linalg.norm(tracks[s][seen] - paths[p][seen], axis=1)
        assert np.nanmax(err) < 6, f"pup {p} drifted onto another pup"
    assert np.isnan(tracks[owner[0]][30:40]).all()  # hidden reads missing, not borrowed
    assert any(b.merged == 2 for b in blobs[90])


def test_split_is_reproducible():
    frames = [contours_from_mask(_mask([(100, 100), (124, 104), (128, 92)]))] * 3
    unit = unit_area([contours_from_mask(_mask([(60, 60)]))])
    a = [(b.x, b.y) for b in split_blobs(frames[0], unit)]
    assert len(split_blobs([c.astype(np.int16) for c in frames[0]], unit)) == len(a)
    b = [(b.x, b.y) for b in split_blobs(frames[0], unit)]
    assert a == b


def _tracks(n_frames=6):
    paths = np.array(
        [
            [(60.0 + 3 * f, 60.0) for f in range(n_frames)],
            [(200.0, 150.0 - 2 * f) for f in range(n_frames)],
        ]
    )
    result, blobs = track_pups(_frames(paths), n_pups=2, fps=15.0, min_fragment_s=0)
    return tracks_from_result(result, blobs, n_pups=2, meta={"litter_size": 2, "fps": 15.0})


def test_pups_path_sits_beside_the_video(tmp_path):
    assert pups_path(tmp_path / "litter1_pre.avi") == tmp_path / "litter1_pre_pups.npz"


def test_tracks_from_result_keeps_every_slot_and_frame():
    t = _tracks()
    assert (t.n_pups, t.n_frames) == (2, 6)
    assert t.names == ["pup_0", "pup_1"]
    assert t.stitched.shape == (2, 6) and t.stitched.dtype == bool
    assert all(len(frame) == 2 for frame in t.outlines)


def test_save_and_load_round_trip(tmp_path):
    t = _tracks()
    path = tmp_path / "v_pups.npz"
    save_pups(path, t)
    back = load_pups(path)
    np.testing.assert_allclose(back.xy, t.xy, equal_nan=True)
    np.testing.assert_array_equal(back.stitched, t.stitched)
    assert back.meta == t.meta
    for a, b in zip(back.outlines, t.outlines, strict=True):
        assert len(a) == len(b)
        for oa, ob in zip(a, b, strict=True):
            np.testing.assert_array_equal(oa, ob)
    assert not list(tmp_path.glob("*.tmp"))  # atomic: no temp file left


def test_a_frame_with_no_pups_round_trips(tmp_path):
    t = _tracks()
    empty = PupTracks(t.xy, [[]] + t.outlines[1:], t.stitched, t.meta)
    save_pups(tmp_path / "e_pups.npz", empty)
    assert load_pups(tmp_path / "e_pups.npz").outlines[0] == []


def test_outline_for_picks_the_outline_nearest_the_pup():
    t = _tracks()
    outline = t.outline_for(0, 2)
    assert outline is not None
    assert np.hypot(*(outline.mean(axis=0) - t.xy[0, 2])) < 10
    nan_xy = t.xy.copy()
    nan_xy[0, 2] = np.nan
    assert PupTracks(nan_xy, t.outlines, t.stitched, t.meta).outline_for(0, 2) is None


def test_a_bad_file_raises_value_error(tmp_path):
    bad = tmp_path / "bad_pups.npz"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="bad_pups.npz"):
        load_pups(bad)


def test_a_damaged_or_empty_file_raises_value_error(tmp_path):
    good = tmp_path / "good_pups.npz"
    save_pups(good, _tracks())
    raw = bytearray(good.read_bytes())
    raw[53:61] = bytes(b ^ 0xFF for b in raw[53:61])  # inside the first member's deflate stream
    damaged = tmp_path / "damaged_pups.npz"
    damaged.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="damaged_pups.npz"):
        load_pups(damaged)
    empty = tmp_path / "empty_pups.npz"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty_pups.npz"):
        load_pups(empty)


def test_stitched_frames_survive_the_round_trip(tmp_path):
    t = _tracks()
    stitched = t.stitched.copy()
    stitched[1, 3] = True
    t = PupTracks(t.xy, t.outlines, stitched, t.meta)
    save_pups(tmp_path / "s_pups.npz", t)
    np.testing.assert_array_equal(load_pups(tmp_path / "s_pups.npz").stitched, stitched)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_a_saved_pups_file_is_not_private(tmp_path):
    path = tmp_path / "v_pups.npz"
    save_pups(path, _tracks())
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask  # shared folders read it


class _Mask:
    def __init__(self, array):
        self.data = array  # numpy here; ultralytics gives a torch tensor


def _result(mask, orig_shape):
    return SimpleNamespace(semantic_mask=_Mask(mask), orig_shape=orig_shape)


class _Model:
    """Yields one semantic result per mask, like YOLO(...).predict(stream=True)."""

    def __init__(self, masks, orig_shape=(H, W)):
        self.masks, self.orig_shape = masks, orig_shape
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return (_result(m, self.orig_shape) for m in self.masks)


def test_segment_video_returns_contours_per_frame():
    masks = [_mask([(60, 60), (200, 150)]), _mask([(63, 60), (200, 148)])]
    frames, size = segment_video(_Model(masks), Path("v.mp4"), device="cpu")
    assert size == (W, H)
    assert [len(f) for f in frames] == [2, 2]


def test_a_mask_smaller_than_the_video_is_scaled_to_video_pixels():
    # The model returned a half-size mask for a full-size video.
    small = cv2.resize(_mask([(100, 100)]), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    frames, _ = segment_video(_Model([small], orig_shape=(H, W)), Path("v.mp4"), device="cpu")
    (contour,) = frames[0]
    assert np.hypot(*(contour.mean(axis=0) - (100, 100))) < 6


def test_segment_video_stops_when_cancelled():
    from glider.vision.pose.core import PoseCancelledError

    with pytest.raises(PoseCancelledError):
        segment_video(
            _Model([_mask([(60, 60)])] * 3), Path("v.mp4"), device="cpu", cancel_cb=lambda: True
        )


def test_track_video_pups_writes_the_npz(tmp_path, monkeypatch):
    import glider.vision.pose.core as core

    monkeypatch.setattr(core, "_video_fps", lambda p: 15.0)
    video = tmp_path / "v.mp4"
    video.touch()
    paths = np.array([[(60.0 + f, 60.0) for f in range(10)], [(200.0, 150.0) for _ in range(10)]])
    model = _Model([_mask(paths[:, f]) for f in range(10)])
    out = track_video_pups(
        Path("pups.pt"), video, litter_size=2, device="cpu", load_model=lambda p: model
    )
    assert out == tmp_path / "v_pups.npz"
    tracks = load_pups(out)
    assert (tracks.n_pups, tracks.n_frames) == (2, 10)
    assert tracks.meta["litter_size"] == 2 and tracks.meta["model"] == "pups.pt"
    assert model.kwargs["stream"] is True
