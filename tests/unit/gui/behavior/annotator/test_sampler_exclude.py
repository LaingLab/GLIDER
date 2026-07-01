"""Tests for the labelled-zone exclusion mask used by the clip sampler."""

from __future__ import annotations

import numpy as np

from glider.gui.behavior.annotator.sampler import _excluded_window_mask


def test_keeps_windows_far_from_zones():
    idx = np.array([0, 50, 500, 1000])
    keep = _excluded_window_mask(idx, n_frames=2000, exclude_zones=[(100, 110)], margin=5)
    assert keep.tolist() == [True, True, True, True]


def test_drops_windows_inside_zone_plus_margin():
    idx = np.array([90, 95, 105, 114, 116])
    # zone [100,110) dilated by margin 5 -> forbidden [95,115)
    keep = _excluded_window_mask(idx, n_frames=2000, exclude_zones=[(100, 110)], margin=5)
    assert keep.tolist() == [True, False, False, False, True]


def test_multiple_zones_union():
    idx = np.array([10, 30, 70, 90])
    keep = _excluded_window_mask(idx, n_frames=200, exclude_zones=[(20, 25), (80, 85)], margin=10)
    # forbidden: [10,35) and [70,95)
    assert keep.tolist() == [False, False, False, False]


def test_clamps_at_bounds():
    idx = np.array([0, 5, 195, 199])
    keep = _excluded_window_mask(idx, n_frames=200, exclude_zones=[(2, 4), (197, 199)], margin=10)
    # forbidden clamps to [0,14) and [187,200)
    assert keep.tolist() == [False, False, False, False]


def test_empty_zones_keeps_all():
    idx = np.array([1, 2, 3])
    keep = _excluded_window_mask(idx, n_frames=100, exclude_zones=[], margin=5)
    assert keep.all()
