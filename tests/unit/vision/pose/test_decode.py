"""Analytic tests for pose-network output decoding.

No models, no onnxruntime, no GPU: a Gaussian is planted at a known location in
a synthetic heatmap and the decoder has to find it. That makes the maths — the
part most likely to be quietly wrong — checkable to the pixel.

What these tests prove is that the decoders are *self-consistent* with the
conventions documented in decode.py. Whether those conventions match what a
real DeepLabCut or SLEAP export actually emits is a separate question, settled
only by test_parity.py against real fixtures.
"""

import numpy as np
import pytest

from glider.vision.pose.decode import decode_dlc_locref, decode_sleap_confmaps


def _blob(h, w, row, col, peak=1.0, sigma=1.0):
    """A Gaussian centred at (row, col) — sub-pixel centres allowed."""
    rr, cc = np.mgrid[0:h, 0:w]
    return peak * np.exp(-(((rr - row) ** 2 + (cc - col) ** 2) / (2 * sigma**2)))


def test_recovers_grid_centre_without_locref():
    heat = np.zeros((1, 8, 8))
    heat[0] = _blob(8, 8, 3, 5)
    xy, conf = decode_dlc_locref(heat, None, stride=8.0, locref_stdev=7.2831)
    # DLC's convention: a grid cell covers `stride` pixels and reports its centre.
    assert xy[0] == pytest.approx([5 * 8 + 4, 3 * 8 + 4])
    assert conf[0] == pytest.approx(1.0)


def test_locref_shifts_the_peak_by_offset_times_stdev():
    heat = np.zeros((1, 8, 8))
    heat[0] = _blob(8, 8, 3, 5)
    locref = np.zeros((2, 8, 8))
    locref[0, 3, 5] = 0.5  # x offset
    locref[1, 3, 5] = -0.25  # y offset
    xy, _ = decode_dlc_locref(heat, locref, stride=8.0, locref_stdev=8.0)
    assert xy[0] == pytest.approx([5 * 8 + 4 + 4.0, 3 * 8 + 4 - 2.0])


def test_each_keypoint_decodes_independently():
    heat = np.zeros((3, 8, 8))
    heat[0] = _blob(8, 8, 0, 0)
    heat[1] = _blob(8, 8, 7, 7)
    heat[2] = _blob(8, 8, 4, 2)
    xy, conf = decode_dlc_locref(heat, None, stride=4.0, locref_stdev=1.0)
    assert xy.shape == (3, 2)
    assert conf.shape == (3,)
    assert xy[0] == pytest.approx([2.0, 2.0])
    assert xy[1] == pytest.approx([7 * 4 + 2, 7 * 4 + 2])
    assert xy[2] == pytest.approx([2 * 4 + 2, 4 * 4 + 2])


def test_locref_is_read_per_keypoint_from_interleaved_channels():
    # Channel layout is x0,y0,x1,y1,... — a transposed read would swap these.
    heat = np.zeros((2, 4, 4))
    heat[0] = _blob(4, 4, 1, 1)
    heat[1] = _blob(4, 4, 2, 2)
    locref = np.zeros((4, 4, 4))
    locref[0, 1, 1] = 1.0  # kp0 x
    locref[3, 2, 2] = 1.0  # kp1 y
    xy, _ = decode_dlc_locref(heat, locref, stride=2.0, locref_stdev=3.0)
    assert xy[0] == pytest.approx([1 * 2 + 1 + 3.0, 1 * 2 + 1])
    assert xy[1] == pytest.approx([2 * 2 + 1, 2 * 2 + 1 + 3.0])


def test_sigmoid_is_applied_when_requested():
    heat = np.zeros((1, 4, 4))
    heat[0, 1, 1] = 0.0  # sigmoid(0) == 0.5
    _, conf = decode_dlc_locref(heat, None, stride=1.0, locref_stdev=1.0, apply_sigmoid=True)
    assert conf[0] == pytest.approx(0.5)


def test_flat_heatmap_still_returns_finite_values():
    heat = np.zeros((2, 4, 4))
    xy, conf = decode_dlc_locref(heat, None, stride=2.0, locref_stdev=1.0)
    assert np.isfinite(xy).all()
    assert conf == pytest.approx([0.0, 0.0])


def test_wrong_locref_channel_count_is_an_error():
    heat = np.zeros((2, 4, 4))
    bad = np.zeros((3, 4, 4))  # should be 2*K == 4
    with pytest.raises(ValueError, match="locref"):
        decode_dlc_locref(heat, bad, stride=1.0, locref_stdev=1.0)


def test_non_3d_heatmaps_are_an_error():
    with pytest.raises(ValueError, match="K, H, W"):
        decode_dlc_locref(np.zeros((4, 4)), None, stride=1.0, locref_stdev=1.0)


# --- SLEAP confidence maps --------------------------------------------------


def test_symmetric_blob_refines_to_its_centre():
    cm = np.zeros((1, 16, 16))
    cm[0] = _blob(16, 16, 8, 6, sigma=1.5)
    xy, conf = decode_sleap_confmaps(cm, stride=1.0, window=5)
    assert xy[0] == pytest.approx([6.0, 8.0], abs=1e-6)
    assert conf[0] == pytest.approx(1.0)


def test_offset_blob_refines_between_cells():
    # Peak sits between columns 6 and 7; refinement must land in between rather
    # than snapping to the argmax cell.
    cm = np.zeros((1, 16, 16))
    cm[0] = _blob(16, 16, 8, 6.5, sigma=1.5)
    xy, _ = decode_sleap_confmaps(cm, stride=1.0, window=5)
    assert 6.2 < xy[0][0] < 6.8
    assert xy[0][1] == pytest.approx(8.0, abs=1e-6)


def test_stride_scales_without_a_half_cell_shift():
    # SLEAP's grid convention differs from DLC's: cell i maps to i * stride,
    # with no + stride/2. Getting this wrong offsets every keypoint silently.
    cm = np.zeros((1, 8, 8))
    cm[0] = _blob(8, 8, 2, 3, sigma=0.4)
    xy, _ = decode_sleap_confmaps(cm, stride=4.0, window=3)
    assert xy[0] == pytest.approx([12.0, 8.0], abs=0.2)


def test_peak_on_the_border_does_not_index_out_of_bounds():
    cm = np.zeros((2, 6, 6))
    cm[0] = _blob(6, 6, 0, 0, sigma=0.6)
    cm[1] = _blob(6, 6, 5, 5, sigma=0.6)
    xy, conf = decode_sleap_confmaps(cm, stride=1.0, window=5)
    assert np.isfinite(xy).all()
    assert conf.shape == (2,)


def test_flat_confmap_falls_back_to_the_argmax_cell():
    cm = np.zeros((1, 5, 5))
    xy, conf = decode_sleap_confmaps(cm, stride=2.0, window=5)
    assert np.isfinite(xy).all()
    assert conf[0] == pytest.approx(0.0)


def test_negative_values_do_not_corrupt_the_centroid():
    cm = np.full((1, 7, 7), -1.0)
    cm[0, 3, 3] = 1.0
    xy, _ = decode_sleap_confmaps(cm, stride=1.0, window=5)
    assert xy[0] == pytest.approx([3.0, 3.0])


def test_each_channel_refines_independently():
    cm = np.zeros((2, 12, 12))
    cm[0] = _blob(12, 12, 3, 4, sigma=1.0)
    cm[1] = _blob(12, 12, 9, 7, sigma=1.0)
    xy, _ = decode_sleap_confmaps(cm, stride=1.0, window=5)
    assert xy[0] == pytest.approx([4.0, 3.0], abs=1e-6)
    assert xy[1] == pytest.approx([7.0, 9.0], abs=1e-6)


def test_sleap_non_3d_input_is_an_error():
    with pytest.raises(ValueError, match="K, H, W"):
        decode_sleap_confmaps(np.zeros((4, 4)), stride=1.0)
