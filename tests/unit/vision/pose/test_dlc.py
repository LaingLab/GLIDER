"""Round-trip tests for the DLC CSV converter."""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.pose import dlc as dlc_io


def test_dlc_csv_roundtrip(synthetic_pose, tmp_path):
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out)
    assert out.exists()

    restored = dlc_io.from_dlc_csv(out, fps=synthetic_pose.fps)
    np.testing.assert_allclose(restored.xy, synthetic_pose.xy, rtol=1e-5)
    np.testing.assert_allclose(restored.confidence, synthetic_pose.confidence, rtol=1e-5)
    assert restored.keypoint_names == synthetic_pose.keypoint_names
    assert restored.source == synthetic_pose.source


def test_dlc_csv_header_format(synthetic_pose, tmp_path):
    """Verify the 3 header rows match DLC convention."""
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out)
    with out.open() as f:
        lines = [next(f) for _ in range(4)]
    assert lines[0].startswith("scorer,")
    assert lines[1].startswith("bodyparts,")
    assert lines[2].startswith("coords,")
    # First data row starts with frame index "0"
    assert lines[3].startswith("0,")


def test_dlc_h5_roundtrip(synthetic_pose, tmp_path):
    pytest.importorskip("tables")
    out = tmp_path / "out.h5"
    dlc_io.to_dlc_h5(synthetic_pose, out)
    assert out.exists()


def test_dlc_roundtrip_preserves_nans(kpt_names, tmp_path):
    """NaNs (no-detection frames) must survive a round trip."""
    from glider.vision.pose.core import PoseData

    xy = np.full((10, len(kpt_names), 2), 100.0)
    xy[3] = np.nan
    cf = np.ones((10, len(kpt_names)))
    cf[3] = 0.0
    pose = PoseData(xy=xy, confidence=cf, keypoint_names=kpt_names)

    out = tmp_path / "nans.csv"
    dlc_io.to_dlc_csv(pose, out)
    restored = dlc_io.from_dlc_csv(out)
    assert np.isnan(restored.xy[3]).all()
    assert (restored.confidence[3] == 0.0).all()
