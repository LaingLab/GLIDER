import numpy as np
import pytest

from glider.vision.pose import PoseData, pose_from_array


def test_posedata_validates_shape(kpt_names):
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((10, 4, 2)),  # 4 keypoints, but 5 names
            confidence=np.zeros((10, 4)),
            keypoint_names=kpt_names,
        )


def test_posedata_validates_confidence_shape(kpt_names):
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((10, 5, 2)),
            confidence=np.zeros((10, 4)),  # mismatched
            keypoint_names=kpt_names,
        )


def test_posedata_rejects_duplicate_names():
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((3, 2, 2)),
            confidence=np.zeros((3, 2)),
            keypoint_names=["a", "a"],
        )


def test_posedata_rejects_bad_xy_dim(kpt_names):
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((10, 5)),  # 2D not 3D
            confidence=np.zeros((10, 5)),
            keypoint_names=kpt_names,
        )


def test_posedata_rejects_bad_fps(kpt_names):
    with pytest.raises(ValueError):
        PoseData(
            xy=np.zeros((10, 5, 2)),
            confidence=np.zeros((10, 5)),
            keypoint_names=kpt_names,
            fps=0.0,
        )


def test_posedata_properties(synthetic_pose):
    assert synthetic_pose.n_frames == 200
    assert synthetic_pose.n_keypoints == 5


def test_posedata_kp_index(synthetic_pose):
    assert synthetic_pose.kp_index("snout") == 0
    with pytest.raises(KeyError):
        synthetic_pose.kp_index("nonexistent")


def test_posedata_slice(synthetic_pose):
    sliced = synthetic_pose.slice_frames(50, 100)
    assert sliced.n_frames == 50
    np.testing.assert_array_equal(sliced.xy, synthetic_pose.xy[50:100])


def test_posedata_copy_is_deep(synthetic_pose):
    cp = synthetic_pose.copy()
    cp.xy[0, 0, 0] = 9999
    assert synthetic_pose.xy[0, 0, 0] != 9999


def test_pose_from_array(kpt_names):
    xy = np.zeros((5, len(kpt_names), 2))
    cf = np.ones((5, len(kpt_names)))
    p = pose_from_array(xy, cf, kpt_names, fps=60.0, source="test")
    assert p.fps == 60.0
    assert p.source == "test"
