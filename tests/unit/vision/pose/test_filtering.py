import numpy as np

from glider.vision.pose.filtering import (
    interpolate_gaps,
    mask_low_confidence,
    median_filter,
    smooth,
)


def test_mask_low_confidence(gappy_pose):
    masked = mask_low_confidence(gappy_pose, threshold=0.5)
    # Frames 10–13 had confidence 0.1 on keypoint 0
    assert np.all(np.isnan(masked.xy[10:14, 0]))
    # Other keypoints untouched
    assert not np.any(np.isnan(masked.xy[10:14, 1]))


def test_interpolate_gaps_fills_short_gaps(gappy_pose):
    masked = mask_low_confidence(gappy_pose, threshold=0.5)
    # Confirm the 4-frame gap is NaN before interpolation
    assert np.all(np.isnan(masked.xy[10:14, 0]))

    interp = interpolate_gaps(masked, max_gap=5)
    # Short gap (4 frames) gets filled
    assert not np.any(np.isnan(interp.xy[10:14, 0]))
    # Long gap (10 frames) does NOT
    assert np.all(np.isnan(interp.xy[50:60, 0]))


def test_interpolate_gaps_preserves_non_gappy(synthetic_pose):
    out = interpolate_gaps(synthetic_pose, max_gap=5)
    np.testing.assert_array_equal(out.xy, synthetic_pose.xy)


def test_median_filter_smooths_jitter(synthetic_pose):
    rng = np.random.default_rng(0)
    noisy = synthetic_pose.copy()
    noisy.xy = noisy.xy + rng.normal(0, 5.0, size=noisy.xy.shape)

    filtered = median_filter(noisy, window=5)
    # Filtered trace should be closer to the original than the noisy one.
    err_noisy = np.nanmean((noisy.xy - synthetic_pose.xy) ** 2)
    err_filt = np.nanmean((filtered.xy - synthetic_pose.xy) ** 2)
    assert err_filt < err_noisy


def test_median_filter_preserves_nans(kpt_names):
    from glider.vision.pose.core import PoseData

    xy = np.full((20, len(kpt_names), 2), 50.0)
    xy[5:8] = np.nan
    cf = np.ones((20, len(kpt_names)))
    cf[5:8] = 0.0
    pose = PoseData(xy=xy, confidence=cf, keypoint_names=kpt_names)

    out = median_filter(pose, window=3)
    assert np.all(np.isnan(out.xy[5:8]))


def test_smooth_pipeline(gappy_pose):
    out = smooth(
        gappy_pose,
        confidence_threshold=0.5,
        max_gap=5,
        median_window=3,
    )
    # Short gap filled, long gap still NaN.
    assert not np.any(np.isnan(out.xy[10:14, 0]))
    assert np.all(np.isnan(out.xy[52:58, 0]))
