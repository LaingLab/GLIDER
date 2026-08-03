"""The batch apply path must score a session exactly as the streaming one does.

The streaming pipeline is the reference: it is what every existing ethogram
was produced with. A faster path that quietly scores differently would be
worse than a slow one, so the central test here drives the same keypoints
through both and demands identical rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from glider.analysis.behavior.classify.batch import (  # noqa: E402
    batch_apply,
    classify_pose_data,
    write_ethogram_csv,
)
from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer  # noqa: E402
from glider.analysis.behavior.classify.features_stream import (  # noqa: E402
    StreamingFeatureExtractor,
    derive_stream_columns,
)
from glider.analysis.behavior.features import FeatureSpec, compute_features  # noqa: E402
from glider.analysis.behavior.model import BehaviorModel  # noqa: E402
from glider.vision.pose.core import PoseData  # noqa: E402

KP = ["nose", "left_ear", "right_ear", "body_center", "tail_base"]
WINDOW = 4
STATS = ("mean", "std", "max")


def _pose(n=120, seed=0, nan_frames=()):
    """A wandering animal, optionally with dropout frames."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    centre = np.stack([200 + 40 * np.sin(t / 9.0), 150 + 30 * np.cos(t / 7.0)], axis=1)
    offsets = np.array([[0, -14], [-8, -8], [8, -8], [0, 0], [0, 18]], dtype=float)
    xy = centre[:, None, :] + offsets[None] + rng.normal(0, 0.6, size=(n, len(KP), 2))
    for f in nan_frames:
        xy[f] = np.nan
    conf = np.where(np.isnan(xy).any(axis=-1), 0.0, 1.0)
    return PoseData(xy=xy, confidence=conf, keypoint_names=KP, fps=30.0)


def _model(pose, seed=0):
    """A real fitted BehaviorModel over this pose's own feature space."""
    from sklearn.tree import DecisionTreeClassifier

    from glider.analysis.behavior.windowing import apply_rolling

    spec = FeatureSpec()
    feats = compute_features(pose, spec)
    rolled = apply_rolling(feats, window=WINDOW, stats=STATS, min_periods=1)
    rolled = rolled.dropna()
    rng = np.random.default_rng(seed)
    y = rng.choice(["groom", "locomote", "rear"], size=len(rolled))
    clf = DecisionTreeClassifier(random_state=0, max_depth=6).fit(rolled, y)
    return BehaviorModel(
        classifier=clf,
        feature_names=list(rolled.columns),
        spec=spec,
        window=WINDOW,
        stats=STATS,
        fps=30.0,
        classes=sorted(set(y)),
    )


def _stream_reference(pose, model, *, predict_every=1, smooth_window=1):
    """What the streaming FeatureEngine + BehaviorClassifier would emit.

    Deliberately a re-statement of those two threads' logic rather than a call
    into them: the threads need queues and a live source, and what is being
    pinned here is the arithmetic, not the plumbing.
    """
    from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother

    per_frame, _ = derive_stream_columns(model)
    extractor = StreamingFeatureExtractor(spec=model.spec, keypoint_names=KP, fps=30.0)
    buffer = SlidingFeatureBuffer(per_frame, window=model.window, stats=model.stats)
    smoother = MajorityVoteSmoother(window=smooth_window)
    lag = extractor.lag
    frames, labels = [], []
    for frame_idx in range(pose.n_frames):
        feats = extractor.push(pose.xy[frame_idx])
        if feats is not None:
            buffer.push_features(feats)
        if (frame_idx + 1) % predict_every != 0 or len(buffer) == 0:
            continue
        names, row = buffer.rolling_features()
        pos = {n: i for i, n in enumerate(names)}
        aligned = np.array(
            [row[pos[n]] if n in pos else np.nan for n in model.feature_names], dtype=float
        )
        frames.append(max(0, frame_idx - lag))
        labels.append(smoother.push(model.predict_one(aligned)))
    return frames, labels


class TestParityWithTheStreamingPath:
    def test_frames_and_labels_match_exactly(self):
        pose = _pose()
        model = _model(pose)
        want_frames, want_labels = _stream_reference(pose, model)
        got = classify_pose_data(pose, model, predict_every=1)
        assert got.frames == want_frames
        assert got.labels == want_labels

    @pytest.mark.parametrize("every", [2, 3, 5])
    def test_parity_holds_at_every_cadence(self, every):
        pose = _pose(seed=1)
        model = _model(pose, seed=1)
        want_frames, want_labels = _stream_reference(pose, model, predict_every=every)
        got = classify_pose_data(pose, model, predict_every=every)
        assert got.frames == want_frames
        assert got.labels == want_labels

    def test_parity_holds_with_smoothing(self):
        pose = _pose(seed=2)
        model = _model(pose, seed=2)
        want_frames, want_labels = _stream_reference(pose, model, smooth_window=5)
        got = classify_pose_data(pose, model, predict_every=1, smooth_window=5)
        assert got.labels == want_labels

    def test_the_first_row_describes_the_first_centered_frame(self):
        """The streaming extractor warms up for 5 frames and emits the middle
        one, so nothing before frame 2 can be scored."""
        pose = _pose()
        got = classify_pose_data(pose, _model(pose), predict_every=1)
        assert got.frames[0] == 2


class TestScoring:
    def test_dropout_frames_score_blank_rather_than_guess(self):
        pose = _pose(nan_frames=(40, 41, 42))
        got = classify_pose_data(pose, _model(_pose()), predict_every=1)
        blanks = {f for f, lab in zip(got.frames, got.labels, strict=True) if lab == ""}
        # The dropout propagates through the rolling window, but the frames
        # themselves must certainly be unscored.
        assert {40, 41, 42} <= blanks

    def test_every_scored_label_is_one_the_model_knows(self):
        pose = _pose()
        model = _model(pose)
        got = classify_pose_data(pose, model, predict_every=1)
        assert set(got.labels) <= set(model.classes) | {""}

    def test_a_cadence_of_n_emits_a_row_every_n_frames(self):
        pose = _pose()
        got = classify_pose_data(pose, _model(pose), predict_every=4)
        gaps = np.diff(got.frames)
        assert set(gaps.tolist()) == {4}

    def test_the_speed_axis_lines_up_with_the_frame_it_labels(self):
        """The speed column describes the same frame as the behavior column."""
        from glider.analysis.behavior.classify.speed_state import CausalSpeed

        pose = _pose()
        got = classify_pose_data(
            pose, _model(pose), predict_every=1, freeze_threshold=0.5, dart_threshold=8.0
        )
        causal = CausalSpeed()
        expected = [float(causal.push(f)) for f in pose.xy]
        for frame, value in zip(got.frames, got.speed_px, strict=True):
            assert value == pytest.approx(expected[frame], nan_ok=True)

    def test_without_thresholds_there_is_no_speed_axis(self):
        pose = _pose()
        got = classify_pose_data(pose, _model(pose), predict_every=1)
        assert got.speed_px == []


class TestWhenBatchDeclines:
    """It must hand back to the streaming path, never score a partial model."""

    class _Config:
        pose_csv_in = "poses.csv"
        output_video = None
        predict_every = 1
        behavior_confidence_threshold = 0.0
        behavior_class_thresholds = None
        smooth_window = 1
        freeze_threshold = None
        dart_threshold = None
        freeze_min_frames = 30
        dart_min_frames = 3
        cm_s_per_px_frame = None
        fps_override = None

    def test_no_pose_csv_means_no_batch(self, tmp_path):
        config = self._Config()
        config.pose_csv_in = None
        assert batch_apply(config, tmp_path / "e.csv", _model(_pose())) is False

    def test_an_annotated_video_needs_the_streaming_path(self, tmp_path):
        config = self._Config()
        config.output_video = tmp_path / "annotated.mp4"
        assert batch_apply(config, tmp_path / "e.csv", _model(_pose())) is False

    def test_a_sequence_model_needs_the_streaming_path(self, tmp_path):
        class Sequenceish:
            classes = ["a", "b"]

        assert batch_apply(self._Config(), tmp_path / "e.csv", Sequenceish()) is False

    def test_spectral_features_are_refused_not_silently_dropped(self):
        """__domfreq columns come from a rolling FFT this path does not do."""
        pose = _pose()
        model = _model(pose)
        model.feature_names = [*model.feature_names, "speed_nose__domfreq"]
        with pytest.raises(NotImplementedError, match="spectral"):
            classify_pose_data(pose, model, predict_every=1)


class TestEthogramFile:
    def _rows(self, pose, model, **kw):
        return classify_pose_data(pose, model, predict_every=1, **kw)

    def test_the_plain_layout_matches_the_streaming_writer(self, tmp_path):
        pose = _pose()
        rows = self._rows(pose, _model(pose))
        out = tmp_path / "ethogram_raw.csv"
        write_ethogram_csv(out, rows, speed_axis=False)
        df = pd.read_csv(out, keep_default_na=False)
        assert list(df.columns) == ["frame", "behavior"]
        assert len(df) == len(rows.frames)

    def test_the_speed_layout_carries_both_units(self, tmp_path):
        pose = _pose()
        rows = self._rows(pose, _model(pose), freeze_threshold=0.5, dart_threshold=8.0)
        out = tmp_path / "ethogram_raw.csv"
        write_ethogram_csv(out, rows, speed_axis=True, cm_s_per_px_frame=2.0)
        df = pd.read_csv(out, keep_default_na=False)
        assert list(df.columns) == [
            "frame",
            "behavior",
            "behavior_postural",
            "speed",
            "speed_px_frame",
            "speed_cm_s",
        ]
        scored = df[df["speed_px_frame"] != ""]
        assert len(scored) > 0
        assert scored["speed_cm_s"].astype(float).tolist() == pytest.approx(
            (scored["speed_px_frame"].astype(float) * 2.0).tolist(), abs=1e-3
        )

    def test_without_a_pixel_scale_the_cm_column_stays_blank(self, tmp_path):
        """A guessed number in real units is worse than none."""
        pose = _pose()
        rows = self._rows(pose, _model(pose), freeze_threshold=0.5, dart_threshold=8.0)
        out = tmp_path / "ethogram_raw.csv"
        write_ethogram_csv(out, rows, speed_axis=True, cm_s_per_px_frame=None)
        df = pd.read_csv(out, keep_default_na=False)
        assert (df["speed_cm_s"] == "").all()


class TestAnalysingOnlyAWindow:
    """Scoring minutes 2-7 rather than the whole recording.

    The point of care: everything upstream of the prediction still runs over
    the whole session. Trimming the poses first would give the window's
    opening frames partial rolling windows and a cold causal-speed filter, so
    the very frames the operator chose to look at would be the least
    comparable ones in the cohort.
    """

    def test_only_the_window_is_returned(self):
        pose = _pose(n=300)
        model = _model(pose)
        rows = classify_pose_data(pose, model, predict_every=1, frame_range=(100, 199))
        assert rows.frames[0] == 100
        assert rows.frames[-1] == 199
        assert len(rows.frames) == 100

    def test_the_labels_match_a_whole_session_run(self):
        """A windowed run must be a subset of the full run, not a rescoring."""
        pose = _pose(n=300, seed=4)
        model = _model(pose, seed=4)
        full = classify_pose_data(pose, model, predict_every=1)
        window = classify_pose_data(pose, model, predict_every=1, frame_range=(100, 199))

        wanted = dict(zip(full.frames, full.labels, strict=True))
        for frame, label in zip(window.frames, window.labels, strict=True):
            assert label == wanted[frame]

    def test_frames_keep_their_absolute_numbering(self):
        """So the ethogram still lines up with the video and the poses."""
        pose = _pose(n=300)
        rows = classify_pose_data(pose, _model(pose), predict_every=1, frame_range=(100, 199))
        assert rows.frames == list(range(100, 200))

    def test_the_cadence_is_preserved_inside_the_window(self):
        pose = _pose(n=300)
        rows = classify_pose_data(pose, _model(pose), predict_every=4, frame_range=(100, 199))
        assert set(np.diff(rows.frames).tolist()) == {4}

    def test_the_speed_axis_is_warmed_up_not_restarted(self):
        """CausalSpeed smooths over the preceding frames; a window that
        restarted it would report a different speed for the same frame."""
        from glider.analysis.behavior.classify.speed_state import CausalSpeed

        pose = _pose(n=300)
        rows = classify_pose_data(
            pose,
            _model(pose),
            predict_every=1,
            freeze_threshold=0.5,
            dart_threshold=8.0,
            frame_range=(100, 199),
        )
        causal = CausalSpeed()
        expected = [float(causal.push(f)) for f in pose.xy]  # from frame 0
        for frame, value in zip(rows.frames, rows.speed_px, strict=True):
            assert value == pytest.approx(expected[frame], nan_ok=True)

    def test_a_window_past_the_end_yields_nothing(self):
        pose = _pose(n=300)
        rows = classify_pose_data(pose, _model(pose), predict_every=1, frame_range=(9000, 9999))
        assert rows.frames == []

    def test_no_range_scores_everything(self):
        pose = _pose(n=300)
        rows = classify_pose_data(pose, _model(pose), predict_every=1)
        assert len(rows.frames) > 250


class TestTheSpeedAxisOverridesPosture:
    """An animal cannot be darting and digging at the same instant.

    The two scorings are computed independently — one from posture, one from a
    thresholded speed trace — and where they disagree the speed axis is the
    stronger claim: a direct measurement of the body moving, against a
    classifier's guess about what the body is doing.
    """

    def _rows(self, **kw):
        pose = _pose(n=200, seed=7)
        return classify_pose_data(pose, _model(pose, seed=7), predict_every=1, **kw)

    def test_a_speed_label_wins_where_it_fired(self):
        rows = self._rows(freeze_threshold=1e9, dart_threshold=1e9 + 1)  # freeze everywhere
        assert set(rows.labels) == {"freezing"}

    def test_posture_survives_where_the_speed_axis_is_silent(self):
        rows = self._rows(freeze_threshold=-1.0, dart_threshold=1e18)  # never fires
        assert "freezing" not in rows.labels
        assert "darting" not in rows.labels
        assert rows.labels == rows.postural_labels

    def test_the_classifier_label_is_kept_beside_it(self):
        """The override is a reading of the data; a reading must be reversible."""
        rows = self._rows(freeze_threshold=1e9, dart_threshold=1e9 + 1)
        assert set(rows.labels) == {"freezing"}
        assert "freezing" not in rows.postural_labels

    def test_without_a_speed_axis_nothing_is_overridden(self):
        rows = self._rows()
        assert rows.speed_labels == []
        assert rows.labels == rows.postural_labels

    def test_resolve_is_positionwise(self):
        from glider.analysis.behavior.classify.batch import resolve_labels

        assert resolve_labels(["a", "b", "c"], ["", "freezing", ""]) == ["a", "freezing", "c"]

    def test_resolve_without_a_speed_axis_is_a_passthrough(self):
        from glider.analysis.behavior.classify.batch import resolve_labels

        assert resolve_labels(["a", "b"], []) == ["a", "b"]
