"""D2 spec Sec5.4: `classify()` must reach `write_animal_ethograms` for a
multi-animal session, and must NOT change for a single-animal one.

`find_pose_csv` returns None for a multi-animal session (D2 Sec2.1), so
`pose_csv_in` stays None and the apply flow used to fall through to
`LiveInferencePipeline` -- silently scoring nothing useful, since the
streaming pipeline is single-animal. `find_pose_csvs` returning more than
one path is the only signal that a multi-animal session exists; these tests
pin that `classify()` now acts on it.

Fixtures use two distinguishable animals (different seeds and arena
positions -> different features -> different scored labels), the same
technique `test_classify_tracks.py` uses: a slot swap would make the two
per-animal ethograms identical, which a same-animal fixture could not catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from glider.analysis.behavior.classify import classify  # noqa: E402
from glider.analysis.behavior.classify.batch import classify_pose_data  # noqa: E402
from glider.analysis.behavior.features import FeatureSpec, compute_features  # noqa: E402
from glider.analysis.behavior.model import BehaviorModel  # noqa: E402
from glider.analysis.behavior.windowing import apply_rolling  # noqa: E402
from glider.vision.pose.batch import animals_dir  # noqa: E402
from glider.vision.pose.core import PoseData  # noqa: E402
from glider.vision.pose.dlc import to_dlc_csv  # noqa: E402

KP = ["nose", "left_ear", "right_ear", "body_center", "tail_base"]
WINDOW = 4
STATS = ("mean", "std", "max")


def _pose(seed: int, base: tuple[float, float], n: int = 120) -> PoseData:
    """A wandering animal; seed and base position make two animals distinguishable."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    bx, by = base
    centre = np.stack([bx + 40 * np.sin(t / 9.0), by + 30 * np.cos(t / 7.0)], axis=1)
    offsets = np.array([[0, -14], [-8, -8], [8, -8], [0, 0], [0, 18]], dtype=float)
    xy = centre[:, None, :] + offsets[None] + rng.normal(0, 0.6, size=(n, len(KP), 2))
    conf = np.ones((n, len(KP)))
    return PoseData(xy=xy, confidence=conf, keypoint_names=KP, fps=30.0)


def _model(pose: PoseData, seed: int) -> BehaviorModel:
    from sklearn.tree import DecisionTreeClassifier

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


class _CnnSequenceModel:
    """Stands in for a sequence bundle: none of the tabular-model attributes
    the multi-animal path checks for, which is exactly how it declines one."""


@pytest.fixture(autouse=True)
def no_streaming(monkeypatch):
    """Neither branch under test may reach the streaming pipeline."""
    import glider.analysis.behavior.classify as mod

    def _explode(*_args, **_kwargs):
        raise AssertionError("the streaming pipeline should not have been used")

    monkeypatch.setattr(mod, "LiveInferencePipeline", _explode)


class TestMultiAnimalSession:
    def test_writes_one_ethogram_per_animal(self, tmp_path):
        pose0 = _pose(seed=100, base=(150.0, 120.0))
        pose1 = _pose(seed=101, base=(400.0, 300.0))
        model = _model(pose0, seed=102)

        animal_dir = tmp_path / "clipDLC_yolo_animals"
        animal_dir.mkdir()
        to_dlc_csv(pose0, animal_dir / "animal0.csv")
        to_dlc_csv(pose1, animal_dir / "animal1.csv")

        out = tmp_path / "out"
        result = classify(
            video="clip.mp4",
            model_path=None,
            yolo_path=None,
            keypoint_names=KP,
            output_dir=out,
            fps_override=30.0,
            reuse_existing_poses=True,
            pose_dir=tmp_path,
            model=model,
            predict_every=1,
        )

        # The door: write_animal_ethograms actually ran, one file per slot.
        assert set(result) == {0, 1}
        assert result[0] == animal_dir / "animal0_ethogram.csv"
        assert result[1] == animal_dir / "animal1_ethogram.csv"
        assert animals_dir(animal_dir.with_name("clipDLC_yolo.csv")) == animal_dir

        etho0 = pd.read_csv(result[0], keep_default_na=False)
        etho1 = pd.read_csv(result[1], keep_default_na=False)
        assert list(etho0.columns) == ["frame", "behavior"]

        # Fixtures distinguishable enough that a slot swap fails: parity
        # against classify_pose_data run directly on each animal's own pose,
        # not just "the two files differ".
        want0 = classify_pose_data(pose0, model, predict_every=1)
        want1 = classify_pose_data(pose1, model, predict_every=1)
        assert etho0["frame"].tolist() == want0.frames
        assert etho0["behavior"].tolist() == want0.labels
        assert etho1["frame"].tolist() == want1.frames
        assert etho1["behavior"].tolist() == want1.labels
        assert not etho0["behavior"].equals(etho1["behavior"])

        # No single-animal artifacts: this session was never routed through
        # the single-file dispatch (batch_apply / LiveInferencePipeline).
        assert not (out / "ethogram_raw.csv").exists()
        assert not (out / "bouts.csv").exists()

    def test_run_manifest_is_written_beside_each_animal_dir(self, tmp_path):
        """SessionView reads the run manifest from ethogram_csv.parent
        (session_view.py's _load_applied_thresholds / _load_scale), which
        for a multi-animal session is animal_dir -- not output_dir, where
        the manifest also lands. Without a copy in animal_dir, opening
        animal0_ethogram.csv in Session Review silently loses the applied
        freeze/dart thresholds, px_per_mm, and video association."""
        from glider.analysis.behavior.classify import read_run_manifest

        pose0 = _pose(seed=140, base=(150.0, 120.0))
        pose1 = _pose(seed=141, base=(400.0, 300.0))
        model = _model(pose0, seed=142)

        animal_dir = tmp_path / "clipDLC_yolo_animals"
        animal_dir.mkdir()
        to_dlc_csv(pose0, animal_dir / "animal0.csv")
        to_dlc_csv(pose1, animal_dir / "animal1.csv")

        out = tmp_path / "out"
        classify(
            video="clip.mp4",
            model_path=None,
            yolo_path=None,
            keypoint_names=KP,
            output_dir=out,
            fps_override=30.0,
            reuse_existing_poses=True,
            pose_dir=tmp_path,
            model=model,
            predict_every=1,
        )

        # The manifest still lands where it always has...
        at_output = read_run_manifest(out)
        assert at_output is not None
        # ...and now also where SessionView will actually look for it.
        at_animal_dir = read_run_manifest(animal_dir)
        assert at_animal_dir is not None
        assert at_animal_dir["video"] == "clip.mp4"
        assert at_animal_dir["fps"] == 30.0
        assert at_animal_dir == at_output

    def test_speed_only_multi_animal_is_refused_not_silently_wrong(self, tmp_path):
        """classify_pose_tracks has no speed-only mode (it always needs a
        model); a multi-animal run with neither must fail loudly rather than
        fall through to the single-animal streaming pipeline."""
        pose0 = _pose(seed=110, base=(150.0, 120.0))
        pose1 = _pose(seed=111, base=(400.0, 300.0))
        animal_dir = tmp_path / "clipDLC_yolo_animals"
        animal_dir.mkdir()
        to_dlc_csv(pose0, animal_dir / "animal0.csv")
        to_dlc_csv(pose1, animal_dir / "animal1.csv")

        with pytest.raises(NotImplementedError, match="speed-only"):
            classify(
                video="clip.mp4",
                model_path=None,
                yolo_path=None,
                keypoint_names=KP,
                output_dir=tmp_path / "out",
                fps_override=30.0,
                reuse_existing_poses=True,
                pose_dir=tmp_path,
                freeze_threshold=1.0,
                dart_threshold=50.0,
            )

    def test_cnn_model_multi_animal_is_refused_not_crashed(self, tmp_path):
        """classify_pose_tracks only knows the tabular scoring path
        (batch.classify_pose_data); there is no per-animal streaming
        fallback to decline into the way batch_apply's single-animal check
        has. Without this guard a CNN bundle reaches
        derive_stream_columns's `model.stats[0]` and dies with a bare
        AttributeError instead of a named refusal."""
        pose0 = _pose(seed=120, base=(150.0, 120.0))
        pose1 = _pose(seed=121, base=(400.0, 300.0))
        animal_dir = tmp_path / "clipDLC_yolo_animals"
        animal_dir.mkdir()
        to_dlc_csv(pose0, animal_dir / "animal0.csv")
        to_dlc_csv(pose1, animal_dir / "animal1.csv")

        with pytest.raises(NotImplementedError, match="CNN sequence model"):
            classify(
                video="clip.mp4",
                model_path=None,
                yolo_path=None,
                keypoint_names=KP,
                output_dir=tmp_path / "out",
                fps_override=30.0,
                reuse_existing_poses=True,
                pose_dir=tmp_path,
                model=_CnnSequenceModel(),
                predict_every=1,
            )

    def test_output_video_multi_animal_is_refused(self, tmp_path):
        """An annotated video is single-animal-only (LiveInferenceConfig /
        batch_apply territory); a multi-animal run must refuse rather than
        silently drop the request."""
        pose0 = _pose(seed=130, base=(150.0, 120.0))
        pose1 = _pose(seed=131, base=(400.0, 300.0))
        model = _model(pose0, seed=132)
        animal_dir = tmp_path / "clipDLC_yolo_animals"
        animal_dir.mkdir()
        to_dlc_csv(pose0, animal_dir / "animal0.csv")
        to_dlc_csv(pose1, animal_dir / "animal1.csv")

        with pytest.raises(ValueError, match="annotated video"):
            classify(
                video="clip.mp4",
                model_path=None,
                yolo_path=None,
                keypoint_names=KP,
                output_dir=tmp_path / "out",
                fps_override=30.0,
                reuse_existing_poses=True,
                pose_dir=tmp_path,
                model=model,
                predict_every=1,
                write_annotated=True,
            )


class TestSingleAnimalSessionUnchanged:
    """The regression guard: a session with exactly one pose CSV must be
    scored exactly as it was before this task -- same call, same file, same
    content -- even though `classify()` now also looks for `_animals/`."""

    def test_same_call_same_file_same_content(self, tmp_path):
        pose = _pose(seed=200, base=(180.0, 140.0))
        model = _model(pose, seed=201)
        pose_csv = tmp_path / "clipDLC_yolo.csv"
        to_dlc_csv(pose, pose_csv)

        out = tmp_path / "out"
        result = classify(
            video="clip.mp4",
            model_path=None,
            yolo_path=None,
            keypoint_names=KP,
            output_dir=out,
            fps_override=30.0,
            reuse_existing_poses=True,
            pose_dir=tmp_path,
            model=model,
            predict_every=1,
        )

        # Same file: the single-animal artifact, at the single-animal path.
        etho = pd.read_csv(out / "ethogram_raw.csv", keep_default_na=False)
        want = classify_pose_data(pose, model, predict_every=1)
        assert etho["frame"].tolist() == want.frames
        assert etho["behavior"].tolist() == want.labels

        # Same content: the return value is still one EthogramResult, not a
        # per-animal mapping.
        assert list(result.ethogram["name"]) == want.labels
        assert (out / "bouts.csv").exists()
        assert (out / "stats.csv").exists()
        assert (out / "transitions.csv").exists()

        # No multi-animal machinery was touched.
        assert not animals_dir(pose_csv).exists()
