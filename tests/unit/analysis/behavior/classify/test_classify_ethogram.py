import csv

import pandas as pd
import pytest

from glider.analysis.behavior.classify import ethogram_from_labels


def test_ethogram_from_labels_uses_glider_primitives():
    labels = ["rear", "rear", "groom", "groom", "groom", "rear"]
    result = ethogram_from_labels(labels, fps=30.0)

    # per-frame ethogram
    assert list(result.ethogram["name"]) == labels
    # intervals: 3 contiguous runs (rear x2-frames, groom x3, rear x1)
    assert len(result.intervals) == 3
    assert list(result.intervals["state"]) == ["rear", "groom", "rear"]
    # bouts: glider.analysis.ethogram.compute_bouts returns a dict[state -> Series
    # of bout durations] (NOT a DataFrame). Two rear runs, one groom run.
    assert set(result.bouts) == {"rear", "groom"}
    assert len(result.bouts["rear"]) == 2
    assert len(result.bouts["groom"]) == 1
    # transitions: count DataFrame (from_state, to_state, count)
    assert isinstance(result.transitions, pd.DataFrame)


class _FakePipeline:
    """Stand-in for LiveInferencePipeline: writes a canned ethogram CSV on
    run() so we can exercise classify()'s config mapping, CSV read-back,
    effective-fps math and output-file writing without cv2/yolo/torch."""

    _rows = [("rear", 0), ("rear", 1), ("groom", 2), ("groom", 3), ("groom", 4), ("rear", 5)]

    def __init__(self, config):
        self.config = config

        class _Producer:
            fps = 30.0

        self.producer = _Producer()

    def run(self):
        with self.config.ethogram_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "behavior"])
            for lab, fidx in self._rows:
                w.writerow([fidx, lab])


def _patch_pipeline(monkeypatch):
    import glider.analysis.behavior.classify as mod

    monkeypatch.setattr(mod, "LiveInferencePipeline", _FakePipeline)


def test_classify_maps_config_reads_csv_and_writes_outputs(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch)

    from glider.analysis.behavior.classify import classify

    out = tmp_path / "out"
    result = classify(
        video="clip.mp4",
        model_path="model.pkl",
        yolo_path="yolo.pt",
        keypoint_names=["nose", "tail"],
        output_dir=out,
        predict_every=1,  # effective_fps == producer.fps
    )

    # Labels recovered from the canned CSV feed the ethogram glue.
    assert list(result.ethogram["name"]) == ["rear", "rear", "groom", "groom", "groom", "rear"]
    assert list(result.intervals["state"]) == ["rear", "groom", "rear"]

    # Output CSVs land in output_dir.
    bouts = pd.read_csv(out / "bouts.csv")
    stats = pd.read_csv(out / "stats.csv")
    transitions = pd.read_csv(out / "transitions.csv")

    # bouts.csv: one row per bout (2 rear runs + 1 groom run).
    assert len(bouts) == 3
    assert set(bouts["state"]) == {"rear", "groom"}

    # stats.csv: durations in seconds. The groom run covers frames 2, 3 and 4
    # at 30 fps, so it occupies 3/30 s — a run of n frames is charged n frame
    # periods, not n-1. n_bouts counts runs, not frames.
    groom = stats.set_index("state").loc["groom"]
    assert groom["n_bouts"] == 1
    assert groom["total_s"] == pytest.approx(3 / 30.0)
    assert groom["mean_s"] == pytest.approx(3 / 30.0)

    # Every frame is accounted for: nothing falls between the bouts.
    assert stats["total_s"].sum() == pytest.approx(6 / 30.0)
    assert stats["fraction"].sum() == pytest.approx(1.0)

    assert set(transitions.columns) >= {"from_state", "to_state", "count"}


def test_classify_missing_ethogram_without_error_yields_empty_result(tmp_path, monkeypatch):
    import glider.analysis.behavior.classify as mod

    class _SilentPipeline(_FakePipeline):
        def run(self):  # never writes the ethogram CSV, records no error
            pass

    monkeypatch.setattr(mod, "LiveInferencePipeline", _SilentPipeline)

    out = tmp_path / "out"
    result = mod.classify(
        video="clip.mp4",
        model_path="model.pkl",
        yolo_path="yolo.pt",
        keypoint_names=["nose", "tail"],
        output_dir=out,
    )

    assert result.intervals.empty
    assert result.bouts == {}
    assert (out / "bouts.csv").exists()


def test_classify_missing_ethogram_with_producer_error_raises(tmp_path, monkeypatch):
    import glider.analysis.behavior.classify as mod

    class _FailingPipeline(_FakePipeline):
        def __init__(self, config):
            super().__init__(config)
            self.producer.error = "could not open video source"

        def run(self):  # fails to produce output and flags the error
            pass

    monkeypatch.setattr(mod, "LiveInferencePipeline", _FailingPipeline)

    with pytest.raises(RuntimeError, match="could not open video source"):
        mod.classify(
            video="bad.mp4",
            model_path="model.pkl",
            yolo_path="yolo.pt",
            keypoint_names=["nose", "tail"],
            output_dir=tmp_path / "out",
        )
