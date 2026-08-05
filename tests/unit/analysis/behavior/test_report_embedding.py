"""The report folder carries the embedding, so the Review tab need not load
a model bundle to draw it.

The embedding lives in the .pkl, which is hundreds of megabytes with the
classifier inside. Opening that to render a scatter would make browsing runs
cost as much as loading a model. The report folder is already the
self-contained record of a run; the coordinates belong there with the rest.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior.report import write_training_report
from glider.analysis.behavior.run_report import TrainingRun

EMBEDDING_FILE = "embedding.npz"


class _Result:
    """The parts of a TrainResult the report reads."""

    def __init__(self, model=None, summary=None):
        self.model = model
        self.summary = summary or {"split_strategy": "no_holdout", "classes": ["a", "b"]}


class _Model:
    def __init__(self, embedding=None):
        self.embedding = embedding


def _artifact(n=60):
    from glider.analysis.behavior.embedding import EmbeddingArtifact

    rng = np.random.default_rng(0)
    return EmbeddingArtifact(
        method="pca",
        scaler=None,
        reducer=None,
        coords=rng.normal(0, 1, (n, 3)),
        labels=np.array(["a"] * (n // 2) + ["b"] * (n - n // 2)),
        feature_names=["f1", "f2"],
    )


def test_the_embedding_is_written_beside_the_summary(tmp_path):
    out = write_training_report(_Result(model=_Model(_artifact())), tmp_path / "r")
    assert (out / EMBEDDING_FILE).exists()


def test_nothing_is_written_when_there_is_no_embedding(tmp_path):
    """Most runs have none; an empty file would look like a broken one."""
    out = write_training_report(_Result(model=_Model(None)), tmp_path / "r")
    assert not (out / EMBEDDING_FILE).exists()


def test_a_result_with_no_model_still_writes_a_report(tmp_path):
    """cross_validate_sessions returns metrics and no model at all."""
    out = write_training_report(_Result(model=None), tmp_path / "r")
    assert (out / "summary.json").exists()


def test_the_run_reads_the_coordinates_back(tmp_path):
    artifact = _artifact()
    out = write_training_report(_Result(model=_Model(artifact)), tmp_path / "r")

    run = TrainingRun.load(out)
    loaded = run.embedding
    assert loaded is not None
    np.testing.assert_allclose(loaded.coords, artifact.coords)
    assert list(loaded.labels) == list(artifact.labels)
    assert loaded.method == "pca"


def test_a_run_without_one_reports_none(tmp_path):
    out = write_training_report(_Result(model=_Model(None)), tmp_path / "r")
    assert TrainingRun.load(out).embedding is None


def test_a_summary_only_run_reports_none():
    """An in-memory run (measure-only CV) has no folder to read from."""
    assert TrainingRun.from_summary({"split_strategy": "cross_validated"}).embedding is None


def test_a_corrupt_embedding_file_does_not_break_the_run(tmp_path):
    """Browsing a run must survive a truncated write."""
    out = write_training_report(_Result(model=_Model(_artifact())), tmp_path / "r")
    (out / EMBEDDING_FILE).write_bytes(b"not an npz")

    run = TrainingRun.load(out)
    assert run.embedding is None
    assert run.summary  # the rest of the run still loads


def test_writing_an_embedding_never_costs_the_report(tmp_path, monkeypatch):
    """A chart is a bonus; the summary is the record."""

    class Exploding:
        @property
        def coords(self):
            raise RuntimeError("boom")

    out = write_training_report(_Result(model=_Model(Exploding())), tmp_path / "r")
    assert (out / "summary.json").exists()


@pytest.mark.parametrize("n", [1, 3])
def test_a_tiny_embedding_round_trips(tmp_path, n):
    artifact = _artifact(n=n)
    out = write_training_report(_Result(model=_Model(artifact)), tmp_path / "r")
    assert TrainingRun.load(out).embedding.coords.shape == (n, 3)
