"""Annotator: close keeps the trim (F9); a failed save keeps the vocabulary (F10)."""

import pytest
from PyQt6.QtWidgets import QMessageBox

from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
from glider.analysis.behavior.vocabulary import Behavior, Vocabulary
from glider.gui.behavior.annotator.main_window import AnnotatorWindow
from glider.gui.behavior.annotator.sampler import ProposedClip

pytestmark = pytest.mark.usefixtures("qtbot")


def _window(tmp_path, zones):
    video = tmp_path / "a.mp4"
    csv = tmp_path / "a_annotations.csv"
    AnnotationStore(zones).save_csv(csv)
    vocab = Vocabulary()
    vocab.add(Behavior(name="groom", hotkey="1", color="#000"))
    vocab.add(Behavior(name="flank", hotkey="2", color="#111"))
    clips = [ProposedClip(0, 110, 100, 120, 0.5, str(video))]
    w = AnnotatorWindow(clips=clips, videos_meta={video: csv}, vocab=vocab)
    return w, csv


def test_close_persists_the_current_trim(tmp_path):
    w, csv = _window(tmp_path, [BehaviorZone("groom", 100, 120)])
    w.current = 0
    w.trim_bar.set_bounds(102, 118)

    w.close()

    zones = [(z.start_frame, z.end_frame) for z in AnnotationStore.load_csv(csv)]
    assert zones == [(102, 118)]


def test_merge_keeps_sources_in_vocab_when_a_save_fails(tmp_path, monkeypatch):
    w, _csv = _window(tmp_path, [BehaviorZone("flank", 100, 120)])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    def _fail(self, path):
        raise OSError("share dropped")

    monkeypatch.setattr(AnnotationStore, "save_csv", _fail)

    w._merge_behavior(["flank"], "groom")

    assert "flank" in w.vocab, "vocabulary dropped a name still on disk"
