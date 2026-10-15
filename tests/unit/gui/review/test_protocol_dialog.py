"""The protocol dialog: every field of the cohort's retrieval rules."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from glider.analysis.behavior.retrieval_setup import RetrievalProtocol  # noqa: E402
from glider.gui.review.protocol_dialog import ProtocolDialog  # noqa: E402


def test_the_dialog_edits_every_field(qtbot):
    d = ProtocolDialog(RetrievalProtocol(), ["0", "1", "nose"], Path("/data/cohort"))
    qtbot.addWidget(d)
    assert "/data/cohort" in d.windowTitle() or "cohort" in d.windowTitle()
    d.cutoff.setValue(600)
    d.nose.setCurrentText("0")
    d.contact.setValue(1.5)
    p = d.protocol()
    assert (p.cutoff_s, p.nose_keypoint, p.contact_cm) == (600.0, "0", 1.5)


def test_an_unknown_nose_name_is_still_offered(qtbot):
    d = ProtocolDialog(RetrievalProtocol(nose_keypoint="snout"), ["0", "1"], Path("/x"))
    qtbot.addWidget(d)
    assert d.nose.currentText() == "snout"


def test_a_blank_nose_keeps_the_old_one(qtbot):
    d = ProtocolDialog(RetrievalProtocol(nose_keypoint="snout"), ["0"], Path("/x"))
    qtbot.addWidget(d)
    d.nose.setCurrentText("  ")
    assert d.protocol().nose_keypoint == "snout"


def test_the_title_says_thresholds(qtbot):
    d = ProtocolDialog(RetrievalProtocol(), ["nose"], Path("/data/cohort"))
    qtbot.addWidget(d)
    assert d.windowTitle() == "Retrieval thresholds — cohort"
