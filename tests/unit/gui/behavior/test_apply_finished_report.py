"""`_on_apply_finished` must report something true for a multi-animal run.

`classify()` returns `dict[int, Path]` for a multi-animal session instead of
an `EthogramResult` (see `glider/analysis/behavior/classify/__init__.py`).
Before this fix, `_on_apply_finished` only knew the `EthogramResult` shape:
`getattr(getattr(result, "ethogram", None), "__len__", lambda: None)()`
silently resolved to `None` for a dict, and every artifact it then looked for
(annotated.mp4, ethogram_raw.csv, bouts.csv, stats.csv, transitions.csv)
lives under `animals_dir` for a multi-animal run, not `output_dir` -- so the
operator got a log entry with nothing in it but the video's name.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PyQt6")

from glider.analysis.behavior.classify import EthogramResult  # noqa: E402
from glider.gui.behavior.window import ApplyTab  # noqa: E402


@pytest.fixture
def tab(qtbot):
    widget = ApplyTab()
    qtbot.addWidget(widget)
    return widget


class TestMultiAnimalResult:
    def test_reports_animal_count_and_ethogram_paths(self, tab, tmp_path):
        animal_dir = tmp_path / "clipDLC_yolo_animals"
        result = {
            0: animal_dir / "animal0_ethogram.csv",
            1: animal_dir / "animal1_ethogram.csv",
        }

        tab._on_apply_finished(result, tmp_path / "clip.mp4", tmp_path / "out")

        text = tab._results.toPlainText()
        assert "clip.mp4" in text
        assert "animals scored: 2" in text
        assert str(result[0]) in text
        assert str(result[1]) in text


class TestSingleAnimalResultUnchanged:
    """Regression guard: the dict branch must not touch how an
    EthogramResult is reported."""

    def test_same_lines_as_before(self, tab, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        for name in (
            "annotated.mp4",
            "ethogram_raw.csv",
            "bouts.csv",
            "stats.csv",
            "transitions.csv",
        ):
            (out / name).write_bytes(b"")

        result = EthogramResult(
            ethogram=pd.DataFrame(
                {"frame": [0, 1, 2], "time_ms": [0, 33, 66], "name": ["a", "a", "b"]}
            ),
            intervals=pd.DataFrame(),
            bouts={},
            transitions=pd.DataFrame(),
        )

        tab._on_apply_finished(result, tmp_path / "clip.mp4", out)

        text = tab._results.toPlainText()
        assert "clip.mp4:" in text
        assert "frames classified: 3" in text
        assert f"annotated video: {out / 'annotated.mp4'}" in text
        assert f"ethogram (raw): {out / 'ethogram_raw.csv'}" in text
        assert f"bouts: {out / 'bouts.csv'}" in text
        assert f"stats: {out / 'stats.csv'}" in text
        assert f"transitions: {out / 'transitions.csv'}" in text
        assert "animals scored" not in text
