"""The Apply tab's centred-vote control has to reach the scoring path.

Every link in this chain is a place the setting can be dropped silently, and
dropping it does not fail — it just scores worse. On eight held-out sessions
the difference was 0.780 vs 0.823 macro F1, which is the kind of gap nobody
notices in an ethogram they did not already have a number for.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PyQt6")

from glider.analysis.behavior.classify.smoothing import (  # noqa: E402
    DEFAULT_OFFLINE_WINDOW,
)
from glider.gui.behavior.workers import ApplyWorker  # noqa: E402


def test_the_worker_accepts_it():
    assert "offline_smooth_window" in inspect.signature(ApplyWorker.__init__).parameters


def test_the_worker_forwards_it_to_classify(monkeypatch, tmp_path):
    """The link that actually carries the setting into the pipeline."""
    seen = {}

    def fake_classify(video, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here — we only care what was passed")

    monkeypatch.setattr("glider.gui.behavior.workers.classify", fake_classify)
    w = ApplyWorker(tmp_path / "v.mp4", None, None, ["nose"], tmp_path, offline_smooth_window=25)
    w.run()  # swallows the RuntimeError into the failed signal

    assert seen.get("offline_smooth_window") == 25


def test_omitting_it_leaves_the_pipeline_default_alone():
    """None must mean 'do not pass', not 'pass zero' -- otherwise the worker
    silently overrides whatever the pipeline chose."""
    seen = {}

    def fake_classify(video, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    import glider.gui.behavior.workers as mod

    original = mod.classify
    mod.classify = fake_classify
    try:
        w = ApplyWorker(None, None, None, ["nose"], None)
        w.run()
    finally:
        mod.classify = original

    assert "offline_smooth_window" not in seen


def test_the_config_carries_it():
    from glider.analysis.behavior.classify.pipeline import LiveInferenceConfig

    assert "offline_smooth_window" in LiveInferenceConfig.__dataclass_fields__
    field = LiveInferenceConfig.__dataclass_fields__["offline_smooth_window"]
    assert field.default == 0, "the library default must stay off; the GUI opts in"


def test_the_apply_tab_defaults_it_on(qtbot, tmp_path):
    """Scoring a recording is the case where looking ahead is free, so the tab
    should not make people find the setting to get the better number."""
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    apply_tab = win.tabs.widget(3)

    spin = getattr(apply_tab, "_offline_smooth", None)
    assert spin is not None, "the Apply tab has no centred-vote control"
    assert spin.value() == DEFAULT_OFFLINE_WINDOW


def test_the_control_can_be_switched_off(qtbot, tmp_path):
    """0 has to be reachable and mean off, for anyone reproducing a live run."""
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    spin = win.tabs.widget(3)._offline_smooth

    assert spin.minimum() == 0
    spin.setValue(0)
    assert spin.value() == 0
    assert spin.specialValueText(), "0 should read as 'off', not as a bare zero"
