"""Session Review's plugin panel hook (glider.review_panel entry points)."""

from __future__ import annotations

import pytest

from glider.gui.review.extensions import PanelHost, ReviewPanel, load_review_panels


class _Panel:
    def __init__(self, title="p", placement="side", fail=None):
        from PyQt6.QtWidgets import QLabel

        self.title, self.placement, self.fail = title, placement, fail
        self._w = QLabel(title)
        self.calls = []

    def widget(self):
        return self._w

    def set_session(self, context):
        if self.fail == "set_session":
            raise RuntimeError("boom")
        self.calls.append(("session", context))

    def set_frame(self, frame):
        if self.fail == "set_frame":
            raise RuntimeError("boom")
        self.calls.append(("frame", frame))


class _EP:
    def __init__(self, name, factory):
        self.name, self._factory = name, factory

    def load(self):
        return self._factory


def _raise():
    raise ImportError("missing dependency")


def test_loads_panels_and_skips_broken_or_unusable_plugins(qtbot):
    good = _Panel("a")
    eps = [
        _EP("b_list", lambda: [_Panel("t", "timeline"), _Panel("bad", "floating")]),
        _EP("a_single", lambda: good),
        _EP("c_broken", _raise),
        _EP("d_not_a_panel", lambda: object()),
    ]
    panels = load_review_panels(eps)
    assert [p.title for p in panels] == ["a", "t"]
    assert all(isinstance(p, ReviewPanel) for p in panels)


def test_host_switches_off_a_failing_panel_and_keeps_the_rest(qtbot):
    ok, bad = _Panel("ok"), _Panel("bad", fail="set_frame")
    host = PanelHost([bad, ok])
    host.set_session("ctx")
    host.set_frame(1)
    host.set_frame(2)
    assert ok.calls == [("session", "ctx"), ("frame", 1), ("frame", 2)]
    assert bad.calls == [("session", "ctx")]
    assert not bad.widget().isEnabled()


@pytest.fixture
def window_with_panels(qtbot, monkeypatch):
    side, timeline = _Panel("Side", "side"), _Panel("Trace", "timeline")
    monkeypatch.setattr("glider.gui.review.extensions.load_review_panels", lambda: [side, timeline])
    from glider.gui.behavior.analysis_window import AnalysisWindow

    window = AnalysisWindow()
    qtbot.addWidget(window)
    return window, side, timeline


def test_review_window_places_plugin_panels(window_with_panels):
    window, side, timeline = window_with_panels
    assert window._top_split.count() == 4
    assert window._top_split.widget(2) is side.widget()
    tabs = window._timeline_panel
    assert tabs.widget(tabs.count() - 1) is timeline.widget()  # after the built-in tabs
    assert tabs.indexOf(window._epoch) < tabs.count() - 1
    assert window._top_key() == "review/top4"
    window._set_frame(7)
    assert ("frame", 7) in side.calls and ("frame", 7) in timeline.calls
