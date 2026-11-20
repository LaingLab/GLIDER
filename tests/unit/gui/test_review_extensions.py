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


def test_hidden_panels_get_nothing_and_catch_up_when_shown(qtbot):
    panel = _Panel("p")
    host = PanelHost([panel])
    host.set_active(panel, False)
    host.set_session("ctx")
    host.set_frame(3)
    host.set_frame(4)
    assert panel.calls == []
    host.set_active(panel, True)
    assert panel.calls == [("session", "ctx"), ("frame", 4)]
    host.set_active(panel, True)  # already shown: no replay
    assert len(panel.calls) == 2


@pytest.fixture
def window_with_panels(qtbot, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings

    ini = str(tmp_path / "review.ini")
    monkeypatch.setattr(
        "glider.gui.behavior.analysis_window._settings",
        lambda: QSettings(ini, QSettings.Format.IniFormat),
    )
    side, timeline = _Panel("Side", "side"), _Panel("Trace", "timeline")
    monkeypatch.setattr("glider.gui.review.extensions.load_review_panels", lambda: [side, timeline])
    from glider.gui.behavior.analysis_window import AnalysisWindow

    def make():
        window = AnalysisWindow()
        qtbot.addWidget(window)
        return window

    return make, side, timeline


def test_review_window_places_plugin_panels_hidden(window_with_panels):
    make, side, timeline = window_with_panels
    window = make()
    window.show()
    assert window._top_split.count() == 4
    assert window._top_split.widget(2) is side.widget()
    tabs = window._timeline_panel
    index = tabs.indexOf(timeline.widget())
    assert index == tabs.count() - 1  # after the built-in tabs
    assert tabs.indexOf(window._epoch) < index
    assert window._top_key() == "review/top4"
    # Plugin panels start hidden and idle.
    assert not side.widget().isVisible() and not tabs.isTabVisible(index)
    window._set_frame(7)
    assert side.calls == [] and timeline.calls == []


def test_view_menu_shows_panels_and_remembers(window_with_panels):
    make, side, timeline = window_with_panels
    window = make()
    window.show()
    window._body.setCurrentIndex(1)  # the editor, as once a session is open
    actions = window._view_actions
    assert {"sessions", "inspector", "timeline", "plugin/Side", "plugin/Trace"} <= set(actions)
    # Plugin entries land in the top bar's View ▾, beside the window's own.
    in_view = window._view_btn.menu().actions()
    assert actions["plugin/Side"] in in_view and actions["plugin/Trace"] in in_view
    window._set_frame(5)
    actions["plugin/Side"].setChecked(True)
    actions["plugin/Trace"].setChecked(True)
    assert side.widget().isVisible() and side.calls[-1] == ("frame", 5)
    tabs = window._timeline_panel
    assert tabs.isTabVisible(tabs.indexOf(timeline.widget()))
    assert window._top_split.sizes()[2] > 0
    actions["inspector"].setChecked(False)
    assert not window._inspector.isVisible()

    window.close()
    again = make()
    again.show()
    again._body.setCurrentIndex(1)
    assert again._view_actions["plugin/Side"].isChecked()
    assert side.widget().isVisible() and not again._inspector.isVisible()
    assert again._top_split.sizes()[2] > 0
    again._view_actions["plugin/Trace"].setChecked(False)
    assert not again._timeline_panel.isTabVisible(again._timeline_panel.count() - 1)
