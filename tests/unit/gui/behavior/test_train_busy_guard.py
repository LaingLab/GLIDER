"""Fit and Cross-validate share one thread slot; the second must not clobber it (F6).

Overwriting ``_train_thread`` dropped the only reference to a running parentless
QThread, which Qt answers by aborting the process.
"""

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")


@pytest.mark.parametrize("handler", ["_on_fit", "_on_cross_validate"])
def test_a_second_run_is_refused_while_one_is_live(qtbot, handler):
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    tab._sessions = [(Path("a.csv"), Path("a_ann.csv"))]
    tab._output_path = Path("model.pkl")
    live = object()
    tab._train_thread = live

    getattr(tab, handler)()

    assert tab._train_thread is live
    tab._train_thread = None


def test_fit_disables_cross_validate(qtbot, monkeypatch):
    from glider.gui.behavior import window

    tab = window.TrainTab()
    qtbot.addWidget(tab)
    tab._sessions = [(Path("a.csv"), Path("a_ann.csv"))]
    tab._output_path = Path("model.pkl")
    monkeypatch.setattr(window.QThread, "start", lambda self: None)

    tab._on_fit()

    assert not tab._cv_btn.isEnabled()
    tab._train_thread = None
