"""The Train tab's frame-rate override.

``resolve_sessions_fps`` refuses a cohort whose pose CSVs disagree on rate and
tells the caller to "pass fps= to force one rate". Until this control existed
that advice named something the window could not do, so a cohort with mixed
sidecars was simply untrainable from the GUI.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PyQt6")


def test_frame_rate_is_automatic_by_default(qtbot):
    """Auto means "read it from the sidecars", which is `fps=None`, not 0 fps."""
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    assert tab._shared_options()["fps"] is None


def test_a_forced_rate_reaches_the_pipeline(qtbot):
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    tab._fps_spin.setValue(30.0)
    assert tab._shared_options()["fps"] == pytest.approx(30.0)


def test_zero_is_shown_as_auto_rather_than_a_rate(qtbot):
    """A literal 0 fps would divide by zero downstream; it has to read as off."""
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    assert tab._fps_spin.specialValueText()
    assert tab._fps_spin.minimum() == pytest.approx(0.0)


def test_both_entry_points_accept_the_option():
    """Fit and Cross-validate share _shared_options, so both must take fps."""
    from glider.analysis.behavior import (
        cross_validate_and_train,
        cross_validate_sessions,
        train_model,
    )

    for fn in (train_model, cross_validate_sessions, cross_validate_and_train):
        assert "fps" in inspect.signature(fn).parameters, fn.__name__
