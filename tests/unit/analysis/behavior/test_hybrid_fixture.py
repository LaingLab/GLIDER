"""Smoke test for the shared ``hybrid_sessions`` fixture (conftest.py)."""

from __future__ import annotations

import pytest

pytest.importorskip("lightgbm")


def test_hybrid_sessions_fixture_trains(hybrid_sessions):
    """The fixture builds a trainable 2-session dataset whose classes cover
    both ``rest`` and ``locomote``."""
    from glider.analysis.behavior import train_model

    sessions, tag_map = hybrid_sessions
    assert set(tag_map) == {"rest", "locomote"}
    assert len(sessions) == 2

    result = train_model(sessions, fps=30.0, classifier_type="lightgbm", n_estimators=40)
    classes = set(result.summary["classes"])
    assert {"rest", "locomote"} <= classes
