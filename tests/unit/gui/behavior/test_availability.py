"""Tests for the behavior-stack availability helper."""

from __future__ import annotations

import importlib.util


def test_missing_behavior_deps_reports_absent(monkeypatch):
    from glider.gui.behavior import availability

    real = importlib.util.find_spec

    def fake(name, *a, **k):
        return None if name in {"umap", "hdbscan", "sklearn"} else real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    availability._CACHE = None  # reset memoization
    assert availability.behavior_available() is False
    assert set(availability.missing_behavior_deps()) >= {
        "umap-learn",
        "hdbscan",
        "scikit-learn",
    }


def test_all_present_reports_available(monkeypatch):
    from glider.gui.behavior import availability

    # Pretend every dependency resolves.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    availability._CACHE = None
    assert availability.behavior_available() is True
    assert availability.missing_behavior_deps() == []


def test_behavior_available_is_memoized(monkeypatch):
    from glider.gui.behavior import availability

    calls = {"n": 0}
    real = importlib.util.find_spec

    def counting(name, *a, **k):
        calls["n"] += 1
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", counting)
    availability._CACHE = None
    availability.behavior_available()
    first = calls["n"]
    availability.behavior_available()  # second call must not re-probe
    assert calls["n"] == first
