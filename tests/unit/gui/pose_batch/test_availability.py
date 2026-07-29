"""The [vision] dependency gate for the batch pose tracking tool."""

from __future__ import annotations

import pytest

from glider.gui.pose_batch import availability


@pytest.fixture(autouse=True)
def clear_cache():
    """The probe is memoized; each test needs a clean slate."""
    availability._CACHE = None
    yield
    availability._CACHE = None


def test_reports_ultralytics_missing(monkeypatch):
    monkeypatch.setattr(importlib_util(), "find_spec", lambda name: None)
    assert availability.missing_pose_batch_deps() == ["ultralytics"]
    assert availability.pose_batch_available() is False


def test_reports_nothing_missing_when_present(monkeypatch):
    monkeypatch.setattr(importlib_util(), "find_spec", lambda name: object())
    assert availability.missing_pose_batch_deps() == []
    assert availability.pose_batch_available() is True


def test_availability_is_memoized(monkeypatch):
    calls = {"n": 0}

    def counting(name):
        calls["n"] += 1
        return object()

    monkeypatch.setattr(importlib_util(), "find_spec", counting)
    availability.pose_batch_available()
    availability.pose_batch_available()
    assert calls["n"] == 1


def importlib_util():
    import importlib.util

    return importlib.util
