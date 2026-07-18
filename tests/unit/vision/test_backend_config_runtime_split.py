"""CVProcessor keeps configuration and runtime state separate.

``_load_yolo_model`` used to degrade ``_settings.backend`` in place when a
backend couldn't be brought up, conflating "what the operator asked for" with
"what we managed to run". Once CV settings became persistent (schema 1.1.0)
that turned into silent data loss, by two routes:

  1. load -> save: a machine lacking the weights wrote the degradation back.
  2. the settings dialog: it is handed ``settings``, so it displayed the
     degraded backend and echoed it back on OK — laundering the runtime
     fallback into the configuration even if the operator changed nothing.

Route 1 was patched with a parallel ``configured_settings`` property; route 2
survived it, because the parallel record only helped at call sites that
remembered to use it. The fix is structural: ``settings`` is configuration and
is never written to from inside the class; ``active_backend`` is runtime.
"""

from __future__ import annotations

import pytest

from glider.vision.cv_processor import (
    BackendDegradation,
    CVProcessor,
    CVSettings,
    DetectionBackend,
)


@pytest.fixture
def absent_model(tmp_path) -> str:
    """A model path that cannot load, forcing a runtime degradation."""
    return str(tmp_path / "does_not_exist.pt")


@pytest.fixture
def degraded(absent_model) -> CVProcessor:
    """A processor asked for YOLO whose weights don't exist."""
    proc = CVProcessor()
    proc.update_settings(
        CVSettings(backend=DetectionBackend.YOLO_BYTETRACK, model_path=absent_model)
    )
    return proc


def test_failed_load_degrades_runtime_not_configuration(degraded):
    assert degraded.active_backend == DetectionBackend.BACKGROUND_SUBTRACTION
    assert degraded.settings.backend == DetectionBackend.YOLO_BYTETRACK


def test_degradation_is_reported(degraded):
    d = degraded.degradation
    assert isinstance(d, BackendDegradation)
    assert d.requested == DetectionBackend.YOLO_BYTETRACK
    assert d.active == DetectionBackend.BACKGROUND_SUBTRACTION
    assert d.reason  # non-empty, human-readable
    assert "YOLO_BYTETRACK" in str(d)


def test_no_degradation_when_backend_comes_up_normally():
    proc = CVProcessor()
    proc.update_settings(CVSettings(backend=DetectionBackend.BACKGROUND_SUBTRACTION))
    assert proc.degradation is None
    assert proc.active_backend == proc.settings.backend


def test_settings_dialog_round_trip_cannot_launder_a_degradation(degraded):
    """The hole that the configured_settings patch missed.

    main_window hands ``cv_processor.settings`` to CameraSettingsDialog and
    feeds whatever comes back into ``update_settings``. If ``settings``
    reflected the degradation, merely opening the dialog and pressing OK —
    changing nothing — would rewrite the operator's YOLO choice to background
    subtraction, and the next save would persist it.
    """
    shown_in_dialog = degraded.settings
    assert shown_in_dialog.backend == DetectionBackend.YOLO_BYTETRACK

    # The dialog echoes back what it was shown (operator changed nothing).
    returned_on_ok = CVSettings.from_dict(shown_in_dialog.to_dict())
    degraded.update_settings(returned_on_ok)

    assert degraded.settings.backend == DetectionBackend.YOLO_BYTETRACK
    assert degraded.active_backend == DetectionBackend.BACKGROUND_SUBTRACTION


def test_degradation_clears_when_the_condition_goes_away(degraded):
    """A later successful init must not leave a stale degradation behind."""
    assert degraded.degradation is not None
    degraded.update_settings(CVSettings(backend=DetectionBackend.MOTION_ONLY))
    assert degraded.degradation is None
    assert degraded.active_backend == DetectionBackend.MOTION_ONLY


def test_degradation_callback_fires(absent_model):
    seen: list[BackendDegradation] = []
    proc = CVProcessor()
    proc.on_backend_degraded(seen.append)
    proc.update_settings(CVSettings(backend=DetectionBackend.YOLO_V8, model_path=absent_model))
    assert len(seen) == 1
    assert seen[0].requested == DetectionBackend.YOLO_V8


def test_degradation_callback_not_fired_on_clean_init():
    seen: list[BackendDegradation] = []
    proc = CVProcessor()
    proc.on_backend_degraded(seen.append)
    proc.update_settings(CVSettings(backend=DetectionBackend.MOTION_ONLY))
    assert seen == []


def test_callback_may_reenter_the_processor(absent_model):
    """Callbacks fire outside the lock, so a listener can read state back.

    A status-bar listener naturally wants to ask "what are we running now?" —
    if we fired under self._lock that would deadlock.
    """
    observed: list[DetectionBackend] = []
    proc = CVProcessor()
    proc.on_backend_degraded(lambda _d: observed.append(proc.active_backend))
    proc.update_settings(CVSettings(backend=DetectionBackend.YOLO_V8, model_path=absent_model))
    assert observed == [DetectionBackend.BACKGROUND_SUBTRACTION]


def test_raising_callback_does_not_break_initialization(absent_model):
    proc = CVProcessor()
    proc.on_backend_degraded(lambda _d: 1 / 0)
    proc.update_settings(CVSettings(backend=DetectionBackend.YOLO_V8, model_path=absent_model))
    assert proc.active_backend == DetectionBackend.BACKGROUND_SUBTRACTION


def test_serialized_settings_carry_the_requested_backend(degraded):
    """What lands in the .glider file is the operator's choice."""
    assert degraded.settings.to_dict()["backend"] == "YOLO_BYTETRACK"


def test_repeated_degradation_reports_the_original_request(degraded):
    """Re-initializing must keep reporting what was configured, not the
    previous fallback — otherwise the reason drifts across reloads."""
    degraded.initialize()
    assert degraded.degradation.requested == DetectionBackend.YOLO_BYTETRACK
    assert degraded.settings.backend == DetectionBackend.YOLO_BYTETRACK
