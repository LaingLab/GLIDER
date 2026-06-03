"""Unit tests for ``glider.vision.yolo_install``.

Three layers match the module's structure:

* Pure detection (``is_ultralytics_available``, ``can_auto_install``).
* The blocking pip-install subprocess (``install_ultralytics_blocking``)
  with :mod:`subprocess.run` monkeypatched.
* The Qt orchestrator (``ensure_ultralytics_installed``) exercised via
  ``pytest-qt`` with the prompt functions stubbed.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from glider.vision import yolo_install

# --- Pure detection ----------------------------------------------------------


class TestIsUltralyticsAvailable:
    def test_cache_is_respected(self, monkeypatch) -> None:
        yolo_install._invalidate_cache()
        calls = [0]

        def fake_find_spec(_name: str):
            calls[0] += 1
            return None  # not available

        monkeypatch.setattr(yolo_install.importlib.util, "find_spec", fake_find_spec)
        assert yolo_install.is_ultralytics_available() is False
        assert yolo_install.is_ultralytics_available() is False
        assert calls[0] == 1, "second call should hit the cache"

    def test_use_cache_false_forces_recheck(self, monkeypatch) -> None:
        yolo_install._invalidate_cache()
        calls = [0]

        def fake_find_spec(_name: str):
            calls[0] += 1
            return object()  # truthy — available

        monkeypatch.setattr(yolo_install.importlib.util, "find_spec", fake_find_spec)
        assert yolo_install.is_ultralytics_available() is True
        assert yolo_install.is_ultralytics_available(use_cache=False) is True
        assert calls[0] == 2


class TestCanAutoInstall:
    def test_source_install_returns_true(self, monkeypatch) -> None:
        # Strip sys.frozen if the runner happens to have set it.
        monkeypatch.delattr(yolo_install.sys, "frozen", raising=False)
        assert yolo_install.can_auto_install() is True

    def test_frozen_returns_false(self, monkeypatch) -> None:
        monkeypatch.setattr(yolo_install.sys, "frozen", True, raising=False)
        assert yolo_install.can_auto_install() is False


# --- Subprocess install ------------------------------------------------------


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    p = MagicMock(spec=subprocess.CompletedProcess)
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


class TestInstallBlocking:
    def test_success_invalidates_cache(self, monkeypatch) -> None:
        # Pre-seed the cache as False; a successful install should force a
        # re-check on the next is_ultralytics_available call.
        yolo_install._cached_available = False
        monkeypatch.setattr(
            yolo_install.subprocess,
            "run",
            lambda *a, **k: _completed(0, "installed\n", ""),
        )
        result = yolo_install.install_ultralytics_blocking()
        assert result.success is True
        assert yolo_install._cached_available is None

    def test_nonzero_exit_is_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(
            yolo_install.subprocess,
            "run",
            lambda *a, **k: _completed(1, "", "ERROR: no wheels\n"),
        )
        result = yolo_install.install_ultralytics_blocking()
        assert result.success is False
        assert "no wheels" in result.message

    def test_timeout_is_failure(self, monkeypatch) -> None:
        def raise_timeout(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="pip", timeout=1)

        monkeypatch.setattr(yolo_install.subprocess, "run", raise_timeout)
        result = yolo_install.install_ultralytics_blocking()
        assert result.success is False
        assert "timed out" in result.message.lower()

    def test_file_not_found_is_failure(self, monkeypatch) -> None:
        def raise_fnf(*_a, **_k):
            raise FileNotFoundError("no python")

        monkeypatch.setattr(yolo_install.subprocess, "run", raise_fnf)
        result = yolo_install.install_ultralytics_blocking()
        assert result.success is False
        assert "pip" in result.message.lower()


# --- Qt orchestrator ---------------------------------------------------------


class TestEnsureInstalled:
    def test_already_available_short_circuits(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr(yolo_install, "is_ultralytics_available", lambda **_: True)
        called = []
        monkeypatch.setattr(yolo_install, "_confirm_agpl", lambda _p: called.append("confirm"))
        assert yolo_install.ensure_ultralytics_installed(None) is True
        assert called == []

    def test_frozen_build_shows_message_and_returns_false(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr(yolo_install, "is_ultralytics_available", lambda **_: False)
        monkeypatch.setattr(yolo_install, "can_auto_install", lambda: False)
        shown = []
        monkeypatch.setattr(
            yolo_install,
            "_show_frozen_build_message",
            lambda _p: shown.append("frozen"),
        )
        monkeypatch.setattr(yolo_install, "_confirm_agpl", lambda _p: shown.append("agpl") or True)

        assert yolo_install.ensure_ultralytics_installed(None) is False
        assert shown == ["frozen"], "should not reach AGPL confirm in frozen builds"

    def test_user_declines_consent(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr(yolo_install, "is_ultralytics_available", lambda **_: False)
        monkeypatch.setattr(yolo_install, "can_auto_install", lambda: True)
        monkeypatch.setattr(yolo_install, "_confirm_agpl", lambda _p: False)

        # Install should never be attempted if consent is declined.
        ran = []
        monkeypatch.setattr(
            yolo_install,
            "install_ultralytics_blocking",
            lambda **_: ran.append("ran") or yolo_install.InstallResult(True, ""),
        )
        assert yolo_install.ensure_ultralytics_installed(None) is False
        assert ran == []

    def test_successful_install_returns_true(self, qapp, monkeypatch) -> None:
        # First availability check returns False (forces install); the
        # re-check after install returns True.
        calls = iter([False, True])
        monkeypatch.setattr(
            yolo_install,
            "is_ultralytics_available",
            lambda **_: next(calls),
        )
        monkeypatch.setattr(yolo_install, "can_auto_install", lambda: True)
        monkeypatch.setattr(yolo_install, "_confirm_agpl", lambda _p: True)

        # Short-circuit the QThread/dialog dance: fabricate a dialog whose
        # result reports success, and a worker that does nothing.
        class _FakeDialog:
            def __init__(self, _parent):
                self._r = yolo_install.InstallResult(True, "ok")

            def set_result(self, r):
                self._r = r

            def result_info(self):
                return self._r

            def exec(self):
                return None

        class _FakeWorker:
            def __init__(self, _parent):
                pass

            completed = MagicMock()

            def start(self):
                pass

            def deleteLater(self):
                pass

        monkeypatch.setattr(yolo_install, "_InstallProgressDialog", _FakeDialog)
        monkeypatch.setattr(yolo_install, "_InstallWorker", _FakeWorker)

        assert yolo_install.ensure_ultralytics_installed(None) is True

    def test_failed_install_shows_warning(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr(yolo_install, "is_ultralytics_available", lambda **_: False)
        monkeypatch.setattr(yolo_install, "can_auto_install", lambda: True)
        monkeypatch.setattr(yolo_install, "_confirm_agpl", lambda _p: True)

        class _FakeDialog:
            def __init__(self, _parent):
                self._r = yolo_install.InstallResult(False, "nope")

            def set_result(self, r):
                self._r = r

            def result_info(self):
                return self._r

            def exec(self):
                return None

        class _FakeWorker:
            def __init__(self, _parent):
                pass

            completed = MagicMock()

            def start(self):
                pass

            def deleteLater(self):
                pass

        monkeypatch.setattr(yolo_install, "_InstallProgressDialog", _FakeDialog)
        monkeypatch.setattr(yolo_install, "_InstallWorker", _FakeWorker)

        warnings = []
        monkeypatch.setattr(
            yolo_install.QMessageBox,
            "warning",
            lambda *a, **k: warnings.append(a),
        )

        assert yolo_install.ensure_ultralytics_installed(None) is False
        assert len(warnings) == 1
