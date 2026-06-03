"""Unit tests for ``glider.updater``.

Split into three groups matching the module's own structure:

* Pure version-comparison logic (no Qt, no network).
* HTTP layer (monkeypatched :func:`httpx.get` — no real network calls).
* Qt controller (requires ``qapp`` from ``pytest-qt``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from PyQt6.QtCore import QSettings

from glider.updater import (
    SKIPPED_VERSION_KEY,
    UpdateChecker,
    UpdateInfo,
    fetch_latest_release,
    is_newer,
    parse_version,
)

# --- Pure logic --------------------------------------------------------------


class TestVersionParsing:
    def test_strips_v_prefix(self) -> None:
        assert str(parse_version("v1.2.3")) == "1.2.3"

    def test_strips_capital_v(self) -> None:
        assert str(parse_version("V2.0.0")) == "2.0.0"

    def test_accepts_bare(self) -> None:
        assert str(parse_version("1.2.3")) == "1.2.3"

    def test_handles_whitespace(self) -> None:
        assert str(parse_version("  v1.2.3 ")) == "1.2.3"


class TestIsNewer:
    def test_newer_patch(self) -> None:
        assert is_newer("1.0.1", "1.0.0") is True

    def test_newer_minor(self) -> None:
        assert is_newer("1.1.0", "1.0.9") is True

    def test_newer_major(self) -> None:
        assert is_newer("2.0.0", "1.99.0") is True

    def test_equal(self) -> None:
        assert is_newer("1.0.0", "1.0.0") is False

    def test_older(self) -> None:
        assert is_newer("1.0.0", "1.0.1") is False

    def test_tag_prefix_both_sides(self) -> None:
        assert is_newer("v1.2.0", "v1.1.0") is True

    def test_mixed_prefix(self) -> None:
        # GitHub tag is v-prefixed; our in-code version is bare.
        assert is_newer("v1.2.0", "1.1.9") is True

    def test_invalid_latest_returns_false(self) -> None:
        # A garbage tag on GitHub must never cause us to nag the user.
        assert is_newer("not-a-version", "1.0.0") is False

    def test_invalid_current_returns_false(self) -> None:
        assert is_newer("1.0.0", "not-a-version") is False


# --- HTTP layer --------------------------------------------------------------


def _make_response(status_code: int = 200, json_body: dict | None = None) -> MagicMock:
    """Build a minimal stand-in for :class:`httpx.Response`."""
    r = MagicMock(spec=httpx.Response)
    r.status_code = status_code
    r.json.return_value = json_body or {}
    if status_code >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=r
        )
    else:
        r.raise_for_status.return_value = None
    return r


class TestFetchLatestRelease:
    def test_happy_path(self) -> None:
        payload = {
            "tag_name": "v1.3.0",
            "html_url": "https://github.com/LaingLab/glider/releases/tag/v1.3.0",
            "body": "## What's new\n- Thing A\n- Thing B",
            "draft": False,
            "prerelease": False,
        }
        with patch("glider.updater.httpx.get", return_value=_make_response(200, payload)):
            info = fetch_latest_release()
        assert info is not None
        assert info.latest_version == "v1.3.0"
        assert "releases/tag/v1.3.0" in info.release_url
        assert "Thing A" in info.release_notes

    def test_network_error_returns_none(self) -> None:
        with patch("glider.updater.httpx.get", side_effect=httpx.ConnectError("dns")):
            assert fetch_latest_release() is None

    def test_timeout_returns_none(self) -> None:
        with patch("glider.updater.httpx.get", side_effect=httpx.ReadTimeout("slow")):
            assert fetch_latest_release() is None

    def test_http_error_returns_none(self) -> None:
        with patch("glider.updater.httpx.get", return_value=_make_response(403)):
            # 403 typically means rate-limited; we must degrade silently.
            assert fetch_latest_release() is None

    def test_draft_is_ignored(self) -> None:
        payload = {"tag_name": "v1.3.0", "draft": True, "prerelease": False}
        with patch("glider.updater.httpx.get", return_value=_make_response(200, payload)):
            assert fetch_latest_release() is None

    def test_prerelease_is_ignored(self) -> None:
        payload = {"tag_name": "v1.3.0-rc1", "draft": False, "prerelease": True}
        with patch("glider.updater.httpx.get", return_value=_make_response(200, payload)):
            assert fetch_latest_release() is None

    def test_missing_tag_returns_none(self) -> None:
        # Shouldn't happen in practice, but the API contract isn't ours.
        with patch("glider.updater.httpx.get", return_value=_make_response(200, {})):
            assert fetch_latest_release() is None


# --- Qt controller -----------------------------------------------------------


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch) -> QSettings:
    """A QSettings backed by a fresh INI file — no cross-test leakage."""
    ini = tmp_path / "glider-test.ini"
    s = QSettings(str(ini), QSettings.Format.IniFormat)
    # Clear anything a previous import may have populated.
    s.clear()
    return s


class TestUpdateCheckerResultHandling:
    """Exercise :meth:`UpdateChecker._on_result` directly.

    Bypassing the worker thread keeps these tests fast and deterministic; the
    thread is a thin wrapper around :func:`fetch_latest_release`, which is
    already covered above.
    """

    def test_silent_mode_no_ui_when_up_to_date(self, qapp, isolated_settings, monkeypatch) -> None:
        checker = UpdateChecker(parent=None, current_version="1.0.0", settings=isolated_settings)
        shown = []
        monkeypatch.setattr(
            "glider.updater.QMessageBox.information",
            lambda *a, **k: shown.append(("info", a, k)),
        )
        # No prompt branch either — just track _show_prompt calls.
        monkeypatch.setattr(
            UpdateChecker, "_show_prompt", lambda self, info: shown.append(("prompt", info))
        )

        checker._silent = True
        checker._on_result(UpdateInfo("1.0.0", "https://x", ""))

        assert shown == []  # nothing user-visible when silent and up-to-date

    def test_manual_mode_shows_up_to_date(self, qapp, isolated_settings, monkeypatch) -> None:
        checker = UpdateChecker(parent=None, current_version="1.0.0", settings=isolated_settings)
        shown = []
        monkeypatch.setattr(
            "glider.updater.QMessageBox.information",
            lambda *a, **k: shown.append("up-to-date"),
        )
        monkeypatch.setattr(
            UpdateChecker, "_show_prompt", lambda self, info: shown.append("prompt")
        )

        checker._silent = False
        checker._on_result(UpdateInfo("1.0.0", "https://x", ""))

        assert shown == ["up-to-date"]

    def test_silent_mode_shows_prompt_when_newer(
        self, qapp, isolated_settings, monkeypatch
    ) -> None:
        checker = UpdateChecker(parent=None, current_version="1.0.0", settings=isolated_settings)
        shown = []
        monkeypatch.setattr(UpdateChecker, "_show_prompt", lambda self, info: shown.append(info))

        checker._silent = True
        info = UpdateInfo("1.1.0", "https://x", "notes")
        checker._on_result(info)

        assert shown == [info]

    def test_silent_mode_respects_skipped_version(
        self, qapp, isolated_settings, monkeypatch
    ) -> None:
        # Simulate the user having previously clicked "Don't ask again" for 1.1.0.
        isolated_settings.setValue(SKIPPED_VERSION_KEY, "1.1.0")
        checker = UpdateChecker(parent=None, current_version="1.0.0", settings=isolated_settings)
        shown = []
        monkeypatch.setattr(UpdateChecker, "_show_prompt", lambda self, info: shown.append(info))

        checker._silent = True
        checker._on_result(UpdateInfo("1.1.0", "https://x", ""))

        assert shown == []  # suppressed

    def test_manual_mode_ignores_skipped_version(
        self, qapp, isolated_settings, monkeypatch
    ) -> None:
        # User asked "don't ask again" for 1.1.0 but then manually clicked
        # Help → Check for Updates. That counts as un-snoozing — the prompt
        # should appear so they can change their mind.
        isolated_settings.setValue(SKIPPED_VERSION_KEY, "1.1.0")
        checker = UpdateChecker(parent=None, current_version="1.0.0", settings=isolated_settings)
        shown = []
        monkeypatch.setattr(UpdateChecker, "_show_prompt", lambda self, info: shown.append(info))

        checker._silent = False
        checker._on_result(UpdateInfo("1.1.0", "https://x", ""))

        assert len(shown) == 1

    def test_skipped_version_suppresses_only_that_version(
        self, qapp, isolated_settings, monkeypatch
    ) -> None:
        isolated_settings.setValue(SKIPPED_VERSION_KEY, "1.1.0")
        checker = UpdateChecker(parent=None, current_version="1.0.0", settings=isolated_settings)
        shown = []
        monkeypatch.setattr(UpdateChecker, "_show_prompt", lambda self, info: shown.append(info))

        # Later release should still prompt.
        checker._silent = True
        checker._on_result(UpdateInfo("1.2.0", "https://x", ""))

        assert len(shown) == 1

    def test_network_failure_silent_mode_stays_quiet(
        self, qapp, isolated_settings, monkeypatch
    ) -> None:
        checker = UpdateChecker(parent=None, current_version="1.0.0", settings=isolated_settings)
        shown = []
        monkeypatch.setattr(
            "glider.updater.QMessageBox.information",
            lambda *a, **k: shown.append("info"),
        )
        checker._silent = True
        checker._on_result(None)
        assert shown == []  # silent on failure at startup

    def test_network_failure_manual_mode_surfaces_message(
        self, qapp, isolated_settings, monkeypatch
    ) -> None:
        checker = UpdateChecker(parent=None, current_version="1.0.0", settings=isolated_settings)
        shown = []
        monkeypatch.setattr(
            "glider.updater.QMessageBox.information",
            lambda *a, **k: shown.append("info"),
        )
        checker._silent = False
        checker._on_result(None)
        assert shown == ["info"]
