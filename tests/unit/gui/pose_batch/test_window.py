"""PoseBatchWindow: drop list, video discovery, and Run-gating validation.

No model is ever loaded here, so nothing imports torch.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from glider.gui.pose_batch.window import PoseBatchWindow, _DropList


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def window(qtbot):
    win = PoseBatchWindow()
    qtbot.addWidget(win)
    return win


# --------------------------------------------------------------------------
# drop list
# --------------------------------------------------------------------------


def test_drop_list_dedupes_paths(qtbot, tmp_path):
    widget = _DropList()
    qtbot.addWidget(widget)
    widget.add_paths([tmp_path, tmp_path])
    assert widget.paths() == [tmp_path]


def test_drop_list_emits_changed_on_add(qtbot, tmp_path):
    widget = _DropList()
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.changed, timeout=1000):
        widget.add_paths([tmp_path])


def test_drop_list_accepts_drops(qtbot):
    widget = _DropList()
    qtbot.addWidget(widget)
    assert widget.acceptDrops() is True


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_counts_videos_from_a_dropped_directory(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mp4")
    window._sources.add_paths([tmp_path])
    assert len(window._videos) == 2
    assert "2 videos found" in window._count_label.text()


def test_recursive_toggle_changes_the_count(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mp4")
    window._sources.add_paths([tmp_path])
    window._recursive.setChecked(False)
    assert len(window._videos) == 1
    assert "1 video found" in window._count_label.text()


def test_empty_directory_reports_none_found(window, tmp_path):
    window._sources.add_paths([tmp_path])
    assert "No videos found" in window._count_label.text()


# --------------------------------------------------------------------------
# validation gating
# --------------------------------------------------------------------------


def test_run_disabled_without_a_model(window):
    assert window._run_button.isEnabled() is False
    assert "Select a model" in window._run_button.toolTip()


def _ready(window, tmp_path, names="a, b", n_keypoints=2):
    """Put the window into an otherwise-runnable state."""
    _touch(tmp_path / "v.mp4")
    window._model_path = tmp_path / "model.pt"
    window._meta = types.SimpleNamespace(
        n_keypoints=n_keypoints, keypoint_names=None, source="checkpoint"
    )
    window._sources.add_paths([tmp_path])
    window._names_field.setText(names)
    return window


def test_run_enabled_when_everything_is_valid(window, tmp_path):
    _ready(window, tmp_path)
    assert window._run_button.isEnabled() is True


def test_run_blocked_on_duplicate_names(window, tmp_path):
    _ready(window, tmp_path, names="a, a")
    assert window._run_button.isEnabled() is False
    assert "unique" in window._run_button.toolTip()


def test_run_blocked_on_keypoint_count_mismatch(window, tmp_path):
    _ready(window, tmp_path, names="a, b, c", n_keypoints=2)
    assert window._run_button.isEnabled() is False
    assert "2 keypoints but 3 names" in window._run_button.toolTip()


def test_run_blocked_when_no_videos_found(window, tmp_path):
    window._model_path = tmp_path / "model.pt"
    window._meta = None
    window._names_field.setText("a, b")
    window._refresh_videos()
    assert window._run_button.isEnabled() is False
    assert "at least one video" in window._run_button.toolTip()


def test_run_blocked_on_empty_names(window, tmp_path):
    _ready(window, tmp_path, names="")
    assert window._run_button.isEnabled() is False


def test_invalid_names_are_marked(window, tmp_path):
    _ready(window, tmp_path, names="a, a")
    assert window._names_field.styleSheet() != ""
    _ready(window, tmp_path, names="a, b")
    assert window._names_field.styleSheet() == ""


# --------------------------------------------------------------------------
# filter settings
# --------------------------------------------------------------------------


def test_filtering_off_by_default(window):
    assert window._filter_group.isChecked() is False
    assert window._filter_settings() is None


def test_filter_settings_are_read_from_the_widgets(window):
    window._filter_group.setChecked(True)
    window._filter_conf.setValue(0.4)
    window._filter_gap.setValue(7)
    settings = window._filter_settings()
    assert settings.confidence_threshold == pytest.approx(0.4)
    assert settings.max_gap == 7


def test_median_window_stays_odd(window):
    window._filter_group.setChecked(True)
    window._filter_window.setValue(6)
    # medfilt rejects even kernels, so the widget coerces upward.
    assert window._filter_window.value() % 2 == 1
    assert window._filter_settings().median_window % 2 == 1


# --------------------------------------------------------------------------
# name parsing
# --------------------------------------------------------------------------


def test_names_are_parsed_and_trimmed(window):
    window._names_field.setText(" nose , l_ear ,, r_ear ")
    assert window._current_names() == ["nose", "l_ear", "r_ear"]
