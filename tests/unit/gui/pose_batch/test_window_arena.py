"""PoseBatchWindow: arenas becoming the centre zones a run scores against."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from glider.gui.pose_batch.window import PoseBatchWindow
from glider.vision.arena import ArenaCalibration
from glider.vision.zones import ZoneShape

TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]


@pytest.fixture
def window(qtbot):
    widget = PoseBatchWindow()
    qtbot.addWidget(widget)
    return widget


def _video(tmp_path, name="t1_d2.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"")
    return path.resolve()


def _arena(corners=TRAPEZOID) -> ArenaCalibration:
    return ArenaCalibration(corners=corners, frame_size=(640, 480))


def _ready_window(window, tmp_path, *, count=1, with_csv=False):
    """A window whose only remaining Run blocker is calibration.

    `_validate` (window.py:1068-1108) checks model path, keypoint names, the
    model's keypoint count and the video list *before* it reaches calibration,
    so a test that only sets videos never exercises the branch it means to.
    """
    videos = [_video(tmp_path, f"t{i}_d1.mp4") for i in range(count)]
    window._videos = [v.resolve() for v in videos]
    window._model_path = tmp_path / "exp-7.pt"
    window._names_field.setText(",".join(f"kp{i}" for i in range(7)))
    window._meta = SimpleNamespace(n_keypoints=7)
    window._cal_table.set_videos(window._videos)  # else selected_videos() == []
    if with_csv:
        for v in window._videos:
            (v.parent / f"{v.stem}DLC_exp-7.csv").write_text("x")
    return videos[0] if count == 1 else videos


def _accept_arena_dialog(monkeypatch, *, returning):
    """Run ``_open_arena`` headless with an accepted dialog returning *returning*.

    Both halves have to be stubbed: the real dialog needs a frame, and the
    fixture videos are empty files that no reader can open.
    """
    import numpy as np
    from PyQt6.QtWidgets import QDialog

    class _Reader:
        frame_count = 100

        def load(self, path):
            return True

        def read_frame(self, n):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            pass

    class _Dialog:
        def __init__(self, frame, title="", parent=None, **kwargs):
            self.arena_spin = SimpleNamespace(setValue=lambda value: None)
            self.canvas = SimpleNamespace(set_corners=lambda corners: None)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def calibration(self):
            return returning

        def zone_size_cm(self):
            return 10.0

        def deleteLater(self):
            pass

    monkeypatch.setattr("glider.vision.video_source.VideoFileSource", _Reader)
    monkeypatch.setattr("glider.gui.dialogs.arena_dialog.ArenaDialog", _Dialog)


def _line_calibration():
    """A CameraCalibration with a drawn line, i.e. what satisfies Run today."""
    from glider.vision.calibration import CalibrationLine, CameraCalibration, LengthUnit

    return CameraCalibration(
        lines=[
            CalibrationLine(
                start_x=0.2,
                start_y=0.5,
                end_x=0.8,
                end_y=0.5,
                length=300.0,
                unit=LengthUnit.MILLIMETERS,
            )
        ],
        calibration_width=640,
        calibration_height=480,
    )


class TestZoneConfigs:
    def test_a_drawn_arena_becomes_a_zone(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        configs = window._zone_configs()
        assert list(configs) == [video]
        assert len(configs[video].zones) == 1

    def test_the_zone_is_a_polygon(self, window, tmp_path):
        # Under perspective a centred square is a quadrilateral; storing it as
        # a rectangle would put back the error the arena removes.
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        zone = window._zone_configs()[video].zones[0]
        assert zone.shape is ZoneShape.POLYGON
        assert len(zone.vertices) == 4

    def test_the_zone_records_the_frame_size(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        config = window._zone_configs()[video]
        assert (config.config_width, config.config_height) == (640, 480)

    def test_the_zone_name_carries_its_size(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._zone_cm = 12.0
        window._calibrations.set_arena(video, _arena())
        assert "12cm" in window._zone_configs()[video].zones[0].name

    def test_the_zone_follows_the_chosen_size(self, window, tmp_path):
        # Not a fixed ratio in image space: the mapping is projective, so a
        # 20 cm zone is not twice as wide on screen as a 10 cm one. What must
        # hold is that the larger zone strictly encloses the smaller.

        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        window._zone_cm = 10.0
        small = window._zone_configs()[video].zones[0].vertices
        window._zone_cm = 20.0
        large_config = window._zone_configs()[video]

        larger = large_config.zones[0]
        assert all(larger.contains_point(x, y) for x, y in small)
        span = lambda vs: max(v[0] for v in vs) - min(v[0] for v in vs)  # noqa: E731
        assert span(larger.vertices) > span(small)

    def test_videos_without_an_arena_get_no_zone(self, window, tmp_path):
        drawn, blank = _video(tmp_path, "a.mp4"), _video(tmp_path, "b.mp4")
        window._videos = [drawn, blank]
        window._calibrations.set_arena(drawn, _arena())
        assert list(window._zone_configs()) == [drawn]

    def test_no_arenas_means_no_zones(self, window, tmp_path):
        window._videos = [_video(tmp_path)]
        assert window._zone_configs() == {}


class TestBadArenas:
    def test_a_degenerate_arena_is_skipped_not_raised(self, window, tmp_path):
        # Pose inference is the expensive part of a run; a half-drawn zone must
        # not be able to stop it.
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena([(0.5, 0.5)] * 4))
        assert window._zone_configs() == {}

    def test_the_skip_is_reported(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena([(0.5, 0.5)] * 4))
        window._zone_configs()
        assert "no zone written" in window._log.toPlainText()

    def test_one_bad_arena_does_not_lose_the_others(self, window, tmp_path):
        good, bad = _video(tmp_path, "a.mp4"), _video(tmp_path, "b.mp4")
        window._videos = [good, bad]
        window._calibrations.set_arena(good, _arena())
        window._calibrations.set_arena(bad, _arena([(0.5, 0.5)] * 4))
        assert list(window._zone_configs()) == [good]

    def test_a_zone_larger_than_the_arena_is_skipped(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._zone_cm = 40.0
        window._calibrations.set_arena(video, _arena())
        assert window._zone_configs() == {}


class TestRunGating:
    def test_an_arena_alone_satisfies_the_calibration_gate(self, window, tmp_path):
        # px_per_mm() answers from the arena, so a video with a perimeter and
        # no drawn line is calibrated as far as Run is concerned.
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        assert window._calibrations.missing(window._videos) == []


def test_load_master_applies_arenas(window, tmp_path):
    """Regression: _load_master applied entries and dropped arenas."""
    from glider.vision.calibration_set import CalibrationSet

    video = _video(tmp_path)
    master = tmp_path / "master.json"
    seed = CalibrationSet()
    seed.set_arena(video, _arena())
    seed.save(master)

    window._videos = [video.resolve()]
    window._load_master(master)

    assert window._calibrations.get_arena(video) is not None


class TestLoadMasterKeepsConfirmedState:
    """Regression: ``_load_master`` merged ``entries`` and ``arenas`` by hand
    and dropped ``_unconfirmed``, so one save/reload cycle confirmed every
    copied arena -- on exactly the workflow copies exist for."""

    def _saved_master(self, tmp_path, video, *, confirmed):
        from glider.vision.calibration_set import CalibrationSet

        master = tmp_path / "master.json"
        seed = CalibrationSet()
        seed.set_arena(video, _arena(), confirmed=confirmed)
        seed.save(master)
        return master

    def test_a_copied_arena_is_still_unconfirmed_after_a_reload(self, window, tmp_path):
        video = _ready_window(window, tmp_path)
        master = self._saved_master(tmp_path, video, confirmed=False)

        window._load_master(master)

        assert window._calibrations.get_arena(video) is not None
        assert not window._calibrations.is_arena_confirmed(video)

    def test_run_stays_blocked_after_the_reload(self, window, tmp_path):
        """The whole point of the flag: Run must not come back on its own."""
        video = _ready_window(window, tmp_path)
        master = self._saved_master(tmp_path, video, confirmed=False)

        window._load_master(master)

        assert window._calibrations.missing_arenas(window._videos) == [video]
        assert not window._run_button.isEnabled()

    def test_loading_a_confirmed_arena_clears_a_stale_flag(self, window, tmp_path):
        """The reverse direction: the file is the truth for the videos it names,
        so an arena confirmed on disk must not stay blocked by an old copy."""
        video = _ready_window(window, tmp_path)
        window._calibrations.set_arena(video, _arena(), confirmed=False)
        master = self._saved_master(tmp_path, video, confirmed=True)

        window._load_master(master)

        assert window._calibrations.is_arena_confirmed(video)


class TestRunRequiresArena:
    """A drawn arena replaces the line as the Run gate's calibration check."""

    def test_run_is_blocked_by_a_line_only_calibration(self, window, tmp_path):
        video = _ready_window(window, tmp_path)
        window._calibrations.set(video, _line_calibration())
        window._validate()
        assert not window._run_button.isEnabled()
        assert "arena" in window._run_button.toolTip().lower()

    def test_run_is_blocked_by_an_unconfirmed_arena(self, window, tmp_path):
        video = _ready_window(window, tmp_path)
        window._calibrations.set_arena(video, _arena(), confirmed=False)
        window._validate()
        assert not window._run_button.isEnabled()

    def test_a_confirmed_arena_alone_enables_run(self, window, tmp_path):
        """No line drawn at all. The arena carries the scale."""
        video = _ready_window(window, tmp_path)
        window._calibrations.set_arena(video, _arena())
        window._validate()
        assert window._run_button.isEnabled()

    def test_the_badge_counts_arenas(self, window, tmp_path):
        videos = _ready_window(window, tmp_path, count=2)
        window._calibrations.set_arena(videos[0], _arena())
        window._validate()
        # Card has no badge getter; read the underlying label it sets.
        assert "1 / 2 arenas drawn" in window._calibration_card._badge.text()

    def test_clear_removes_the_arena_as_well_as_the_line(self, window, tmp_path):
        video = _ready_window(window, tmp_path)
        window._calibrations.set_arena(video, _arena())
        window._cal_table.selectRow(0)
        assert window._cal_table.selected_videos()  # guard: the row really is selected
        window._clear_selected_calibrations()
        assert window._calibrations.get_arena(video) is None


class TestCopyingAnArena:
    """Copying is a starting point, not a calibration: it lands unconfirmed."""

    def test_copying_an_arena_lands_unconfirmed(self, window, tmp_path, monkeypatch):
        """A copied arena that does not fit shows no residual warning, so it must
        not satisfy the Run gate until someone has seen the overlay."""
        from glider.gui.pose_batch import arena_actions

        monkeypatch.setattr(arena_actions, "resolution_of", lambda v: (640, 480))
        videos = _ready_window(window, tmp_path, count=2)
        window._calibrations.set_arena(videos[0], _arena())

        arena_actions.copy_arena_to(window._calibrations, videos[0], videos[1:])

        assert window._calibrations.get_arena(videos[1]) is not None
        assert not window._calibrations.is_arena_confirmed(videos[1])
        assert window._calibrations.missing_arenas(videos) == [videos[1]]

    def test_a_copied_arena_takes_the_target_resolution(self, window, tmp_path, monkeypatch):
        from glider.gui.pose_batch import arena_actions

        monkeypatch.setattr(arena_actions, "resolution_of", lambda v: (1280, 720))
        videos = _ready_window(window, tmp_path, count=2)
        window._calibrations.set_arena(videos[0], _arena())
        arena_actions.copy_arena_to(window._calibrations, videos[0], videos[1:])
        assert window._calibrations.get_arena(videos[1]).frame_size == (1280, 720)

    def test_an_unreadable_target_is_skipped_not_guessed(self, window, tmp_path, monkeypatch):
        from glider.gui.pose_batch import arena_actions

        monkeypatch.setattr(arena_actions, "resolution_of", lambda v: None)
        videos = _ready_window(window, tmp_path, count=2)
        window._calibrations.set_arena(videos[0], _arena())
        skipped = arena_actions.copy_arena_to(window._calibrations, videos[0], videos[1:])
        assert window._calibrations.get_arena(videos[1]) is None
        assert skipped == [videos[1]]

    def test_accepting_the_arena_dialog_confirms_a_copy(self, window, tmp_path, monkeypatch):
        """Opening a copied arena and pressing OK is what confirms it."""
        videos = _ready_window(window, tmp_path, count=2)
        window._calibrations.set_arena(videos[1], _arena(), confirmed=False)
        _accept_arena_dialog(monkeypatch, returning=_arena())
        window._open_arena(videos[1])
        assert window._calibrations.is_arena_confirmed(videos[1])


def _capture_worker(monkeypatch) -> dict:
    """Run ``_start_worker`` for real but stop short of any inference."""
    from glider.gui.pose_batch import worker as worker_mod

    captured: dict = {}
    real = worker_mod.PoseBatchWorker

    class Capturing(real):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

        def run(self):  # never touch the GPU from a unit test
            pass

    monkeypatch.setattr(worker_mod, "PoseBatchWorker", Capturing)
    return captured


class TestTheRunCarriesTheArenas:
    """Inference-time gating and arena-aware candidate selection are reachable
    only through the worker; the window is where the arenas come from."""

    def test_the_drawn_arenas_are_handed_to_the_worker(self, window, tmp_path, monkeypatch):
        captured = _capture_worker(monkeypatch)
        videos = _ready_window(window, tmp_path, count=2)
        arenas = [_arena(), _arena([(0.3, 0.2), (0.7, 0.2), (0.75, 0.8), (0.25, 0.8)])]
        for video, arena in zip(videos, arenas, strict=True):
            window._calibrations.set_arena(video, arena)

        window._start_worker()
        window._teardown_thread()

        assert captured["arenas"] == {v.resolve(): a for v, a in zip(videos, arenas, strict=True)}

    def test_the_gate_is_enabled_for_the_run(self, window, tmp_path, monkeypatch):
        from glider.vision.arena_gate import ArenaGateSettings

        captured = _capture_worker(monkeypatch)
        video = _ready_window(window, tmp_path)
        window._calibrations.set_arena(video, _arena())

        window._start_worker()
        window._teardown_thread()

        # No UI for tuning it yet, so the defaults are the contract; what
        # matters is that gating is on at all, since zones are scored from the
        # gated pose and candidate re-ranking cannot be redone afterwards.
        assert captured["gate"] == ArenaGateSettings()

    def test_only_the_listed_videos_are_carried(self, window, tmp_path, monkeypatch):
        """The set can hold arenas from folders visited earlier this session."""
        captured = _capture_worker(monkeypatch)
        video = _ready_window(window, tmp_path)
        window._calibrations.set_arena(video, _arena())
        window._calibrations.set_arena(_video(tmp_path, "elsewhere.mp4"), _arena())

        window._start_worker()
        window._teardown_thread()

        assert list(captured["arenas"]) == [video.resolve()]


class TestRunIsBlockedDuringARegate:
    """Both passes write the same primary CSVs.

    ``gate_pose_csv`` renames the primary to ``_ungated`` and rewrites it in
    place; ``run_batch`` writes the primary. Overlap them and a half-written
    primary is captured as the permanent "pristine original", and the re-gate
    has no cancel, so the window is tens of seconds wide on a real cohort.
    """

    def _regating(self, window, tmp_path):
        """A window that would otherwise be ready to Run, mid-re-gate."""
        video = _ready_window(window, tmp_path, with_csv=True)
        window._calibrations.set_arena(video, _arena())
        window._validate()
        assert window._run_button.isEnabled()  # guard: nothing else is blocking
        # What _start_regate leaves behind. A stand-in rather than a real
        # QThread, but it answers closeEvent's quit/wait so the fixture can
        # still tear the window down.
        window._regate_thread = SimpleNamespace(quit=lambda: None, wait=lambda ms: None)
        window._run_button.setEnabled(False)
        return video

    def test_validate_does_not_re_enable_run(self, window, tmp_path):
        self._regating(window, tmp_path)
        window._validate()
        assert not window._run_button.isEnabled()

    def test_a_signal_that_calls_validate_does_not_re_enable_run(self, window, tmp_path):
        """_overwrite.toggled is one of four connections that reach _validate."""
        self._regating(window, tmp_path)
        window._overwrite.toggle()
        assert not window._run_button.isEnabled()

    def test_start_refuses(self, window, tmp_path, monkeypatch):
        captured = _capture_worker(monkeypatch)
        self._regating(window, tmp_path)

        window._start()

        assert captured == {}
        assert window._thread is None

    def test_run_comes_back_once_the_regate_is_torn_down(self, window, tmp_path):
        """The block is the re-gate's, not a latch that outlives it."""
        self._regating(window, tmp_path)
        window._regate_thread = None
        window._validate()
        assert window._run_button.isEnabled()


def test_regate_is_disabled_without_pose_csvs(window, tmp_path):
    video = _ready_window(window, tmp_path)
    window._calibrations.set_arena(video, _arena())
    window._validate()
    assert not window._regate_button.isEnabled()


def test_regate_is_disabled_by_an_unconfirmed_arena(window, tmp_path):
    video = _ready_window(window, tmp_path, with_csv=True)
    window._calibrations.set_arena(video, _arena(), confirmed=False)
    window._validate()
    assert not window._regate_button.isEnabled()


def test_regate_is_enabled_with_a_confirmed_arena_and_a_csv(window, tmp_path):
    video = _ready_window(window, tmp_path, with_csv=True)
    window._calibrations.set_arena(video, _arena())
    window._validate()
    assert window._regate_button.isEnabled()


def test_one_bad_video_does_not_end_the_pass(tmp_path, monkeypatch):
    """A refusal or an unreadable file is a skip, not a stop -- the whole point
    of a batch operation is that it finishes."""
    from glider.gui.pose_batch.arena_actions import regate_videos
    from glider.vision.calibration_set import CalibrationSet

    videos = [_video(tmp_path, "a.mp4"), _video(tmp_path, "b.mp4")]
    calibrations = CalibrationSet()
    for video in videos:
        calibrations.set_arena(video, _arena())
        (video.parent / f"{video.stem}DLC_exp-7.csv").write_text("x")

    calls = []

    def flaky(csv, arena, settings=None):
        calls.append(csv)
        if len(calls) == 1:
            raise ValueError("refused")
        return SimpleNamespace(blanked_fraction=0.0)

    monkeypatch.setattr("glider.vision.arena_gate.gate_pose_csv", flaky)
    gated, skipped = regate_videos(videos, calibrations, on_log=lambda m: None)
    assert len(calls) == 2
    assert gated == 1 and skipped == 1
