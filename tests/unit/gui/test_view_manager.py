"""Mode auto-detection: Pi hardware wins; screen-size heuristics otherwise."""

from PyQt6.QtCore import QSize

from glider.gui.view_manager import ViewManager, ViewMode


def _manager(monkeypatch, *, on_pi: bool, screen: QSize) -> ViewManager:
    vm = ViewManager(None)
    monkeypatch.setattr(ViewManager, "_is_raspberry_pi", staticmethod(lambda: on_pi))
    vm._screen_size = screen
    return vm


def test_pi_hardware_forces_runner_even_on_desktop_sized_screen(monkeypatch):
    # A phantom HDMI output / compositor can report a desktop-sized primary
    # screen on the Pi; hardware detection must win over the size heuristic.
    vm = _manager(monkeypatch, on_pi=True, screen=QSize(1920, 1080))
    assert vm.mode == ViewMode.RUNNER
    assert vm.is_runner_mode


def test_non_pi_desktop_screen_detects_desktop(monkeypatch):
    vm = _manager(monkeypatch, on_pi=False, screen=QSize(1920, 1080))
    assert vm.mode == ViewMode.DESKTOP


def test_non_pi_small_touchscreen_still_detects_runner(monkeypatch):
    vm = _manager(monkeypatch, on_pi=False, screen=QSize(480, 800))
    assert vm.mode == ViewMode.RUNNER


def test_forced_mode_bypasses_detection(monkeypatch):
    vm = _manager(monkeypatch, on_pi=True, screen=QSize(480, 800))
    vm.mode = ViewMode.DESKTOP
    assert vm.mode == ViewMode.DESKTOP
    assert not vm.is_runner_mode


def test_is_raspberry_pi_false_when_probe_unreadable(monkeypatch):
    import builtins

    real_open = builtins.open

    def deny(path, *args, **kwargs):
        if "device-tree" in str(path):
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny)
    assert ViewManager._is_raspberry_pi() is False
