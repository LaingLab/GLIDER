"""Watching the open experiment for changes made outside GLIDER (e.g. by an agent)."""

from __future__ import annotations

from pathlib import Path

import pytest

from glider.core.fileio import atomic_write_text
from glider.gui.shell.external_change import ExperimentFileWatcher, ExternalChangeBar

DEBOUNCE_MS = 50


@pytest.fixture
def experiment(tmp_path) -> Path:
    path = tmp_path / "blink.glider"
    atomic_write_text(path, '{"v": 1}')
    return path


@pytest.fixture
def watcher(qtbot, experiment):
    w = ExperimentFileWatcher(debounce_ms=DEBOUNCE_MS)
    w.watch(str(experiment))
    yield w
    w.stop()


def test_an_outside_rewrite_is_reported_once(qtbot, watcher, experiment):
    # Agents (and GLIDER) save by writing a temp file and swapping it in.
    with qtbot.waitSignal(watcher.changed_externally, timeout=3000) as signal:
        atomic_write_text(experiment, '{"v": 2}')
    assert signal.args == [str(experiment)]
    with qtbot.assertNotEmitted(watcher.changed_externally, wait=300):
        pass


def test_a_burst_of_writes_is_reported_once(qtbot, watcher, experiment):
    seen: list[str] = []
    watcher.changed_externally.connect(seen.append)
    for v in range(2, 6):
        atomic_write_text(experiment, f'{{"v": {v}}}')
    qtbot.waitUntil(lambda: bool(seen), timeout=3000)
    qtbot.wait(300)
    assert seen == [str(experiment)]


def test_identical_contents_are_not_reported(qtbot, watcher, experiment):
    with qtbot.assertNotEmitted(watcher.changed_externally, wait=400):
        atomic_write_text(experiment, '{"v": 1}')


def test_glider_own_save_is_not_reported(qtbot, watcher, experiment):
    """The window re-arms with watch() right after its own save."""
    with qtbot.assertNotEmitted(watcher.changed_externally, wait=400):
        atomic_write_text(experiment, '{"v": "saved by GLIDER"}')
        watcher.watch(str(experiment))


def test_watching_continues_after_a_swap(qtbot, watcher, experiment):
    """A swap-in replace can drop the OS watch; later changes must still arrive."""
    with qtbot.waitSignal(watcher.changed_externally, timeout=3000):
        atomic_write_text(experiment, '{"v": 2}')
    with qtbot.waitSignal(watcher.changed_externally, timeout=3000):
        atomic_write_text(experiment, '{"v": 3}')


def test_acknowledge_takes_the_disk_version_as_known(qtbot, watcher, experiment):
    with qtbot.waitSignal(watcher.changed_externally, timeout=3000):
        atomic_write_text(experiment, '{"v": 2}')
    watcher.acknowledge()
    with qtbot.assertNotEmitted(watcher.changed_externally, wait=400):
        atomic_write_text(experiment, '{"v": 2}')


def test_a_deleted_file_is_not_reported(qtbot, watcher, experiment):
    with qtbot.assertNotEmitted(watcher.changed_externally, wait=400):
        experiment.unlink()


def test_stop_and_unset_path_watch_nothing(qtbot, experiment):
    w = ExperimentFileWatcher(debounce_ms=DEBOUNCE_MS)
    w.watch(None)
    assert w.path is None
    w.watch(str(experiment))
    w.stop()
    assert w.path is None
    with qtbot.assertNotEmitted(w.changed_externally, wait=400):
        atomic_write_text(experiment, '{"v": 9}')


def test_bar_offers_reload_and_keep(qtbot):
    bar = ExternalChangeBar()
    qtbot.addWidget(bar)
    assert bar.isHidden()
    with qtbot.waitSignal(bar.reload_requested, timeout=1000):
        bar._reload.click()
    with qtbot.waitSignal(bar.keep_requested, timeout=1000):
        bar._keep.click()
    assert "changed outside GLIDER" in bar._message.text()
