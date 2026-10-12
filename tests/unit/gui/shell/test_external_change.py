"""Watching the open experiment for changes made outside GLIDER (e.g. by an agent)."""

from __future__ import annotations

import json
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


def test_a_change_the_os_never_reports_is_still_found(qtbot, experiment, monkeypatch):
    """Windows' watcher can miss a swap-in save, and network shares report nothing:
    the periodic check must catch it on its own."""
    w = ExperimentFileWatcher(debounce_ms=DEBOUNCE_MS, poll_ms=100)
    monkeypatch.setattr(w, "_arm", lambda: None)  # no OS notifications at all
    w.watch(str(experiment))
    try:
        with qtbot.waitSignal(w.changed_externally, timeout=3000):
            atomic_write_text(experiment, '{"v": "longer than before"}')
    finally:
        w.stop()


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


# --- the main window --------------------------------------------------------


def _open_in_window(main_window_factory, tmp_path):
    window = main_window_factory(desktop_mode=True)
    path = tmp_path / "agent.glider"
    window._core.save_session(str(path))
    assert window._load_experiment_path(str(path))
    return window, path


def _agent_adds_a_node(path: Path, node_id: str = "agent_delay") -> None:
    data = json.loads(path.read_text())
    data["flow"]["nodes"].append(
        {"id": node_id, "node_type": "Delay", "position": [300.0, 0.0], "state": {}}
    )
    atomic_write_text(path, json.dumps(data, indent=2))


def test_window_reloads_an_outside_change_when_nothing_is_unsaved(
    qtbot, main_window_factory, tmp_path
):
    window, path = _open_in_window(main_window_factory, tmp_path)
    assert not window._core.session.is_dirty

    _agent_adds_a_node(path)

    qtbot.waitUntil(lambda: "agent_delay" in window._graph_view.nodes, timeout=5000)
    assert window._external_change_bar.isHidden()


def test_window_asks_before_discarding_unsaved_edits(qtbot, main_window_factory, tmp_path):
    window, path = _open_in_window(main_window_factory, tmp_path)
    window._core.session.mark_dirty()

    _agent_adds_a_node(path)

    qtbot.waitUntil(lambda: not window._external_change_bar.isHidden(), timeout=5000)
    assert "agent_delay" not in window._graph_view.nodes

    window._external_change_bar._reload.click()
    assert "agent_delay" in window._graph_view.nodes
    assert window._external_change_bar.isHidden()


def test_keep_mine_dismisses_without_reloading(qtbot, main_window_factory, tmp_path):
    window, path = _open_in_window(main_window_factory, tmp_path)
    window._core.session.mark_dirty()
    _agent_adds_a_node(path)
    qtbot.waitUntil(lambda: not window._external_change_bar.isHidden(), timeout=5000)

    window._external_change_bar._keep.click()

    assert window._external_change_bar.isHidden()
    assert "agent_delay" not in window._graph_view.nodes
    assert window._core.session.is_dirty


def test_glider_saving_its_own_file_does_not_reload(qtbot, main_window_factory, tmp_path):
    window, path = _open_in_window(main_window_factory, tmp_path)
    window._core.session.mark_dirty()

    assert window._on_save()

    with qtbot.assertNotEmitted(window._file_watcher.changed_externally, wait=800):
        pass
    assert window._external_change_bar.isHidden()


def test_a_change_during_a_run_waits_for_the_run_to_end(qtbot, main_window_factory, tmp_path):
    from glider.core.experiment_session import SessionState

    window, path = _open_in_window(main_window_factory, tmp_path)
    window._core.session.state = SessionState.RUNNING

    _agent_adds_a_node(path)
    qtbot.wait(1000)
    assert "agent_delay" not in window._graph_view.nodes

    window._core.session.state = SessionState.IDLE
    qtbot.waitUntil(lambda: "agent_delay" in window._graph_view.nodes, timeout=5000)


def test_new_experiment_stops_watching(qtbot, main_window_factory, tmp_path):
    window, path = _open_in_window(main_window_factory, tmp_path)

    assert window._on_new()

    assert window._file_watcher.path is None
