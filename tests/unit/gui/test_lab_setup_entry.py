"""How Lab Setup is reached: once after the walkthrough, and always from the menu.

Two failures this file exists to catch, both silent.

**Skip and Done both mean "seen".** ``LabSetupDialog`` exits through
``reject()`` for Skip and ``accept()`` for Done. Recording the first run as
complete only on ``Accepted`` would re-ask at every launch, turning a
first-class exit into a nag -- and a wizard that punishes skipping is exactly
what gets junk typed into it to make it go away. That is the whole point of
``test_skipping_setup_does_not_re_offer_it``.

**Re-entry is required, not optional.** The person doing first launch is often
not the person who knows the lab's strains, so the knowledge arrives after the
only chance to enter it unless Experiment -> Lab Setup... reopens the same
dialog afterwards.

Every test injects its own ``QSettings`` file, so nothing here reads or writes
the developer's real settings, and nothing depends on whether this machine has
already seen the tour.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, QSettings, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QDialog, QMainWindow, QWidget  # noqa: E402

from glider.gui.main_window import LAB_SETUP_COMPLETE_KEY, lab_setup_complete  # noqa: E402
from glider.gui.onboarding.tour import TOUR_COMPLETE_KEY, Tour, TourStep  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> QSettings:
    """A QSettings backed by a fresh INI file -- no leakage either way."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture(autouse=True)
def library_dir(tmp_path, monkeypatch):
    """Point the vocabulary file at a tmp dir, never the developer's ~/.glider."""
    from glider.core.config import get_config

    path = tmp_path / "library"
    path.mkdir()
    monkeypatch.setattr(get_config().paths, "library_dir", path)
    return path


@pytest.fixture
def dialogs(qtbot, monkeypatch):
    """Record every Lab Setup dialog opened, and choose how the user closes it.

    Patches the real class rather than substituting a stub: a change to the
    dialog's constructor signature must fail here, not pass against a fake.
    Only ``exec`` is replaced, so nothing blocks on a modal window.
    """
    from glider.gui.dialogs.lab_setup_dialog import LabSetupDialog

    record = types.SimpleNamespace(opened=[], result=QDialog.DialogCode.Rejected.value)
    original_init = LabSetupDialog.__init__

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        qtbot.addWidget(self)
        record.opened.append(self)

    monkeypatch.setattr(LabSetupDialog, "__init__", _init)
    monkeypatch.setattr(LabSetupDialog, "exec", lambda self: record.result)
    return record


def _menu_only_window():
    """A MainWindow with only ``_setup_menu`` run (no panels).

    The full ``__init__`` builds panels against a live core and blocks; this
    mirrors the bypass used by ``tests/unit/gui/test_main_window_tools_menu.py``
    so the real menu code path is exercised in isolation.
    """
    from glider.gui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)  # skip the heavy __init__
    QMainWindow.__init__(win)  # real Qt base so menuBar() works
    win._view_manager = types.SimpleNamespace(is_runner_mode=False)
    win._setup_menu()
    return win


@pytest.fixture
def window(qtbot):
    # Deliberately not qtbot.addWidget'd: closeEvent touches panels the
    # bypassed __init__ never created.
    win = _menu_only_window()
    yield win
    win.deleteLater()


def _menu(win, title):
    for action in win.menuBar().actions():
        if action.text().replace("&", "") == title:
            return action.menu()
    return None


def _lab_setup_action(win):
    experiment = _menu(win, "Experiment")
    assert experiment is not None, "Experiment menu missing"
    return next(
        (a for a in experiment.actions() if "Lab Setup" in a.text().replace("&", "")),
        None,
    )


# --- The first-run offer -------------------------------------------------------


def test_setup_is_offered_when_unseen_and_the_walkthrough_has_resolved(window, settings, dialogs):
    settings.setValue(TOUR_COMPLETE_KEY, True)

    assert window.offer_lab_setup_once(settings) is True
    assert len(dialogs.opened) == 1


def test_setup_is_not_offered_once_it_has_been_seen(window, settings, dialogs):
    settings.setValue(TOUR_COMPLETE_KEY, True)
    settings.setValue(LAB_SETUP_COMPLETE_KEY, True)

    assert window.offer_lab_setup_once(settings) is False
    assert dialogs.opened == []


def test_setup_is_not_offered_before_the_walkthrough_resolves(window, settings, dialogs):
    """A form popped over the spotlight overlay hides the thing it points at."""
    assert lab_setup_complete(settings) is False

    assert window.offer_lab_setup_once(settings) is False
    assert dialogs.opened == []


def test_setup_is_not_offered_while_a_walkthrough_is_on_screen(window, settings, dialogs):
    """The flag can be set by an earlier walkthrough while a replay is running."""
    settings.setValue(TOUR_COMPLETE_KEY, True)
    window._active_tour = object()

    assert window.offer_lab_setup_once(settings) is False
    assert dialogs.opened == []


def test_skipping_setup_does_not_re_offer_it(window, settings, dialogs):
    """The one way this feature turns into a nag.

    Skip closes the dialog with ``Rejected``. If "seen" were recorded only on
    ``Accepted``, every launch would ask again until the user typed something
    -- anything -- to stop it, which is precisely the junk data the vocabulary
    exists to prevent.
    """
    settings.setValue(TOUR_COMPLETE_KEY, True)
    dialogs.result = QDialog.DialogCode.Rejected.value

    assert window.offer_lab_setup_once(settings) is True
    assert lab_setup_complete(settings) is True

    assert window.offer_lab_setup_once(settings) is False
    assert len(dialogs.opened) == 1, "skipping re-asked on the next launch"


def test_completing_setup_does_not_re_offer_it(window, settings, dialogs):
    settings.setValue(TOUR_COMPLETE_KEY, True)
    dialogs.result = QDialog.DialogCode.Accepted.value

    assert window.offer_lab_setup_once(settings) is True
    assert window.offer_lab_setup_once(settings) is False
    assert len(dialogs.opened) == 1


def test_the_setup_flag_is_its_own_key(settings):
    """Sharing the tour's flag would silence one by resolving the other."""
    assert LAB_SETUP_COMPLETE_KEY != TOUR_COMPLETE_KEY
    settings.setValue(TOUR_COMPLETE_KEY, True)

    assert lab_setup_complete(settings) is False


# --- Wiring: the offer follows the walkthrough ---------------------------------


class _FakeTour(QObject):
    finished = pyqtSignal()


def test_a_resolved_walkthrough_announces_itself(qtbot, settings):
    """Skip and the final Next both land in ``Tour._finish``; the signal is how
    anything downstream learns the overlay is gone."""
    host = QWidget()
    qtbot.addWidget(host)
    tour = Tour(host, steps=[TourStep(None, "T", "B")], settings=settings)
    seen = []
    tour.finished.connect(lambda: seen.append(True))

    tour.start()
    tour._finish()

    assert seen == [True]


def test_the_walkthrough_resolving_offers_setup(qtbot, window, monkeypatch):
    """Gating alone is not enough: something has to ask once the tour is done."""
    import glider.gui.onboarding as onboarding

    fake = _FakeTour()
    monkeypatch.setattr(onboarding, "start_tour", lambda host: fake)
    calls = []
    monkeypatch.setattr(window, "offer_lab_setup_once", lambda *a, **k: calls.append(True))

    window._start_tour()
    assert calls == [], "asked while the walkthrough was still on screen"

    fake.finished.emit()
    qtbot.waitUntil(lambda: calls == [True], timeout=1000)


# --- Re-entry from the menu ----------------------------------------------------


def test_the_experiment_menu_offers_lab_setup(window):
    action = _lab_setup_action(window)

    assert action is not None, "Experiment -> Lab Setup... missing"
    assert action.isEnabled() is True


def test_the_menu_action_opens_the_dialog(window, dialogs):
    _lab_setup_action(window).trigger()

    assert len(dialogs.opened) == 1


def test_the_menu_action_still_works_after_first_run_is_complete(window, settings, dialogs):
    """Re-entry is the whole point: the knowledge usually arrives later."""
    settings.setValue(TOUR_COMPLETE_KEY, True)
    settings.setValue(LAB_SETUP_COMPLETE_KEY, True)

    _lab_setup_action(window).trigger()

    assert len(dialogs.opened) == 1
