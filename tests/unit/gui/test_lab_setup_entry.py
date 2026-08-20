"""How Lab Setup is reached: once per install, and always from the menu.

Three failures this file exists to catch, all silent.

**Skip and Done both mean "seen".** ``LabSetupDialog`` exits through
``reject()`` for Skip and ``accept()`` for Done. Recording the first run as
complete only on ``Accepted`` would re-ask at every launch, turning a
first-class exit into a nag -- and a wizard that punishes skipping is exactly
what gets junk typed into it to make it go away. See
``test_skipping_setup_does_not_re_offer_it``.

**An existing install must be offered it.** Gating the offer on the golden-path
walkthrough finishing reaches new installs only: every lab already running
GLIDER resolved that walkthrough long ago, and they are precisely the people
who reported that the subject fields are invisible. So the offer is made at
launch as well as when a tour resolves, and both paths are gated on one flag
written before the dialog opens, so they cannot double-offer. See
``test_an_existing_install_is_offered_setup_at_launch``.

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
from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow, QWidget  # noqa: E402

from glider.first_run import FIRST_RUN_COMPLETE_KEY  # noqa: E402
from glider.gui.main_window import LAB_SETUP_COMPLETE_KEY, lab_setup_complete  # noqa: E402
from glider.gui.onboarding.tour import TOUR_COMPLETE_KEY, Tour, TourStep  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> QSettings:
    """A QSettings backed by a fresh INI file -- no leakage either way."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def existing_install(settings) -> QSettings:
    """A machine that has been running GLIDER since before this feature.

    The welcome is long since answered and the walkthrough long since resolved,
    so nothing will ever fire the tour-finished path for them again.
    """
    settings.setValue(FIRST_RUN_COMPLETE_KEY, True)
    settings.setValue(TOUR_COMPLETE_KEY, True)
    return settings


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
    """The named menu, read off the window's registry rather than the bar.

    Task 6 took Experiment off the menu bar; the menu is still built, and is
    what the command palette reads. See ``test_main_window_tools_menu.py``,
    where the reasoning is written out.
    """
    for menu in win.menus():
        if menu.title().replace("&", "") == title:
            return menu
    return None


def _lab_setup_action(win):
    experiment = _menu(win, "Experiment")
    assert experiment is not None, "Experiment menu missing"
    return next(
        (a for a in experiment.actions() if "Lab Setup" in a.text().replace("&", "")),
        None,
    )


# --- The gate ------------------------------------------------------------------


def test_setup_is_offered_when_it_has_never_been_seen(window, existing_install, dialogs):
    assert window.offer_lab_setup_once(existing_install) is True
    assert len(dialogs.opened) == 1


def test_setup_is_not_offered_once_it_has_been_seen(window, existing_install, dialogs):
    existing_install.setValue(LAB_SETUP_COMPLETE_KEY, True)

    assert window.offer_lab_setup_once(existing_install) is False
    assert dialogs.opened == []


def test_setup_is_not_offered_while_the_welcome_is_still_up(window, settings, dialogs):
    """The launch-time offer fires from a timer inside the welcome's own nested
    event loop, so an ungated one would land on top of it."""
    assert lab_setup_complete(settings) is False

    assert window.offer_lab_setup_once(settings) is False
    assert dialogs.opened == []


def test_setup_is_not_offered_while_a_walkthrough_is_on_screen(window, existing_install, dialogs):
    """A modal form covers the very widget the spotlight is pointing at."""
    window._active_tour = object()

    assert window.offer_lab_setup_once(existing_install) is False
    assert dialogs.opened == []


def test_setup_does_not_require_the_walkthrough_to_have_been_taken(window, settings, dialogs):
    """Skipping the welcome never sets first_run/tour_complete.

    Gating on that flag would leave everyone who declined the tour -- one of the
    two populations this offer exists to reach -- permanently unasked.
    """
    settings.setValue(FIRST_RUN_COMPLETE_KEY, True)
    assert settings.value(TOUR_COMPLETE_KEY, False, type=bool) is False

    assert window.offer_lab_setup_once(settings) is True
    assert len(dialogs.opened) == 1


def test_skipping_setup_does_not_re_offer_it(window, existing_install, dialogs):
    """The one way this feature turns into a nag.

    Skip closes the dialog with ``Rejected``. If "seen" were recorded only on
    ``Accepted``, every launch would ask again until the user typed something
    -- anything -- to stop it, which is precisely the junk data the vocabulary
    exists to prevent.
    """
    dialogs.result = QDialog.DialogCode.Rejected.value

    assert window.offer_lab_setup_once(existing_install) is True
    assert lab_setup_complete(existing_install) is True

    assert window.offer_lab_setup_once(existing_install) is False
    assert len(dialogs.opened) == 1, "skipping re-asked on the next launch"


def test_completing_setup_does_not_re_offer_it(window, existing_install, dialogs):
    dialogs.result = QDialog.DialogCode.Accepted.value

    assert window.offer_lab_setup_once(existing_install) is True
    assert window.offer_lab_setup_once(existing_install) is False
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
    """A fresh install turns the launch-time offer away while the welcome is up,
    so something has to ask again once the walkthrough is done."""
    import glider.gui.onboarding as onboarding

    fake = _FakeTour()
    monkeypatch.setattr(onboarding, "start_tour", lambda host: fake)
    calls = []
    monkeypatch.setattr(window, "offer_lab_setup_once", lambda *a, **k: calls.append(True))

    window._start_tour()
    assert calls == [], "asked while the walkthrough was still on screen"

    fake.finished.emit()
    qtbot.waitUntil(lambda: calls == [True], timeout=1000)


def _welcome_answering(choice: str):
    """A stand-in welcome dialog that answers ``choice``.

    It pumps the event loop, because the real ``QMessageBox.exec`` does: the
    launch-time lab-setup offer is queued during ``MainWindow.__init__`` and
    comes due *inside* that nested loop, where the first-run gate turns it
    away and spends it. A fake that skips the pumping leaves the timer
    unfired, and the test then passes on a path no user takes.
    """

    def _welcome(parent, data_dir, *, offer_tour=False):
        QApplication.processEvents()
        return choice

    return _welcome


# --- At launch, on a real window -----------------------------------------------


def test_an_existing_install_is_offered_setup_at_launch(
    qtbot, main_window_factory, existing_install, dialogs
):
    """The case the whole feature exists for.

    A lab already running GLIDER finished the walkthrough long ago, so the
    tour-finished path will never fire for them again -- and they are exactly
    the people who reported the subject fields being invisible and went back to
    spreadsheets.
    """
    main_window_factory(settings=existing_install)

    qtbot.waitUntil(lambda: len(dialogs.opened) == 1, timeout=3000)
    assert lab_setup_complete(existing_install) is True


def test_the_launch_after_a_skipped_welcome_offers_setup(
    qtbot, main_window_factory, settings, dialogs
):
    """The launch *after* the welcome was answered without taking the tour.

    Pre-setting first_run/complete is what makes this the second launch: on the
    first one the welcome is still up and the offer is turned away. Declining
    the tour leaves first_run/tour_complete unset forever, so nothing on this
    or any later launch would ask if the offer were gated on the walkthrough.
    The same launch they skipped on is covered by
    ``test_a_fresh_install_that_declines_the_tour_is_still_offered_setup``.
    """
    settings.setValue(FIRST_RUN_COMPLETE_KEY, True)

    main_window_factory(settings=settings)

    qtbot.waitUntil(lambda: len(dialogs.opened) == 1, timeout=3000)


def test_runner_mode_is_never_offered_setup(qtbot, main_window_factory, existing_install, dialogs):
    """The Pi runner is a 480px touch surface with no menu bar; a modal form of
    five editable lists is the wrong shape there, and there is no way to dismiss
    it into a menu that would let it be reopened."""
    win = main_window_factory(desktop_mode=False, settings=existing_install)

    assert win.offer_lab_setup_once(existing_install) is False
    qtbot.wait(100)
    assert dialogs.opened == []
    assert lab_setup_complete(existing_install) is False, "the flag was burned unseen"


def test_setup_is_not_offered_twice_in_one_session(
    qtbot, main_window_factory, existing_install, dialogs
):
    """Launch offers it, then the user replays the tutorial: still once."""
    win = main_window_factory(settings=existing_install)
    qtbot.waitUntil(lambda: len(dialogs.opened) == 1, timeout=3000)

    win._on_tour_finished()
    qtbot.wait(100)

    assert len(dialogs.opened) == 1


def test_a_fresh_install_gets_welcome_then_tour_then_setup(
    qtbot, main_window_factory, settings, dialogs, monkeypatch, tmp_path
):
    """The ordering that has to hold on a genuinely fresh install.

    The launch-time offer is queued during ``MainWindow.__init__``, so it comes
    due inside the welcome dialog's nested event loop -- before the walkthrough
    has even started. Setup must appear after both, exactly once.
    """
    import glider.first_run as first_run_mod
    import glider.gui.onboarding as onboarding

    order: list[str] = []
    fake = _FakeTour()

    monkeypatch.setattr(first_run_mod, "ensure_data_dir", lambda path=None: tmp_path)

    def _fake_welcome(parent, data_dir, *, offer_tour=False):
        order.append("welcome")
        assert offer_tour, "a fresh Builder launch should offer the tour"
        QApplication.processEvents()  # what the real box.exec() nested loop does
        assert dialogs.opened == [], "lab setup opened on top of the welcome dialog"
        return "tour"

    def _fake_start_tour(host):
        order.append("tour")
        host._active_tour = fake  # the overlay is now dimming the window
        return fake

    monkeypatch.setattr(first_run_mod, "show_welcome_dialog", _fake_welcome)
    monkeypatch.setattr(onboarding, "start_tour", _fake_start_tour)

    win = main_window_factory(settings=settings)
    first_run_mod.run_first_run_if_needed(win, settings=settings)

    qtbot.wait(100)
    assert order == ["welcome", "tour"]
    assert dialogs.opened == [], "lab setup opened while the walkthrough was on screen"

    win._active_tour = None  # the user finished or skipped it
    fake.finished.emit()
    qtbot.waitUntil(lambda: len(dialogs.opened) == 1, timeout=3000)
    order.append("setup")

    assert order == ["welcome", "tour", "setup"]

    win._on_tour_finished()
    qtbot.wait(100)
    assert len(dialogs.opened) == 1, "offered twice on one install"


def test_a_fresh_install_that_declines_the_tour_is_still_offered_setup(
    qtbot, main_window_factory, settings, dialogs, monkeypatch, tmp_path
):
    """The path that reached nobody: a fresh install that skips the tour.

    The launch-time offer is queued in ``MainWindow.__init__`` and comes due
    inside the welcome dialog's own nested event loop, where the first-run gate
    correctly turns it away -- and that ``singleShot`` is then spent. Only the
    tour-finished path asked again, so answering the welcome with anything but
    **Take the Tour** meant no offer at all that session. The person missed is
    the new lab member who skips things, which is exactly who the vocabulary
    exists for.
    """
    import glider.first_run as first_run_mod

    monkeypatch.setattr(first_run_mod, "ensure_data_dir", lambda path=None: tmp_path)

    monkeypatch.setattr(first_run_mod, "show_welcome_dialog", _welcome_answering("start"))

    win = main_window_factory(settings=settings)
    first_run_mod.run_first_run_if_needed(win, settings=settings)

    qtbot.waitUntil(lambda: len(dialogs.opened) == 1, timeout=3000)
    assert lab_setup_complete(settings) is True


def test_a_walkthrough_that_fails_to_start_still_leads_to_setup(
    qtbot, main_window_factory, settings, dialogs, monkeypatch, tmp_path
):
    """The tour is a nice-to-have and its failure is swallowed. If the offer
    rode on the tour alone, that swallowed failure would silently take the
    lab vocabulary down with it."""
    import glider.first_run as first_run_mod

    monkeypatch.setattr(first_run_mod, "ensure_data_dir", lambda path=None: tmp_path)
    monkeypatch.setattr(first_run_mod, "show_welcome_dialog", _welcome_answering("tour"))

    win = main_window_factory(settings=settings)
    monkeypatch.setattr(win, "_start_tour", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    first_run_mod.run_first_run_if_needed(win, settings=settings)

    qtbot.waitUntil(lambda: len(dialogs.opened) == 1, timeout=3000)


def test_a_started_walkthrough_does_not_also_get_the_declined_offer(
    qtbot, main_window_factory, settings, dialogs, monkeypatch, tmp_path
):
    """Taking the tour must still wait for it, not be offered underneath it."""
    import glider.first_run as first_run_mod
    import glider.gui.onboarding as onboarding

    fake = _FakeTour()
    monkeypatch.setattr(first_run_mod, "ensure_data_dir", lambda path=None: tmp_path)
    monkeypatch.setattr(first_run_mod, "show_welcome_dialog", _welcome_answering("tour"))

    def _fake_start_tour(host):
        host._active_tour = fake
        return fake

    monkeypatch.setattr(onboarding, "start_tour", _fake_start_tour)

    win = main_window_factory(settings=settings)
    first_run_mod.run_first_run_if_needed(win, settings=settings)

    qtbot.wait(100)
    assert dialogs.opened == [], "offered while the walkthrough was on screen"


# --- Re-entry from the menu ----------------------------------------------------


def test_the_experiment_menu_offers_lab_setup(window):
    action = _lab_setup_action(window)

    assert action is not None, "Experiment -> Lab Setup... missing"
    assert action.isEnabled() is True


def test_the_menu_action_opens_the_dialog(window, dialogs):
    _lab_setup_action(window).trigger()

    assert len(dialogs.opened) == 1


def test_the_menu_action_still_works_after_first_run_is_complete(window, existing_install, dialogs):
    """Re-entry is the whole point: the knowledge usually arrives later."""
    existing_install.setValue(LAB_SETUP_COMPLETE_KEY, True)

    _lab_setup_action(window).trigger()

    assert len(dialogs.opened) == 1
