"""The golden-path walkthrough, on the page it actually starts from."""

from __future__ import annotations


def test_the_tour_reaches_targets_on_a_page_that_is_not_showing(qtbot, main_window_factory):
    """Regression: the walkthrough starts on the landing page.

    ``_raise_tabs`` handled a QStackedWidget owned by a QTabWidget but skipped a
    bare one -- and the window's pages are a bare stack. So on a first run every
    target was on a page that was not showing (measured: all seven), and each
    step rendered as a centred caption with nothing highlighted. A tour of
    nothing, on the one path where a tour is the point.
    """
    from glider.gui.onboarding.tour import Tour, golden_path_steps

    window = main_window_factory(desktop_mode=True)
    window.show()
    assert window.is_on_landing(), "the fixture no longer starts where users do"

    steps = golden_path_steps()
    tour = Tour(window, steps)
    tour.start()

    hidden = []
    for step in steps:
        if step.target_key:
            target = window.tour_targets().get(step.target_key)
            # None is the documented case for a target this mode never builds.
            if target is not None and not target.isVisible():
                hidden.append(step.target_key)
        tour._next()

    assert not hidden, f"spotlighting widgets nobody can see: {hidden}"
