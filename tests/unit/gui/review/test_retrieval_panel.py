"""The Retrieval tab on its own: setup lines, the pup table, and its signals."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from glider.analysis.behavior.retrieval import PupRetrieval  # noqa: E402
from glider.analysis.behavior.retrieval_setup import (  # noqa: E402
    RetrievalProtocol,
    RetrievalSetup,
)
from glider.gui.review.retrieval_panel import RetrievalPanel  # noqa: E402
from glider.vision.zones import Zone, ZoneShape  # noqa: E402


def _panel(qtbot):
    panel = RetrievalPanel()
    qtbot.addWidget(panel)
    return panel


def _result(trial_start_frame=0, fps=15.0):
    r = PupRetrieval(False, 8.0, 11.5, 1, 0, 9.0, np.zeros(3, bool), np.zeros(3, bool))
    return type(
        "R",
        (),
        {
            "pups": {"pup_0": r},
            "litter": {
                "pups_scattered": 1,
                "pups_retrieved": 1,
                "time_all_retrieved_s": 11.5,
                "retrieval_order": {"pup_0": 1},
            },
            "dam_in_nest_frac": 0.41,
            "t0": trial_start_frame,
            "fps": fps,
        },
    )()


def test_an_unset_trial_start_says_so(qtbot):
    panel = _panel(qtbot)
    panel.show_setup(RetrievalSetup(), RetrievalProtocol(), writable=True, n_pups=5)
    panel.show_result(None, "Set trial start to score")
    assert "not set" in panel.trial_label.text().lower()
    assert panel.status.text() == "Set trial start to score"
    assert panel.table.rowCount() == 0


def test_nest_periods_are_listed_and_clickable(qtbot):
    panel = _panel(qtbot)
    setup = RetrievalSetup(trial_start_s=0.0)
    zone = Zone("n", "nest", ZoneShape.RECTANGLE, [(0.1, 0.1), (0.3, 0.3)])
    setup.set_nest(0.0, zone)
    setup.set_nest(252.0, zone)
    panel.show_setup(setup, RetrievalProtocol(), writable=True, n_pups=5)
    assert [panel.nest_list.item(i).text() for i in range(2)] == [
        "Nest 1 · from 0:00",
        "Nest 2 · from 4:12",
    ]
    with qtbot.waitSignal(panel.jump_to) as sig:
        panel.nest_list.itemClicked.emit(panel.nest_list.item(1))
    assert sig.args == [252.0]


def test_a_read_only_setup_disables_editing(qtbot):
    panel = _panel(qtbot)
    panel.show_setup(
        RetrievalSetup(),
        RetrievalProtocol(),
        writable=False,
        error="v.retrieval.json is not valid JSON",
    )
    assert "not valid JSON" in panel.status.text()
    assert not panel.trial_button.isEnabled()


def test_the_read_only_reason_outlives_the_score_reason(qtbot):
    # Otherwise "Draw the nest to score" hides why the buttons are greyed out.
    panel = _panel(qtbot)
    panel.show_setup(RetrievalSetup(), RetrievalProtocol(), writable=False, error="bad JSON")
    panel.show_result(None, "Draw the nest to score")
    assert "bad JSON" in panel.status.text()
    assert "Draw the nest" in panel.status.text()


def test_results_fill_the_table(qtbot):
    panel = _panel(qtbot)
    panel.show_result(_result())
    assert panel.table.rowCount() == 1
    assert [panel.table.item(0, c).text() for c in range(5)] == ["pup_0", "8.0", "11.5", "1", "0"]
    assert "1 of 1 retrieved" in panel.summary.text()


def test_clicking_a_pup_jumps_to_its_retrieval(qtbot):
    panel = _panel(qtbot)
    panel.show_result(_result(trial_start_frame=30, fps=15.0))  # trial starts at 2 s
    with qtbot.waitSignal(panel.jump_to) as sig:
        panel.table.cellClicked.emit(0, 1)
    assert sig.args == [pytest.approx(13.5)]


def test_the_buttons_emit_their_signals(qtbot):
    panel = _panel(qtbot)
    panel.show_setup(RetrievalSetup(), RetrievalProtocol(), writable=True)
    with qtbot.waitSignal(panel.set_trial_start):
        panel.trial_button.click()
    with qtbot.waitSignal(panel.redraw_nest):
        panel.nest_button.click()


# What Session Review's top splitter asks the inspector for (setSizes([250, 860, 330])).
# A 1440 px window with the Pups toggle squeezes it to ~300, where the table has
# ~20 px to spare on macOS fonts; the extra here is for the wider fonts of CI's
# Windows and Linux runners, whose text is a few percent wider.
_INSPECTOR_W = 330


def _fitted_panel(qtbot):
    from glider.gui.widgets.tool_ui import apply_tool_theme

    panel = _panel(qtbot)
    apply_tool_theme(panel)  # the window's own theme: its widths are what matter
    setup = RetrievalSetup(trial_start_s=3.0, px_per_mm=1.0)
    setup.set_nest(0.0, Zone("n", "nest", ZoneShape.RECTANGLE, [(0.1, 0.1), (0.3, 0.3)]))
    panel.show_setup(setup, RetrievalProtocol(), writable=True, n_pups=5)
    panel.show_result(_result())
    panel.resize(_INSPECTOR_W, 400)  # short enough that the vertical scrollbar appears
    panel.show()
    qtbot.waitExposed(panel)
    # Geometry measured before the layout settles is a different panel: the
    # table's width, and so whether it fits, still changes for a few events.
    last = None
    for _ in range(50):
        QApplication.processEvents()
        now = (panel.table.viewport().width(), panel.table.horizontalHeader().length())
        if now == last:
            break
        last = now
    return panel


def _skip_without_a_font(panel):
    """Text widths mean nothing where the platform has no font to measure.

    Windows CI's offscreen platform reports an empty family and a -1 px size
    and lays text out 1.6-2.7x wider than any real font; these two tests ask
    whether real text fits, so they skip there (as the shell tests do).
    """
    info = panel.trial_button.fontInfo()
    if not info.family() or info.pixelSize() <= 0:
        pytest.skip("no font on this platform: text widths are meaningless")


def _widths(panel, area) -> str:
    """What made a row too wide, for a failure on a platform we cannot run."""
    from PyQt6.QtWidgets import QFormLayout

    parts = [
        f"viewport={area.viewport().width()}",
        f"content_min={area.widget().minimumSizeHint().width()}",
        f"font={panel.trial_button.fontInfo().family()}/{panel.trial_button.fontInfo().pixelSize()}px",
    ]
    column = area.widget().layout()
    for i in range(column.count()):
        item = column.itemAt(i)
        if isinstance(item.layout(), QFormLayout):
            form = item.layout()
            for r in range(form.rowCount()):
                for role in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole):
                    cell = form.itemAt(r, role)
                    if cell is not None:
                        parts.append(f"form[{r}].{role.name}={cell.minimumSize().width()}")
        elif item.widget() is not None:
            parts.append(
                f"{type(item.widget()).__name__}={item.widget().minimumSizeHint().width()}"
            )
        elif item.layout() is not None:
            parts.append(f"layout={item.layout().minimumSize().width()}")
    return " ".join(parts)


def test_the_tab_fits_the_default_inspector_without_clipping(qtbot):
    """Found by rendering Session Review: rows ran off the pane's edge.

    The column's horizontal scrollbar is off, so a row wider than the pane was
    simply cut: the Remove button sat half off the edge and the pup table's
    last columns could only be reached by scrolling the table itself.
    """
    panel = _fitted_panel(qtbot)
    _skip_without_a_font(panel)
    area = panel.findChild(QScrollArea)
    assert not area.horizontalScrollBar().isVisible(), _widths(panel, area)
    for button in (
        panel.nest_button,
        panel.remove_button,
        panel.trial_button,
        panel.scale_button,
        panel.protocol_button,
    ):
        right = button.mapTo(area.viewport(), QPoint(button.width(), 0)).x()
        assert right <= area.viewport().width(), f"{button.text()} runs off the pane"
    assert panel.table.horizontalHeader().length() <= panel.table.viewport().width()
    assert not panel.table.horizontalScrollBar().isVisible(), "pup table columns are cut off"


def test_remove_sits_under_the_nest_row_not_beside_the_draw_button(qtbot):
    """Side by side they were wider than a 300 px inspector in any font."""
    panel = _fitted_panel(qtbot)
    draw = panel.nest_button.mapTo(panel, QPoint(0, 0)).y() + panel.nest_button.height()
    assert panel.remove_button.mapTo(panel, QPoint(0, 0)).y() >= draw


def test_the_thresholds_row_is_named_for_what_it_holds(qtbot):
    """The experiment's protocol is picked elsewhere; this row holds the thresholds."""
    from PyQt6.QtWidgets import QLabel

    panel = _panel(qtbot)
    assert panel.protocol_button.text() == "Thresholds…"
    texts = [label.text() for label in panel.findChildren(QLabel)]
    assert "Thresholds" in texts and "Protocol" not in texts


def test_setup_rows_wrap_rather_than_overflow_with_wider_text(qtbot):
    """Windows' text is wider than macOS's (CI caught it at 330 px): a setup
    row whose label, value and button no longer fit side by side must wrap
    under its label, never push past the pane's edge. Simulated here with a
    font about a third larger than the theme's."""
    panel = _fitted_panel(qtbot)
    _skip_without_a_font(panel)
    # The theme sizes text in its stylesheet, which outranks setFont.
    panel.setStyleSheet(panel.styleSheet() + "\nQLabel, QPushButton { font-size: 17px; }")
    for _ in range(20):
        QApplication.processEvents()
    area = panel.findChild(QScrollArea)
    assert not area.horizontalScrollBar().isVisible(), _widths(panel, area)
    for button in (panel.trial_button, panel.scale_button, panel.protocol_button):
        right = button.mapTo(area.viewport(), QPoint(button.width(), 0)).x()
        assert right <= area.viewport().width(), f"{button.text()} runs off the pane"


def test_setup_rows_are_a_form_that_wraps_long_rows(qtbot):
    """The overflow fix itself, checked on every platform (fonts or not): the
    setup rows live in a QFormLayout that wraps a field under its label."""
    from PyQt6.QtWidgets import QFormLayout

    panel = _panel(qtbot)
    [form] = panel.findChildren(QFormLayout)
    assert form.rowWrapPolicy() == QFormLayout.RowWrapPolicy.WrapLongRows
    fields = set()
    for row in range(form.rowCount()):
        cell = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
        layout = cell.layout() if cell is not None else None
        for i in range(layout.count() if layout is not None else 0):
            fields.add(layout.itemAt(i).widget())
    assert {panel.trial_button, panel.scale_button, panel.protocol_button} <= fields
