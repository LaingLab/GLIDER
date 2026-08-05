"""The Review tab, and the path a finished run takes to reach it."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QLabel  # noqa: E402

from glider.analysis.behavior.run_report import TrainingRun  # noqa: E402
from glider.gui.behavior.review_tab import ReviewTab  # noqa: E402

CLASSES = ["groom", "locomote", "rear"]


def _summary(**overrides):
    base = {
        "n_sessions": 4,
        "n_rows_kept": 10_000,
        "kept_label_counts": {"locomote": 6_000, "groom": 3_000, "rear": 1_000},
        "test_size": 2_000,
        "train_accuracy": 0.99,
        "test_accuracy": 0.81,
        "split_strategy": "cross_session",
        "n_holdout_sessions": 2,
        "classifier_type": "lightgbm",
        "per_class_metrics": {
            "groom": {"precision": 0.9, "recall": 0.88, "f1": 0.89, "support": 900},
            "rear": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 300},
        },
        "confusion_matrix": {"labels": CLASSES, "matrix": [[8, 1, 1], [1, 8, 1], [3, 3, 4]]},
        "top_features": [{"feature": "speed_mean", "importance": 0.4}],
        "window": 30,
        "fps": 30.0,
        "n_features": 400,
    }
    base.update(overrides)
    return base


def _report(tmp_path, summary=None, name="m_report"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps(summary or _summary()))
    return folder


class TestReviewTab:
    def test_an_empty_tab_paints(self, qtbot):
        tab = ReviewTab()
        qtbot.addWidget(tab)
        tab.resize(700, 500)
        tab.grab()  # forces a paint

    def test_loading_a_report_builds_the_cards(self, qtbot, tmp_path):
        tab = ReviewTab()
        qtbot.addWidget(tab)
        assert tab.load(_report(tmp_path)) is True
        assert tab._cards  # noqa: SLF001 - the built content is the assertion
        tab.resize(900, 900)
        tab.grab()

    def test_loading_a_second_run_replaces_the_first(self, qtbot, tmp_path):
        """A stale card from the previous run would misreport the current one."""
        tab = ReviewTab()
        qtbot.addWidget(tab)
        tab.load(_report(tmp_path, name="one"))
        first = len(tab._cards)  # noqa: SLF001
        tab.load(_report(tmp_path, _summary(n_sessions=9), name="two"))
        assert len(tab._cards) == first  # noqa: SLF001
        assert tab._run.summary["n_sessions"] == 9  # noqa: SLF001

    def test_a_bad_folder_is_refused_without_raising(self, qtbot, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "glider.gui.behavior.review_tab.QMessageBox.warning",
            lambda *a, **k: None,
        )
        tab = ReviewTab()
        qtbot.addWidget(tab)
        empty = tmp_path / "empty"
        empty.mkdir()
        assert tab.load(empty) is False

    def test_a_run_with_no_metrics_still_shows_something(self, qtbot, tmp_path):
        """A no-holdout run has no per-class table; it must not render blank."""
        folder = _report(
            tmp_path,
            _summary(
                per_class_metrics={},
                confusion_matrix={},
                test_size=0,
                test_accuracy=None,
                split_strategy="no_holdout",
            ),
        )
        tab = ReviewTab()
        qtbot.addWidget(tab)
        assert tab.load(folder) is True
        tab.resize(900, 700)
        tab.grab()

    def test_an_in_memory_run_needs_no_folder(self, qtbot):
        tab = ReviewTab()
        qtbot.addWidget(tab)
        tab.show_run(TrainingRun.from_summary(_summary()))
        assert tab._cards  # noqa: SLF001


class TestConfusionWidget:
    def test_it_paints_a_matrix(self, qtbot):
        from glider.gui.behavior.review_tab import ConfusionMatrix

        grid = ConfusionMatrix()
        qtbot.addWidget(grid)
        grid.set_matrix(CLASSES, [[0.8, 0.1, 0.1]] * 3, [[8, 1, 1]] * 3)
        grid.resize(400, 200)
        grid.grab()

    def test_an_empty_matrix_does_not_crash(self, qtbot):
        from glider.gui.behavior.review_tab import ConfusionMatrix

        grid = ConfusionMatrix()
        qtbot.addWidget(grid)
        grid.set_matrix([], [], [])
        grid.resize(200, 100)
        grid.grab()


class TestWindowWiring:
    def test_the_review_tab_sits_between_train_and_apply(self, qtbot, tmp_path):
        from glider.gui.behavior.window import BehaviorAnalysisWindow

        win = BehaviorAnalysisWindow(project_dir=tmp_path)
        qtbot.addWidget(win)
        titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        assert titles == ["Annotate", "Train", "Review", "Apply"]

    def test_a_finished_run_opens_its_report_and_raises_the_tab(self, qtbot, tmp_path):
        from glider.gui.behavior.window import BehaviorAnalysisWindow

        win = BehaviorAnalysisWindow(project_dir=tmp_path)
        qtbot.addWidget(win)
        win._on_run_reported(_report(tmp_path), _summary())  # noqa: SLF001
        assert win.tabs.currentWidget() is win.review_tab
        assert win.review_tab._run is not None  # noqa: SLF001

    def test_a_run_with_no_report_falls_back_to_the_summary(self, qtbot, tmp_path):
        """A read-only output folder should cost the charts, not the review."""
        from glider.gui.behavior.window import BehaviorAnalysisWindow

        win = BehaviorAnalysisWindow(project_dir=tmp_path)
        qtbot.addWidget(win)
        win._on_run_reported(None, _summary())  # noqa: SLF001
        assert win.tabs.currentWidget() is win.review_tab
        assert win.review_tab._run.summary["n_sessions"] == 4  # noqa: SLF001

    def test_nothing_to_show_leaves_the_tab_alone(self, qtbot, tmp_path):
        from glider.gui.behavior.window import BehaviorAnalysisWindow

        win = BehaviorAnalysisWindow(project_dir=tmp_path)
        qtbot.addWidget(win)
        win.tabs.setCurrentIndex(1)
        win._on_run_reported(None, None)  # noqa: SLF001
        assert win.tabs.currentIndex() == 1

    def test_the_train_tab_announces_a_finished_fit(self, qtbot, tmp_path):
        """TrainTab must not reach for the Review tab itself."""
        from glider.gui.behavior.window import TrainTab

        tab = TrainTab()
        qtbot.addWidget(tab)
        seen = []
        tab.run_reported.connect(lambda report, summary: seen.append((report, summary)))
        tab._on_report_ready(tmp_path / "r")  # noqa: SLF001
        tab._on_train_finished({"n_sessions": 2})  # noqa: SLF001
        assert seen == [(tmp_path / "r", {"n_sessions": 2})]

    def test_a_failed_run_does_not_leave_a_stale_report_behind(self, qtbot, tmp_path, monkeypatch):
        """The next run's review must not show the previous run's numbers."""
        from glider.gui.behavior.window import TrainTab

        monkeypatch.setattr("glider.gui.behavior.window.QMessageBox.critical", lambda *a, **k: None)
        tab = TrainTab()
        qtbot.addWidget(tab)
        tab._on_report_ready(tmp_path / "old")  # noqa: SLF001
        tab._on_train_failed("boom")  # noqa: SLF001

        seen = []
        tab.run_reported.connect(lambda report, summary: seen.append(report))
        tab._on_train_finished({})  # noqa: SLF001
        assert seen == [None]


class TestLayout:
    """The report is a page, not a list. These pin the shape of it."""

    def test_the_verdict_leads_and_carries_its_warnings(self, qtbot, tmp_path):
        """Score, evidence and caveats are one block -- they must not separate."""
        from glider.gui.behavior.review_tab import Verdict

        tab = ReviewTab()
        qtbot.addWidget(tab)
        tab.load(
            _report(
                tmp_path,
                _summary(
                    split_strategy="no_holdout",
                    test_size=0,
                    test_accuracy=None,
                    kept_label_counts={"common": 50_000, "rare": 500},
                ),
            )
        )
        assert isinstance(tab._cards[0], Verdict)  # noqa: SLF001
        shown = tab._cards[0].findChildren(QLabel)  # noqa: SLF001
        text = " ".join(lab.text() for lab in shown)
        assert "No held-out data" in text
        # The eyebrow is uppercased for display; the label is what matters.
        assert "TRAIN ACCURACY" in text.upper()

    def test_the_panels_are_paired_into_a_grid(self, qtbot, tmp_path):
        """One full-width column per panel is what made this read as a list."""
        from glider.gui.widgets.tool_ui import CardGrid

        tab = ReviewTab()
        qtbot.addWidget(tab)
        tab.load(_report(tmp_path))
        assert any(isinstance(c, CardGrid) for c in tab._cards)  # noqa: SLF001

    def test_bars_share_one_width_so_rows_are_comparable(self, qtbot):
        """A bar that stretches to fill its panel encodes the panel, not the value."""
        from glider.gui.behavior.review_tab import ScoreBar

        wide = ScoreBar(0.9, "#38bdf8")
        narrow = ScoreBar(0.9, "#38bdf8")
        qtbot.addWidget(wide)
        qtbot.addWidget(narrow)
        wide.resize(900, 16)
        assert wide.width() == narrow.width() == ScoreBar.WIDTH

    def test_a_card_badge_never_grows_into_a_column(self, qtbot, tmp_path):
        """A stretched card handed its slack to the header and inflated the pill."""
        tab = ReviewTab()
        qtbot.addWidget(tab)
        tab.load(_report(tmp_path))
        tab.resize(1100, 1400)
        tab.show()
        badges = [
            lab
            for lab in tab.findChildren(QLabel)
            if lab.objectName() == "CardBadge" and lab.isVisible()
        ]
        assert badges, "expected at least one badge"
        for badge in badges:
            assert badge.height() <= 30, f"{badge.text()!r} grew to {badge.height()}px"

    def test_the_split_strategy_is_said_in_words(self, qtbot, tmp_path):
        """'no_holdout' is a field name, not an answer."""
        tab = ReviewTab()
        qtbot.addWidget(tab)
        tab.load(_report(tmp_path, _summary(split_strategy="no_holdout", test_size=0)))
        text = " ".join(lab.text() for lab in tab.findChildren(QLabel))
        assert "no_holdout" not in text
        assert "fitted on everything" in text
