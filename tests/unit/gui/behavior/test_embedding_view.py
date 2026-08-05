"""The 3D embedding: requesting one, and rendering it in the Review tab.

The artifact stores only ``coords (M, 3)`` and ``labels``. Nothing in the GUI
could ask for one, and nothing could draw one, so a bundle that carried an
embedding had no way to show it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from glider.gui.behavior.embedding_view import EmbeddingView  # noqa: E402


def _artifact(n=300, seed=0):
    """Three well-separated clusters, so structure is obvious when drawn."""
    from glider.analysis.behavior.embedding import EmbeddingArtifact

    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 4.0]])
    names = ["walk", "groom", "still"]
    coords = np.vstack([c + rng.normal(0, 0.7, (n // 3, 3)) for c in centres])
    labels = np.array([names[i // (n // 3)] for i in range(len(coords))])
    return EmbeddingArtifact(
        method="pca",
        scaler=None,
        reducer=None,
        coords=coords,
        labels=labels,
        feature_names=["a", "b", "c"],
    )


# ---------------------------------------------------------------------------
# Asking for one
# ---------------------------------------------------------------------------


def test_the_train_tab_offers_the_embedding_choices(qtbot):
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    choices = {tab._embedding_combo.itemData(i) for i in range(tab._embedding_combo.count())}
    assert choices == {"none", "umap", "pca"}


def test_no_embedding_is_requested_by_default(qtbot):
    """Fitting one costs real time on a large cohort."""
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    assert tab._shared_options()["embedding"] == "none"


def test_the_chosen_method_reaches_the_pipeline(qtbot):
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    tab._embedding_combo.setCurrentIndex(tab._embedding_combo.findData("umap"))
    assert tab._shared_options()["embedding"] == "umap"


def test_both_entry_points_accept_the_option(qtbot):
    """Fit and Cross-validate share _shared_options, so both must take it."""
    import inspect

    from glider.analysis.behavior.pipeline import cross_validate_and_train, train_model

    for fn in (train_model, cross_validate_and_train):
        assert "embedding" in inspect.signature(fn).parameters


# ---------------------------------------------------------------------------
# Drawing one
# ---------------------------------------------------------------------------


def test_it_starts_empty(qtbot):
    view = EmbeddingView()
    qtbot.addWidget(view)
    assert view.has_data() is False


def test_it_accepts_an_artifact(qtbot):
    view = EmbeddingView()
    qtbot.addWidget(view)
    view.set_artifact(_artifact())
    assert view.has_data() is True
    assert set(view.class_names()) == {"walk", "groom", "still"}


def test_every_class_gets_its_own_colour(qtbot):
    """Colour is the only channel separating clusters; a collision hides one."""
    view = EmbeddingView()
    qtbot.addWidget(view)
    view.set_artifact(_artifact())
    colours = [view.color_for(name).name() for name in view.class_names()]
    assert len(set(colours)) == len(colours)


def test_the_same_class_keeps_its_colour_across_renders(qtbot):
    """Otherwise the legend means something different every repaint."""
    view = EmbeddingView()
    qtbot.addWidget(view)
    view.set_artifact(_artifact())
    first = view.color_for("groom").name()
    view.set_artifact(_artifact(seed=5))
    assert view.color_for("groom").name() == first


@pytest.mark.parametrize(
    "prepare",
    [
        lambda v: None,
        lambda v: v.set_artifact(_artifact()),
        lambda v: v.set_artifact(_artifact(n=3)),
        # A degenerate cloud: every point identical, so the extent is zero and
        # a naive scale-to-fit divides by it.
        lambda v: v.set_artifact(_degenerate()),
        # One class only — the legend and the colour cycle must still cope.
        lambda v: v.set_artifact(_single_class()),
    ],
)
def test_it_paints_without_raising(qtbot, prepare):
    from PyQt6.QtGui import QPixmap

    view = EmbeddingView()
    qtbot.addWidget(view)
    view.resize(420, 320)
    prepare(view)
    view.render(QPixmap(view.size()))


def _degenerate():
    from glider.analysis.behavior.embedding import EmbeddingArtifact

    return EmbeddingArtifact(
        method="pca",
        scaler=None,
        reducer=None,
        coords=np.zeros((20, 3)),
        labels=np.array(["x"] * 20),
        feature_names=["a"],
    )


def _single_class():
    from glider.analysis.behavior.embedding import EmbeddingArtifact

    rng = np.random.default_rng(1)
    return EmbeddingArtifact(
        method="pca",
        scaler=None,
        reducer=None,
        coords=rng.normal(0, 1, (50, 3)),
        labels=np.array(["only"] * 50),
        feature_names=["a"],
    )


def test_dragging_rotates_the_view(qtbot):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    view = EmbeddingView()
    qtbot.addWidget(view)
    view.resize(400, 300)
    view.set_artifact(_artifact())
    before = view.rotation()

    def _event(kind, pos):
        return QMouseEvent(
            kind,
            QPointF(pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    view.mousePressEvent(_event(QMouseEvent.Type.MouseButtonPress, QPoint(100, 100)))
    view.mouseMoveEvent(_event(QMouseEvent.Type.MouseMove, QPoint(160, 130)))

    assert view.rotation() != before


def test_points_are_capped_so_a_huge_cloud_stays_interactive(qtbot):
    """20k points repainted on every drag frame would make rotation crawl."""
    from glider.analysis.behavior.embedding import EmbeddingArtifact

    rng = np.random.default_rng(2)
    big = EmbeddingArtifact(
        method="pca",
        scaler=None,
        reducer=None,
        coords=rng.normal(0, 1, (20_000, 3)),
        labels=np.array(["a"] * 20_000),
        feature_names=["f"],
    )
    view = EmbeddingView()
    qtbot.addWidget(view)
    view.set_artifact(big)
    assert view.drawn_point_count() <= EmbeddingView.MAX_POINTS
    assert view.drawn_point_count() > 0
