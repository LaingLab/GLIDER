"""Add Images: pick videos, how many frames from each, and how to choose them."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from glider.gui.animal_annotator.style import muted, secondary, studio_stylesheet
from glider.vision.training.project import StudioProject

__all__ = ["AddImagesDialog"]

_VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v);;All files (*)"


class AddImagesDialog(QDialog):
    """Videos to pull frames from: the project's own (tick to top up) and new ones."""

    def __init__(self, project: StudioProject, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Images")
        self.setStyleSheet(studio_stylesheet())
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(10)
        title = QLabel("Add images")
        title.setProperty("studioTitle", True)
        outer.addWidget(title)
        outer.addWidget(
            muted(
                QLabel(
                    "Several sessions, lighting conditions and animals make a model that "
                    "works on the next recording, not just these."
                ),
                small=True,
            )
        )
        outer.itemAt(outer.count() - 1).widget().setWordWrap(True)

        self._list = QListWidget()
        self._list.setProperty("compact", True)
        self._list.itemChanged.connect(self._update_ok)
        for path in project.videos.values():
            self._add_item(Path(path), Qt.CheckState.Unchecked)
        outer.addWidget(self._list, 1)
        browse = secondary(QPushButton("Add Videos…"))
        browse.clicked.connect(self._browse)
        outer.addWidget(browse, 0, Qt.AlignmentFlag.AlignLeft)

        form = QFormLayout()
        self._per_video = QSpinBox()
        self._per_video.setRange(1, 2000)
        self._per_video.setValue(20)
        self._per_video.setToolTip("Frames added from each ticked video")
        form.addRow("Frames per video", self._per_video)
        self._method = QComboBox()
        self._method.addItem("Most varied", "diverse")
        self._method.addItem("Evenly spaced", "uniform")
        self._method.setToolTip(
            "Most varied skips near-identical frames, such as a mouse sitting still"
        )
        form.addRow("Choose frames", self._method)
        outer.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setFlat(True)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self._ok = QPushButton("Extract")
        self._ok.setDefault(True)
        self._ok.clicked.connect(self.accept)
        buttons.addWidget(self._ok)
        outer.addLayout(buttons)
        self._update_ok()
        self.resize(520, 420)

    @property
    def videos(self) -> list[Path]:
        return [
            Path(item.data(Qt.ItemDataRole.UserRole))
            for item in (self._list.item(i) for i in range(self._list.count()))
            if item.checkState() == Qt.CheckState.Checked
        ]

    @property
    def per_video(self) -> int:
        return self._per_video.value()

    @property
    def method(self) -> str:
        return self._method.currentData()

    def add_paths(self, paths: list[str]) -> None:
        """Newly picked files, ticked; ones already listed are left as they are."""
        known = {
            Path(self._list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._list.count())
        }
        for p in paths:
            path = Path(p).resolve()
            if path not in known and Path(p) not in known:
                known.add(path)
                self._add_item(path, Qt.CheckState.Checked)

    def _add_item(self, path: Path, state: Qt.CheckState) -> None:
        item = QListWidgetItem(path.name)
        item.setToolTip(str(path))
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(state)
        self._list.addItem(item)

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos", "", _VIDEO_FILTER)
        self.add_paths(paths)

    def _update_ok(self, *_) -> None:
        self._ok.setEnabled(bool(self.videos))
