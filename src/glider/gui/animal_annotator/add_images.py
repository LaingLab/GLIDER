"""Add Images: image files copied in as they are, and videos to pull frames from."""

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
from glider.vision.training.project import IMAGE_SUFFIXES, VIDEO_SUFFIXES, StudioProject

__all__ = ["AddImagesDialog"]


def _filter(name: str, suffixes: tuple[str, ...]) -> str:
    return f"{name} ({' '.join('*' + s for s in suffixes)});;All files (*)"


_VIDEO_FILTER = _filter("Videos", VIDEO_SUFFIXES)
_IMAGE_FILTER = _filter("Images", IMAGE_SUFFIXES)
#: Item data: the file's kind, "video" or "image".
_KIND = Qt.ItemDataRole.UserRole + 1


class AddImagesDialog(QDialog):
    """Image files to copy in, and videos to pull frames from.

    The project's own videos are listed unticked, so ticking one tops it up.
    """

    def __init__(self, project: StudioProject, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Images or Videos")
        self.setStyleSheet(studio_stylesheet())
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(10)
        title = QLabel("Add images or videos")
        title.setProperty("studioTitle", True)
        outer.addWidget(title)
        hint = muted(
            QLabel(
                "Images are added as they are. From each video, the frames chosen below are "
                "pulled out. Several sessions, lighting conditions and animals make a model "
                "that works on the next recording, not just these."
            ),
            small=True,
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self._list = QListWidget()
        self._list.setProperty("compact", True)
        self._list.itemChanged.connect(self._update_ok)
        for key, path in project.videos.items():
            if not project.is_image_folder(key):
                self._add_item(Path(path), "video", Qt.CheckState.Unchecked)
        outer.addWidget(self._list, 1)
        pickers = QHBoxLayout()
        add_images = secondary(QPushButton("Add Images…"))
        add_images.setToolTip("Pick image files (" + ", ".join(IMAGE_SUFFIXES) + ")")
        add_images.clicked.connect(self._browse_images)
        pickers.addWidget(add_images)
        add_videos = secondary(QPushButton("Add Videos…"))
        add_videos.setToolTip("Pick videos to pull frames from")
        add_videos.clicked.connect(self._browse_videos)
        pickers.addWidget(add_videos)
        pickers.addStretch(1)
        outer.addLayout(pickers)

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
        self._ok = QPushButton("Add")
        self._ok.setDefault(True)
        self._ok.clicked.connect(self.accept)
        buttons.addWidget(self._ok)
        outer.addLayout(buttons)
        self._update_ok()
        self.resize(560, 440)

    def _ticked(self, kind: str) -> list[Path]:
        return [
            Path(item.data(Qt.ItemDataRole.UserRole))
            for item in (self._list.item(i) for i in range(self._list.count()))
            if item.checkState() == Qt.CheckState.Checked and item.data(_KIND) == kind
        ]

    @property
    def videos(self) -> list[Path]:
        return self._ticked("video")

    @property
    def images(self) -> list[Path]:
        return self._ticked("image")

    @property
    def per_video(self) -> int:
        return self._per_video.value()

    @property
    def method(self) -> str:
        return self._method.currentData()

    def add_paths(self, paths: list[str]) -> None:
        """Newly picked files, ticked; ones already listed are left as they are.

        Each file is a video or an image by its extension; anything else is a video,
        as before images could be added.
        """
        known = {
            Path(self._list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._list.count())
        }
        for p in paths:
            path = Path(p).resolve()
            if path not in known and Path(p) not in known:
                known.add(path)
                kind = "image" if path.suffix.lower() in IMAGE_SUFFIXES else "video"
                self._add_item(path, kind, Qt.CheckState.Checked)
        # Adding a ticked item does not emit itemChanged, so refresh by hand.
        self._update_ok()

    def _add_item(self, path: Path, kind: str, state: Qt.CheckState) -> None:
        item = QListWidgetItem(f"{path.name}   ({kind})")
        item.setToolTip(f"{'Image' if kind == 'image' else 'Video'}: {path}")
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setData(_KIND, kind)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(state)
        self._list.addItem(item)

    def _browse_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos", "", _VIDEO_FILTER)
        self.add_paths(paths)

    def _browse_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", "", _IMAGE_FILTER)
        self.add_paths(paths)

    def _update_ok(self, *_) -> None:
        self._ok.setEnabled(bool(self.videos or self.images))
        self._per_video.setEnabled(bool(self.videos))
        self._method.setEnabled(bool(self.videos))
