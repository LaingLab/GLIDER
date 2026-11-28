"""Train Model: export the project and train a YOLO model on it, in a child process.

Training runs in its own Python process (:class:`QProcess`), not a thread: it
holds the GPU and a lot of memory for a long time, a crash in it must not take
the annotator down, and Stop has to be able to end it at once. Its output is
streamed into the dialog, and the epoch counter drives the progress bar.

Runs land in ``<project>/runs/<name>/``; the trained weights are
``weights/best.pt`` there.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from glider.gui.animal_annotator.style import muted, secondary, studio_stylesheet
from glider.vision.training.dataset import DatasetError, export_dataset
from glider.vision.training.project import StudioProject

__all__ = ["TrainDialog", "base_model", "train_script"]

#: Model sizes offered: (suffix, label).
SIZES = (
    ("n", "Nano: fastest, fine for one animal"),
    ("s", "Small"),
    ("m", "Medium"),
    ("l", "Large: most accurate, needs a good GPU"),
)

#: The child process: argv[1] is a JSON object of arguments.
_SCRIPT = """
import json, sys
from ultralytics import YOLO
args = json.loads(sys.argv[1])
model = YOLO(args.pop("model"))
model.train(**args)
"""

_EPOCH = re.compile(r"^\s*(\d+)/(\d+)\s")


def base_model(project: StudioProject, size: str) -> str:
    """The pretrained Ultralytics weights a project of this kind starts from."""
    return f"yolo11{size}-{'seg' if project.is_segment else 'pose'}.pt"


def train_script() -> str:
    return _SCRIPT


class TrainDialog(QDialog):
    """Settings, Start/Stop, a live log, and where the trained model went."""

    def __init__(self, project: StudioProject, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Model")
        self.setStyleSheet(studio_stylesheet())
        self._project = project
        self._process: QProcess | None = None
        self._run_dir: Path | None = None
        self._data_yaml: Path | None = None
        self._partial = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(10)
        title = QLabel("Train a model")
        title.setProperty("studioTitle", True)
        outer.addWidget(title)
        kind = "outlines" if project.is_segment else "keypoints"
        intro = muted(
            QLabel(
                f"Trains a YOLO model on this project's annotated images, so it can find the "
                f"{kind} on new videos. Images are split into train, val and test as shown in "
                "the grid; skipped and unannotated images are left out. The first run "
                "downloads the starting weights."
            ),
            small=True,
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        form = QFormLayout()
        self._size = QComboBox()
        for suffix, label in SIZES:
            self._size.addItem(label, suffix)
        self._size.setToolTip("Bigger models are more accurate but slower to train and run")
        form.addRow("Model size", self._size)
        self._epochs = QSpinBox()
        self._epochs.setRange(1, 2000)
        self._epochs.setValue(100)
        self._epochs.setToolTip(
            "Passes over the training images; training stops early if it stalls"
        )
        form.addRow("Epochs", self._epochs)
        self._imgsz = QComboBox()
        for size in (320, 480, 640, 800, 960, 1280):
            self._imgsz.addItem(f"{size} px", size)
        self._imgsz.setCurrentIndex(2)
        self._imgsz.setToolTip("Images are resized to this; larger sees small animals better")
        form.addRow("Image size", self._imgsz)
        self._device = QComboBox()
        self._device.addItem("Automatic (GPU if there is one)", "")
        self._device.addItem("GPU 0", "0")
        self._device.addItem("CPU (slow)", "cpu")
        self._device.addItem("Apple GPU (MPS)", "mps")
        form.addRow("Run on", self._device)
        outer.addLayout(form)

        self._status = QLabel()
        self._status.setWordWrap(True)
        outer.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)
        font = QFont("Consolas" if sys.platform == "win32" else "Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSizeF(max(8.0, font.pointSizeF() * 0.9))
        self._log.setFont(font)
        self._log.setPlaceholderText("Training output appears here.")
        outer.addWidget(self._log, 1)

        buttons = QHBoxLayout()
        self._open_btn = secondary(QPushButton("Open Run Folder"))
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_folder)
        buttons.addWidget(self._open_btn)
        buttons.addStretch()
        self._close_btn = QPushButton("Close")
        self._close_btn.setFlat(True)
        self._close_btn.clicked.connect(self.reject)
        buttons.addWidget(self._close_btn)
        self._start_btn = QPushButton("Start Training")
        self._start_btn.setDefault(True)
        self._start_btn.clicked.connect(self._start_or_stop)
        buttons.addWidget(self._start_btn)
        outer.addLayout(buttons)
        self.resize(720, 620)
        self._check_ready()

    # -- state -------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._process is not None

    def _check_ready(self) -> None:
        counts = self._project.counts()
        if counts["labelled"] == 0:
            self._status.setText("Annotate some images first: there is nothing to train on yet.")
            self._start_btn.setEnabled(False)
        else:
            self._status.setText(
                f"{counts['labelled']} annotated image(s) will be used. "
                "A few hundred across several videos is a good start."
            )

    def train_args(self, data_yaml: Path, name: str) -> dict:
        """The arguments handed to ``YOLO(model).train``."""
        args = {
            "model": base_model(self._project, self._size.currentData()),
            "data": str(data_yaml),
            "epochs": self._epochs.value(),
            "imgsz": self._imgsz.currentData(),
            "project": str(self._project.folder / "runs"),
            "name": name,
            "exist_ok": False,
            "plots": True,
        }
        if self._device.currentData():
            args["device"] = self._device.currentData()
        return args

    # -- running -----------------------------------------------------------

    def _start_or_stop(self) -> None:
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        try:
            result = export_dataset(self._project)
        except (DatasetError, OSError) as e:
            self._status.setText(f"Can't train yet: {e}")
            return
        self._data_yaml = result.data_yaml
        name = time.strftime("train-%Y%m%d-%H%M%S")
        self._run_dir = self._project.folder / "runs" / name
        args = self.train_args(result.data_yaml, name)

        self._log.clear()
        self._log.appendPlainText(
            f"Dataset: {len(result.train)} train, {len(result.val)} val"
            f"{f', {len(result.test)} test' if result.test else ''} image(s)\n"
            f"Starting from {args['model']}\n"
        )
        process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(self._project.folder))
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        self._partial = ""
        self._progress.setRange(0, args["epochs"])
        self._progress.setValue(0)
        self._progress.setFormat("Starting…")
        self._progress.setVisible(True)
        self._set_running(True)
        self._status.setText("Training. You can keep labelling in the annotator meanwhile.")
        process.start(sys.executable, ["-c", _SCRIPT, json.dumps(args)])

    def _stop(self) -> None:
        if self._process is not None:
            self._status.setText("Stopping…")
            self._process.kill()

    def _set_running(self, running: bool) -> None:
        self._start_btn.setText("Stop" if running else "Start Training")
        self._close_btn.setEnabled(not running)
        for widget in (self._size, self._epochs, self._imgsz, self._device):
            widget.setEnabled(not running)

    def _read_output(self) -> None:
        if self._process is None:
            return
        text = self._partial + bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        # Progress bars rewrite their line with \r: keep only the last state.
        lines = re.split(r"\r?\n", text)
        self._partial = lines.pop()
        for line in lines:
            line = line.split("\r")[-1].rstrip()
            if line:
                self._log.appendPlainText(line)
                self._note_epoch(line)
        if "\r" in self._partial:
            self._note_epoch(self._partial.split("\r")[-1])

    def _note_epoch(self, line: str) -> None:
        match = _EPOCH.match(line)
        if match and int(match.group(2)) == self._progress.maximum():
            epoch = int(match.group(1))
            self._progress.setValue(epoch)
            self._progress.setFormat(f"Epoch {epoch} of {self._progress.maximum()}")

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._log.appendPlainText(f"Could not start Python ({sys.executable}).")
            self._finish(ok=False)

    def _on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._partial.strip():
            self._log.appendPlainText(self._partial.split("\r")[-1])
        self._finish(ok=status == QProcess.ExitStatus.NormalExit and code == 0)

    def _finish(self, ok: bool) -> None:
        if self._process is None:
            return
        self._process = None
        self._set_running(False)
        best = self._run_dir / "weights" / "best.pt" if self._run_dir else None
        self._open_btn.setEnabled(bool(self._run_dir and self._run_dir.exists()))
        if ok and best is not None and best.exists():
            self._progress.setValue(self._progress.maximum())
            self._progress.setFormat("Done")
            self._status.setText(f"Done. The trained model is {best}")
        elif "No module named 'ultralytics'" in self._log.toPlainText():
            self._status.setText(
                "Training needs Ultralytics and PyTorch, which aren't installed in this Python."
            )
        else:
            self._progress.setFormat("Stopped")
            self._status.setText(
                "Training stopped before it finished; the log above says why. "
                "Weights saved so far are in the run folder."
            )

    def _open_folder(self) -> None:
        if self._run_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._run_dir)))

    def reject(self) -> None:
        if self.running:
            return  # Stop first; closing must not orphan a training process
        super().reject()

    def closeEvent(self, event) -> None:
        if self.running:
            event.ignore()
            return
        super().closeEvent(event)
