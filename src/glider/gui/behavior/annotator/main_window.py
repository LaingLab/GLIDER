"""Active-learning clip review UI.

The window opens with a pre-computed batch of :class:`ProposedClip`
objects (from :func:`glider.gui.behavior.annotator.sampler.propose_clips`).
The user labels them one at a time: each clip plays on loop in the
main pane, the user clicks a behavior button or presses a hotkey, the
UI advances to the next clip.

The output is the same ``<stem>_annotations.csv`` the downstream
training pipeline already consumes. Reserved labels
``multi-behavior`` / ``unclear`` are persisted in the CSV but
:func:`glider.analysis.behavior.labels.build_label_series` treats them as
exclusion markers so they never make it to model training.

Focus / hotkey contract
-----------------------

* Single-character behavior hotkeys go through ``QShortcut`` with
  ``WindowShortcut`` context — they fire regardless of which child
  widget has focus, **except** while a ``QLineEdit`` is being typed
  into (Qt's text-input absorbs the key first).
* Pressing ``Escape`` or clicking the clip surface returns focus to
  the clip player so hotkeys come back online.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from glider.analysis.behavior.annotations import (
    AnnotationStore,
    BehaviorZone,
    OverlapError,
    merge_behavior_zones,
)
from glider.analysis.behavior.vocabulary import (
    Behavior,
    Vocabulary,
    VocabularyError,
)
from glider.gui.behavior.annotator.clip_player import ClipPlayer
from glider.gui.behavior.annotator.sampler import ProposedClip
from glider.gui.behavior.annotator.speed_source import SpeedCache, load_session_speed
from glider.gui.behavior.annotator.speed_trace import SpeedTrace
from glider.gui.behavior.annotator.trim_bar import TrimBar, compute_window
from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import apply_tool_theme, readable_text_on

if TYPE_CHECKING:
    from glider.analysis.behavior.cohort_speed import CohortSpeedThresholds
    from glider.gui.behavior.annotator.capture_cache import VideoCaptureCache


# Reserved label sentinels (kept in sync with train/labels.py).
MULTI_BEHAVIOR = "multi-behavior"
UNCLEAR = "unclear"
RESERVED_LABELS: tuple[str, ...] = (MULTI_BEHAVIOR, UNCLEAR)

# Padding (seconds) added on each side of a proposed clip to form the
# trim editor's seekable window. Behaviors often start/end just outside
# the short proposed clip, so the labeler needs reach into the context.
TRIM_PAD_SECONDS = 1.0


def _next_free_hotkey(vocab: Vocabulary) -> str:
    """Suggest the next unused single-character hotkey."""
    taken = set(vocab.hotkeys())
    for ch in "123456789" + "0" + "abcdefghijklmnopqrstuvwxyz":
        if ch not in taken:
            return ch
    return ""


class _SpeedWorker(QObject):
    """Parses one video's pose CSV off the UI thread.

    Parsing a 17 MB CSV takes a second or two. Doing that inline once per
    video turns a labelling session into thirty short freezes, which reads as
    the application hanging rather than as work being done.

    Deliberately self-contained: it is given the paths it needs and reports
    back by signal, rather than holding the window and calling into it. A
    worker that reaches back into a window the user has since closed is
    touching a deleted C++ object from another thread, which does not raise —
    it takes the process down.
    """

    loaded = pyqtSignal(object, object)  # video, SessionSpeed
    failed = pyqtSignal(object, str)  # video, reason
    finished = pyqtSignal()

    def __init__(self, video: Path, pose_csv: Path, px_per_mm: float | None):
        super().__init__()
        self._video = Path(video)
        self._pose_csv = Path(pose_csv)
        self._px_per_mm = px_per_mm

    def run(self) -> None:
        try:
            session = load_session_speed(self._pose_csv, px_per_mm=self._px_per_mm)
        except Exception as e:  # noqa: BLE001 - reported, never fatal
            self.failed.emit(self._video, f"{type(e).__name__}: {e}")
        else:
            self.loaded.emit(self._video, session)
        finally:
            # Always emitted: a worker that dies silently would leave the
            # trace saying "loading…" for the rest of the session.
            self.finished.emit()


class AnnotatorWindow(QMainWindow):
    """Clip-by-clip labelling UI."""

    def __init__(
        self,
        clips: list[ProposedClip],
        videos_meta: dict[Path, Path],  # video_path -> annotations_csv_path
        fps: float = 30.0,
        vocab: Vocabulary | None = None,
        vocab_path: Path | None = None,
        capture_cache: VideoCaptureCache | None = None,
        clip_sampler: Callable[[int], list[ProposedClip]] | None = None,
        pose_csvs: dict[Path, Path] | None = None,
        cohort: CohortSpeedThresholds | None = None,
        px_per_mm: dict[Path, float] | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Behavior Annotator")
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        # ---------- State ----------
        # Optional callable that samples N fresh proposed clips on demand
        # (the sidebar "render more" button). None → review-only, no button.
        self._clip_sampler = clip_sampler
        self.render_button = None
        self.clips: list[ProposedClip] = list(clips)
        self.videos_meta = {Path(v): Path(p) for v, p in videos_meta.items()}
        if not self.videos_meta:
            raise ValueError("AnnotatorWindow requires at least one video")
        self.fps = float(fps)
        self.vocab: Vocabulary = vocab or Vocabulary()
        self.vocab_path: Path | None = vocab_path
        # Eagerly load one AnnotationStore per source video. Lazy loading
        # was considered but the sidebar / wrap-around / first-unlabeled
        # logic asks across-all-clips questions on every refresh, so the
        # cost of eager load (small CSV reads) is amortized.
        self.stores: dict[Path, AnnotationStore] = {}
        # A file that won't parse is not the same as a file that isn't there.
        # load_csv already returns an empty store for a missing path (the
        # normal first-run case, which must stay saveable). A *malformed* file
        # holds labelling work we can't read, and save_csv truncates, so the
        # first keypress would destroy it. Record the failure instead and
        # refuse to write over that path for the rest of the session.
        self.load_errors: dict[Path, str] = {}
        for video_path, ann_path in self.videos_meta.items():
            try:
                self.stores[video_path] = AnnotationStore.load_csv(ann_path)
            except (ValueError, OverlapError, OSError) as e:
                self.stores[video_path] = AnnotationStore()
                self.load_errors[video_path] = str(e)
        # Speed trace. Entirely optional: with no pose CSVs the widget stays
        # hidden and nothing else about the window changes, which is what
        # every caller predating this feature relies on.
        self.pose_csvs: dict[Path, Path] = {Path(v): Path(p) for v, p in (pose_csvs or {}).items()}
        self.cohort = cohort
        self.px_per_mm: dict[Path, float] = {
            Path(v): float(s) for v, s in (px_per_mm or {}).items()
        }
        self.speed_cache = SpeedCache()
        self._speed_threads: list[QThread] = []
        self._speed_workers: list[_SpeedWorker] = []

        # Map proposed-clip index → the BehaviorZone it produced. This is
        # how a clip is recognised as "labeled" now that a saved zone's
        # trimmed bounds no longer equal the proposed clip's bounds. Seeded
        # from existing annotations by overlap so resuming a session shows
        # already-labeled clips as done.
        self._clip_zone: dict[int, BehaviorZone] = {}
        self._seed_clip_zones()
        self.current: int = self._first_unlabeled_index()
        self._behavior_shortcuts: list[QShortcut] = []
        # Pad (frames) for the trim window, derived from the fps.
        self._trim_pad_frames = int(round(TRIM_PAD_SECONDS * float(fps)))
        # Pass the capture cache down to the clip player.
        self._capture_cache = capture_cache
        # First video is the deterministic default for vocab fallback and
        # the initial title (refreshed per clip in _refresh_clip).
        self._primary_video: Path = next(iter(self.videos_meta))

        # ---------- Layout ----------
        self._build_menubar()
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)
        self.setCentralWidget(root)

        # Sidebar.
        self.sidebar = self._build_sidebar()
        root_layout.addWidget(self.sidebar)

        # Main pane.
        main = QVBoxLayout()
        main.setSpacing(8)
        root_layout.addLayout(main, 1)

        # Title row at the top of the main pane.
        title_row = QHBoxLayout()
        self.title_label = QLabel(self._current_video_path().name)
        self.title_label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {colors.TEXT_PRIMARY};"
        )
        self.save_indicator = QLabel("")
        self.save_indicator.setStyleSheet(f"color: {colors.TEXT_TERTIARY}; font-size: 12px;")
        # Big, color-coded readout of the current clip's label (top-right).
        self.big_label = QLabel("—")
        self.big_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        title_row.addWidget(self.save_indicator)
        title_row.addSpacing(12)
        title_row.addWidget(self.big_label)
        main.addLayout(title_row)

        self.clip = ClipPlayer(capture_cache=self._capture_cache)
        main.addWidget(self.clip, 1)

        # Trim editor: drag IN/OUT handles to tighten the saved zone to the
        # exact behavior. Defaults to the whole proposed clip, so doing
        # nothing reproduces the pre-trim behavior.
        self.trim_bar = TrimBar()
        self.trim_bar.bounds_changed.connect(self._on_trim_changed)
        main.addWidget(self.trim_bar)
        # Speed trace, directly under the trim bar and sharing its frame
        # window, so a peak in the trace sits above the frames that produced
        # it. Hidden unless this session was given pose data.
        self.speed_trace = SpeedTrace()
        # Whether the feature is on for this session -- NOT whether the widget
        # is currently on screen. isVisible() is false until the window is
        # shown, so gating the logic on it would leave the trace unconfigured
        # for the first clip and blank behind a hidden parent.
        self._speed_enabled = bool(self.pose_csvs)
        self.speed_trace.setVisible(self._speed_enabled)
        main.addWidget(self.speed_trace)
        self.clip.frame_changed.connect(self._on_player_frame)

        self.trim_hint = QLabel("trim: drag handles · [ ] in · { } out")
        self.trim_hint.setStyleSheet(f"color: {colors.TEXT_MUTED}; font-size: 11px;")
        self.trim_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.trim_hint)

        self.caption_label = QLabel("(no clip)")
        self.caption_label.setStyleSheet(f"color: {colors.TEXT_TERTIARY}; font-size: 12px;")
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.caption_label)

        # Label buttons card.
        label_card = QFrame()
        # Qualified by object name. Unqualified, a stylesheet set on a
        # container also matches every descendant, so the card's border was
        # drawn around each of its child labels too -- which is why every
        # heading in this window used to sit in its own little box.
        label_card.setObjectName("LabelCard")
        label_card.setStyleSheet(
            f"QFrame#LabelCard {{ background: {colors.SURFACE_1}; "
            f"border: 1px solid {colors.BORDER}; border-radius: 10px; }}"
        )
        lcv = QVBoxLayout(label_card)
        lcv.setContentsMargins(16, 14, 16, 14)
        lcv.setSpacing(8)
        h3 = _section_header("Label this clip")
        lcv.addWidget(h3)
        # Behavior chips row (rebuilt on vocab change).
        self.chips_row = QHBoxLayout()
        self.chips_row.setSpacing(8)
        chips_holder = QFrame()
        chips_holder.setLayout(self.chips_row)
        lcv.addWidget(chips_holder)
        # Markers row.
        markers = QHBoxLayout()
        self.btn_multi = QPushButton("⚠ multi-behavior   M")
        self.btn_multi.setObjectName(CHIP_ID)
        self.btn_multi.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_multi.setStyleSheet(_chip_warn_css())
        self.btn_multi.clicked.connect(lambda: self._apply_label(MULTI_BEHAVIOR))
        self.btn_unclear = QPushButton("? unclear   U")
        self.btn_unclear.setObjectName(CHIP_ID)
        self.btn_unclear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_unclear.setStyleSheet(_chip_unknown_css())
        self.btn_unclear.clicked.connect(lambda: self._apply_label(UNCLEAR))
        self.btn_skip = QPushButton("skip   Space")
        self.btn_skip.setObjectName(CHIP_ID)
        self.btn_skip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_skip.setStyleSheet(_chip_subtle_css())
        self.btn_skip.clicked.connect(self._skip)
        markers.addWidget(self.btn_multi)
        markers.addWidget(self.btn_unclear)
        markers.addWidget(self.btn_skip)
        markers.addStretch(1)
        lcv.addLayout(markers)

        main.addWidget(label_card)

        # Nav row.
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("← prev")
        self.btn_prev.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_prev.clicked.connect(lambda: self._go(-1))
        self.btn_next = QPushButton("next →")
        self.btn_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_next.clicked.connect(lambda: self._go(+1))
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        main.addLayout(nav)

        self.setStatusBar(QStatusBar())
        # Opened from the Annotate tab with no parent, so Qt hands it no
        # stylesheet -- the same gap the three tool windows had. This one was
        # additionally built light throughout, so the hardcoded surfaces below
        # move to the shared tokens rather than just inheriting the theme.
        apply_tool_theme(self)
        # ---------- Wiring ----------
        self._build_transport_shortcuts()
        self._rebuild_behavior_shortcuts()
        self._refresh_all()
        # Initial focus on the clip player so hotkeys are live.
        self.clip.setFocus()
        self.clip.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Per-video lookups
    # ------------------------------------------------------------------
    def _store_for(self, clip: ProposedClip) -> AnnotationStore:
        return self.stores[Path(clip.video_path)]

    def _annotations_path_for(self, clip: ProposedClip) -> Path:
        return self.videos_meta[Path(clip.video_path)]

    def _current_video_path(self) -> Path:
        """The video that the currently-displayed clip belongs to (or the
        primary video if there are no clips)."""
        if self.clips and 0 <= self.current < len(self.clips):
            return Path(self.clips[self.current].video_path)
        return self._primary_video

    # ------------------------------------------------------------------
    # Menubar
    # ------------------------------------------------------------------
    def _build_menubar(self) -> None:
        bar = self.menuBar()
        m = bar.addMenu("&File")
        open_vocab = QAction("Open vocabulary…", self)
        open_vocab.triggered.connect(self._open_vocab_dialog)
        m.addAction(open_vocab)
        save_vocab = QAction("Save vocabulary as…", self)
        save_vocab.triggered.connect(self._save_vocab_as)
        m.addAction(save_vocab)
        m.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        m.addAction(quit_act)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setFixedWidth(300)
        sidebar.setObjectName("Sidebar")  # qualified -- see LabelCard above
        sidebar.setStyleSheet(
            f"QFrame#Sidebar {{ background: {colors.SURFACE_1}; "
            f"border: 1px solid {colors.BORDER}; border-radius: 10px; }}"
        )
        v = QVBoxLayout(sidebar)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)

        # Progress card.
        v.addWidget(_section_header("Progress"))
        self.progress_label = QLabel("clip 0 / 0")
        self.progress_label.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {colors.TEXT_PRIMARY};"
        )
        v.addWidget(self.progress_label)
        self.labeled_label = QLabel("0 labeled · 0 marked")
        self.labeled_label.setStyleSheet(f"color: {colors.TEXT_TERTIARY}; font-size: 12px;")
        v.addWidget(self.labeled_label)

        # Vocabulary.
        v.addSpacing(4)
        v.addWidget(_section_header("Vocabulary"))
        self.vocab_list_box = QVBoxLayout()
        self.vocab_list_box.setSpacing(2)
        v.addLayout(self.vocab_list_box)
        self.vocab_empty_hint = QLabel(
            "No behaviors yet. Add some below — they save next to the video."
        )
        self.vocab_empty_hint.setWordWrap(True)
        self.vocab_empty_hint.setStyleSheet(f"color: {colors.TEXT_MUTED}; font-size: 12px;")
        v.addWidget(self.vocab_empty_hint)

        # Add-behavior form.
        v.addSpacing(4)
        v.addWidget(_section_header("Add behavior"))
        add_row = QHBoxLayout()
        self.add_name = QLineEdit()
        self.add_name.setPlaceholderText("name (e.g. rearing)")
        self.add_name.returnPressed.connect(self._on_add_behavior)
        self.add_hotkey = QLineEdit()
        self.add_hotkey.setPlaceholderText("key")
        self.add_hotkey.setMaxLength(1)
        self.add_hotkey.setFixedWidth(36)
        self.add_hotkey.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_hotkey.returnPressed.connect(self._on_add_behavior)
        self.add_button = QPushButton("Add")
        self.add_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_button.clicked.connect(self._on_add_behavior)
        add_row.addWidget(self.add_name, 1)
        add_row.addWidget(self.add_hotkey)
        add_row.addWidget(self.add_button)
        v.addLayout(add_row)

        # Merge behaviors: fold one or more source behaviors into a target.
        v.addSpacing(4)
        v.addWidget(_section_header("Merge behaviors"))
        self.merge_sources = QListWidget()
        self.merge_sources.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.merge_sources.setFixedHeight(72)
        v.addWidget(self.merge_sources)
        merge_row = QHBoxLayout()
        merge_row.addWidget(QLabel("into"))
        self.merge_target = QComboBox()
        merge_row.addWidget(self.merge_target, 1)
        self.merge_button = QPushButton("Merge")
        self.merge_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.merge_button.clicked.connect(self._on_merge_clicked)
        merge_row.addWidget(self.merge_button)
        v.addLayout(merge_row)

        # Render-more (review mode with a sampler available).
        if self._clip_sampler is not None:
            v.addSpacing(4)
            v.addWidget(_section_header("Render more clips"))
            more_row = QHBoxLayout()
            self.render_count = QLineEdit("30")
            self.render_count.setFixedWidth(56)
            self.render_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.render_button = QPushButton("Render more")
            self.render_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.render_button.clicked.connect(self._on_render_more_clicked)
            more_row.addWidget(self.render_count)
            more_row.addWidget(self.render_button, 1)
            v.addLayout(more_row)

        v.addStretch(1)
        return sidebar

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------
    def _build_transport_shortcuts(self) -> None:
        for key, slot in (
            ("M", lambda: self._apply_label(MULTI_BEHAVIOR)),
            ("U", lambda: self._apply_label(UNCLEAR)),
            ("Space", self._skip),
            ("Left", lambda: self._go(-1)),
            ("Right", lambda: self._go(+1)),
            ("N", self._focus_add_name),
            ("[", lambda: self.trim_bar.nudge_in(-1)),
            ("]", lambda: self.trim_bar.nudge_in(+1)),
            ("{", lambda: self.trim_bar.nudge_out(-1)),
            ("}", lambda: self.trim_bar.nudge_out(+1)),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(slot)
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        esc.activated.connect(self._focus_clip)

    def _rebuild_behavior_shortcuts(self) -> None:
        for sc in self._behavior_shortcuts:
            sc.setEnabled(False)
            sc.setParent(None)
        self._behavior_shortcuts.clear()
        for b in self.vocab:
            seq = QKeySequence(b.hotkey)
            if seq.isEmpty():
                continue
            sc = QShortcut(seq, self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda name=b.name: self._apply_label(name))
            self._behavior_shortcuts.append(sc)

    def _focus_add_name(self) -> None:
        self.add_name.setFocus()

    def _focus_clip(self) -> None:
        self.add_name.clear()
        self.add_hotkey.clear()
        self.clip.setFocus()

    # ------------------------------------------------------------------
    # Vocab management
    # ------------------------------------------------------------------
    def _on_add_behavior(self) -> None:
        name = self.add_name.text().strip()
        hotkey = self.add_hotkey.text().strip()
        if not name:
            return
        if not hotkey:
            hotkey = _next_free_hotkey(self.vocab)
            if not hotkey:
                QMessageBox.warning(
                    self,
                    "No hotkeys left",
                    "Every single-character hotkey is taken.",
                )
                return
        try:
            self.vocab.add(Behavior(name=name, hotkey=hotkey))
        except VocabularyError as e:
            QMessageBox.warning(self, "Couldn't add behavior", str(e))
            return
        self.add_name.setText("")
        self.add_hotkey.setText("")
        self.clip.setFocus()
        self._save_vocab()
        self._rebuild_behavior_shortcuts()
        self._refresh_sidebar()
        self._refresh_chips()
        # Apply the freshly-added behavior to the current clip too —
        # that's almost always what the user wants when they type a new
        # name during labeling.
        self._apply_label(name)

    def _remove_behavior(self, name: str) -> None:
        # Collect zones using this behavior across every per-video store.
        zones_using_by_video: dict[Path, list] = {
            video: store.zones_for_behavior(name) for video, store in self.stores.items()
        }
        total_zones = sum(len(zs) for zs in zones_using_by_video.values())
        msg = f"Remove behavior {name!r} from the vocabulary?"
        if total_zones:
            msg += (
                f"\n\n{total_zones} clip(s) are labeled with this "
                f"behavior. They will be removed from the annotations."
            )
        ok = QMessageBox.question(
            self,
            "Remove behavior?",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        for video, zones in zones_using_by_video.items():
            store = self.stores[video]
            for z in zones:
                store.remove(z)
        self.vocab.remove(name)
        self._save_vocab()
        # Persist every store whose contents changed.
        for video, zones in zones_using_by_video.items():
            if not zones:
                continue
            self._save_annotations_for_video(video)
        self._rebuild_behavior_shortcuts()
        self._refresh_all()
        self.clip.setFocus()

    def _on_merge_clicked(self) -> None:
        target = self.merge_target.currentText().strip()
        sources = [
            item.text() for item in self.merge_sources.selectedItems() if item.text() != target
        ]
        if not target or not sources:
            return
        n_zones = sum(
            len(store.zones_for_behavior(s)) for store in self.stores.values() for s in sources
        )
        ok = QMessageBox.question(
            self,
            "Merge behaviors?",
            f"Merge {', '.join(sources)} into {target!r}?\n\n"
            f"{n_zones} zone(s) across {len(self.stores)} video(s) will be "
            f"relabelled and the source behavior(s) removed from the "
            f"vocabulary. This can't be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        self._merge_behavior(sources, target)

    def _merge_behavior(self, sources: list[str], target: str) -> None:
        """Fold ``sources`` into ``target`` across every video's store
        (renaming zones, unioning overlaps), drop the sources from the
        vocabulary, and persist. Destructive — annotations are rewritten."""
        sources = [s for s in sources if s != target and s in self.vocab]
        if not sources or target not in self.vocab:
            return
        for video, store in self.stores.items():
            store.replace(merge_behavior_zones(list(store), sources, target))
            self._save_annotations_for_video(video)
        for s in sources:
            self.vocab.remove(s)
        self._save_vocab()
        self._rebuild_behavior_shortcuts()
        # Zone objects were replaced; rebuild the clip→zone associations.
        self._clip_zone.clear()
        self._seed_clip_zones()
        self._refresh_all()
        self.clip.setFocus()

    # ------------------------------------------------------------------
    # Clip labelling
    # ------------------------------------------------------------------
    def _apply_label(self, label: str) -> None:
        if not self.clips or self.current >= len(self.clips):
            return
        clip = self.clips[self.current]
        video = Path(clip.video_path)
        if video in self.load_errors:
            # Refusing at the source rather than at save time: silently
            # accepting labels that can never be written would cost the
            # operator a whole session's work before anything said so.
            QMessageBox.warning(
                self,
                "Couldn't label",
                f"{self.videos_meta[video].name} could not be read, so labels "
                f"for {video.name} can't be saved without overwriting it:\n\n"
                f"{self.load_errors[video]}\n\n"
                "Move or repair that file, then reopen the annotator.",
            )
            return
        store = self._store_for(clip)
        in_frame, out_frame = self.trim_bar.bounds()
        # Remove the zone previously created for THIS clip (so the user can
        # change their mind) before adding the trimmed one. Tracked via
        # _clip_zone because trimmed bounds no longer equal the proposed
        # clip's bounds.
        prior = self._clip_zone.get(self.current)
        if prior is not None:
            store.remove(prior)
        try:
            zone = BehaviorZone(
                behavior=label,
                start_frame=int(in_frame),
                end_frame=int(out_frame),
            )
            store.add(zone)
        except (OverlapError, ValueError) as e:
            # Restore the prior label so a failed re-label doesn't lose it.
            if prior is not None:
                try:
                    store.add(prior)
                except OverlapError:
                    pass
            QMessageBox.warning(self, "Couldn't label", str(e))
            return
        self._clip_zone[self.current] = zone
        self._save_annotations_for(clip)
        # Advance to next unlabeled.
        self.current = self._next_unlabeled_index(after=self.current)
        self._refresh_all()
        self.clip.setFocus()

    def _persist_current_trim(self) -> None:
        """Save an in-place trim of the current clip's zone — same behavior,
        new bounds — so boundary edits stick when you move to another clip
        without having to re-press the label.

        Only applies to clips that already have a saved zone; an unlabeled
        clip has nothing to update until a label is applied. A no-op when
        the trim hasn't moved, or restores the original if the new span
        would collide with another zone."""
        if not self.clips or self.current >= len(self.clips):
            return
        prior = self._clip_zone.get(self.current)
        if prior is None:
            return
        in_frame, out_frame = self.trim_bar.bounds()
        if (int(in_frame), int(out_frame)) == (prior.start_frame, prior.end_frame):
            return
        clip = self.clips[self.current]
        store = self._store_for(clip)
        store.remove(prior)
        try:
            zone = BehaviorZone(
                behavior=prior.behavior,
                start_frame=int(in_frame),
                end_frame=int(out_frame),
            )
            store.add(zone)
        except (OverlapError, ValueError):
            store.add(prior)  # collision/invalid → keep the original untouched
            return
        self._clip_zone[self.current] = zone
        self._save_annotations_for(clip)

    def _skip(self) -> None:
        self._persist_current_trim()
        self.current = self._next_unlabeled_index(after=self.current)
        self._refresh_all()
        self.clip.setFocus()

    def _go(self, delta: int) -> None:
        if not self.clips:
            return
        self._persist_current_trim()
        self.current = max(0, min(self.current + delta, len(self.clips) - 1))
        self._refresh_all()
        self.clip.setFocus()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def _first_unlabeled_index(self) -> int:
        for i in range(len(self.clips)):
            if i not in self._clip_zone:
                return i
        return 0  # all done — fall back to the first clip

    def _next_unlabeled_index(self, after: int) -> int:
        n = len(self.clips)
        if n == 0:
            return 0
        for i in range(after + 1, n):
            if i not in self._clip_zone:
                return i
        # Wrap once back to start; if nothing's left, stay on current.
        for i in range(0, after + 1):
            if i not in self._clip_zone:
                return i
        return min(after, n - 1)

    def _label_for_index(self, idx: int) -> str:
        zone = self._clip_zone.get(idx)
        return zone.behavior if zone is not None else ""

    def _seed_clip_zones(self) -> None:
        """Associate each proposed clip with the existing zone (if any) that
        overlaps its frame range the most, so resuming a session shows
        already-labeled clips as done. Reserved markers count too — a clip
        previously flagged multi-behavior/unclear shouldn't be re-proposed."""
        for i, clip in enumerate(self.clips):
            store = self._store_for(clip)
            best: BehaviorZone | None = None
            best_overlap = 0
            for z in store:
                overlap = min(z.end_frame, clip.end_frame) - max(z.start_frame, clip.start_frame)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = z
            if best is not None:
                self._clip_zone[i] = best

    # ------------------------------------------------------------------
    # Speed trace
    # ------------------------------------------------------------------
    def _on_player_frame(self, frame: int) -> None:
        """Move the playhead to the frame the viewer is actually looking at."""
        if self._speed_enabled:
            self.speed_trace.set_playhead(int(frame))

    def _load_speed_now(self, video: Path) -> None:
        """Compute one video's speed trace and file the result. Blocking.

        Separated from the threading so the expensive work is testable
        synchronously and the worker below is a thin shell around it. Failure
        is recorded rather than raised: a session with one unreadable pose CSV
        must keep labelling, just without a trace for that video.
        """
        video = Path(video)
        pose_csv = self.pose_csvs.get(video)
        if pose_csv is None:
            self.speed_cache.fail(video, "no pose CSV for this video")
            return
        try:
            session = load_session_speed(pose_csv, px_per_mm=self.px_per_mm.get(video))
        except Exception as e:  # noqa: BLE001 - shown in the trace, never fatal
            self.speed_cache.fail(video, f"{type(e).__name__}: {e}")
            return
        self.speed_cache.store(video, session)

    def _ensure_speed_loaded(self, video: Path) -> None:
        """Start a background load for ``video`` if nothing has yet.

        :meth:`SpeedCache.begin` is the gate, so landing on a second clip from
        the same video does not start a second parse of a 17 MB CSV, and a
        video already known bad is not retried.
        """
        video = Path(video)
        if not self.pose_csvs:
            return
        pose_csv = self.pose_csvs.get(video)
        if pose_csv is None:
            return
        if not self.speed_cache.begin(video):
            return

        thread = QThread(self)
        worker = _SpeedWorker(video, pose_csv, self.px_per_mm.get(video))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Bound methods, so Qt drops the connections when this window is
        # destroyed rather than delivering into a dead object.
        worker.loaded.connect(self._on_speed_loaded)
        worker.failed.connect(self._on_speed_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # Keep the worker alive. It has no parent (moveToThread does not set
        # one), so once this method returns the only Python reference would be
        # gone, PyQt would drop the underlying C++ object, and
        # thread.started would fire into nothing -- leaving the trace on
        # "loading pose data…" for the rest of the session.
        self._speed_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget_speed_worker(w))
        self._speed_threads.append(thread)
        thread.start()

    def _forget_speed_worker(self, worker: _SpeedWorker) -> None:
        if worker in self._speed_workers:
            self._speed_workers.remove(worker)

    def _on_speed_loaded(self, video, session) -> None:
        self.speed_cache.store(video, session)
        self._refresh_speed_trace()

    def _on_speed_failed(self, video, reason: str) -> None:
        self.speed_cache.fail(video, reason)
        self._refresh_speed_trace()

    def _stop_speed_threads(self) -> None:
        """Let every in-flight parse finish before this window goes away.

        Qt aborts the process when a running QThread is destroyed, so closing
        the annotator while a pose CSV is still being read must not simply
        drop the thread on the floor.
        """
        for thread in list(self._speed_threads):
            if thread is None:
                continue
            try:
                thread.quit()
                thread.wait(5000)
            except RuntimeError:
                # Already deleted by deleteLater; nothing left to wait for.
                pass
        self._speed_threads.clear()
        # Safe only after every thread above has stopped: these are the
        # objects those threads were running.
        self._speed_workers.clear()

    def _refresh_speed_trace(self) -> None:
        """Point the trace at the current clip's video, window and thresholds."""
        if not self._speed_enabled or not self.clips:
            return
        if self.current >= len(self.clips):
            return
        video = self._current_video_path()
        state = self.speed_cache.state(video)
        if state == "ready":
            self.speed_trace.set_session(self.speed_cache.get(video))
        elif state == "failed":
            self.speed_trace.set_failed(self.speed_cache.error(video) or "unavailable")
        elif state == "loading":
            self.speed_trace.set_loading()
        else:
            self.speed_trace.set_session(None)

        self.speed_trace.set_window(self.trim_bar._win_start, self.trim_bar._win_end)
        if self.cohort is not None:
            self.speed_trace.set_thresholds(self.cohort.freeze, self.cohort.dart, self.cohort.unit)

    # ------------------------------------------------------------------
    # Render-more (review mode)
    # ------------------------------------------------------------------
    def _on_render_more_clicked(self) -> None:
        try:
            n = int(self.render_count.text())
        except (ValueError, AttributeError):
            n = 30
        self._render_more_clips(max(1, n))

    def _render_more_clips(self, n: int) -> None:
        """Sample up to ``n`` fresh clips that avoid already-labelled spans
        and aren't already in the list, append them, and jump to the first.

        Re-samples with a fresh seed (the injected sampler advances its own
        seed per call) up to a few times to reach ``n`` survivors before
        giving up — heavily-labelled videos may yield fewer."""
        if self._clip_sampler is None:
            return
        existing_keys = {(c.video_path, c.start_frame, c.end_frame) for c in self.clips}
        collected: list[ProposedClip] = []
        for _attempt in range(5):
            if len(collected) >= n:
                break
            for c in self._clip_sampler(n):
                key = (c.video_path, c.start_frame, c.end_frame)
                if key in existing_keys:
                    continue
                if self._center_in_labelled_zone(c):
                    continue
                existing_keys.add(key)
                collected.append(c)
                if len(collected) >= n:
                    break
        if not collected:
            self.statusBar().showMessage(
                "no new clips found (already-labelled regions filtered)", 4000
            )
            return
        first_new = len(self.clips)
        self.clips.extend(collected[:n])
        self.current = first_new
        self._refresh_all()
        self.clip.setFocus()

    def _center_in_labelled_zone(self, clip: ProposedClip) -> bool:
        store = self.stores.get(Path(clip.video_path))
        return bool(store and store.zones_at_frame(clip.center_frame))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load_error_message(self) -> str:
        """Operator-facing description of the unreadable annotation files.

        Empty when everything loaded. Separate from the dialog that shows it
        so the wording can be asserted without a running event loop.
        """
        if not self.load_errors:
            return ""
        lines = [
            f"{self.videos_meta[video].name}\n    {reason}"
            for video, reason in sorted(self.load_errors.items())
        ]
        return (
            "These annotation files could not be read, so their existing "
            "labels are not shown:\n\n"
            + "\n".join(lines)
            + "\n\nThey have been left untouched — labelling these videos is "
            "disabled so the files aren't overwritten. Move or repair them, "
            "then reopen the annotator."
        )

    def warn_about_load_errors(self) -> bool:
        """Show the unreadable-file warning. Returns True when one was shown."""
        message = self.load_error_message()
        if not message:
            return False
        QMessageBox.warning(self, "Annotations not loaded", message)
        return True

    def _save_annotations_for_video(self, video: Path) -> None:
        """Persist the store for `video` to its annotations CSV."""
        ann_path = self.videos_meta[video]
        if video in self.load_errors:
            # Writing here would truncate a file we could not read, throwing
            # away whatever labelling it holds.
            self.save_indicator.setText(f"not saved · {ann_path.name} is unreadable")
            return
        store = self.stores[video]
        try:
            ann_path.parent.mkdir(parents=True, exist_ok=True)
            store.save_csv(ann_path)
            self.save_indicator.setText(f"saved · {ann_path.name}")
        except OSError as e:
            self.save_indicator.setText(f"save failed: {e}")

    def _save_annotations_for(self, clip: ProposedClip) -> None:
        self._save_annotations_for_video(Path(clip.video_path))

    def _save_vocab(self) -> None:
        if self.vocab_path is None and self._primary_video is not None:
            self.vocab_path = (
                self._primary_video.parent / f"{self._primary_video.stem}_behaviors.yaml"
            )
        if self.vocab_path is None:
            return
        try:
            self.vocab.save(self.vocab_path)
        except OSError as e:
            self.statusBar().showMessage(f"vocab save failed: {e}", 4000)

    def _open_vocab_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open vocabulary", "", "Vocabulary (*.yaml *.yml *.json)"
        )
        if not path:
            return
        try:
            self.vocab = Vocabulary.load(path)
            self.vocab_path = Path(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Couldn't load vocabulary", str(e))
            return
        self._rebuild_behavior_shortcuts()
        self._refresh_all()

    def _save_vocab_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save vocabulary",
            str(self.vocab_path or "behaviors.yaml"),
            "YAML (*.yaml *.yml);;JSON (*.json)",
        )
        if not path:
            return
        self.vocab_path = Path(path)
        self._save_vocab()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def _refresh_all(self) -> None:
        self._refresh_sidebar()
        self._refresh_chips()
        self._refresh_clip()

    def _refresh_sidebar(self) -> None:
        n = len(self.clips)
        idx = self.current + 1 if self.clips else 0
        self.progress_label.setText(f"clip {idx} / {n}")
        labeled = len(self._clip_zone)
        # Aggregate reserved-label counts across every per-video store so
        # the sidebar shows a session-wide tally, not just the active video.
        marked = sum(
            1 for store in self.stores.values() for z in store if z.behavior in RESERVED_LABELS
        )
        self.labeled_label.setText(f"{labeled} labeled · {marked} marked unclear/multi")

        # Vocabulary list.
        while self.vocab_list_box.count():
            item = self.vocab_list_box.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if len(self.vocab) == 0:
            self.vocab_empty_hint.show()
        else:
            self.vocab_empty_hint.hide()
        # Aggregate per-behavior counts across every per-video store.
        counts: dict[str, int] = {}
        for store in self.stores.values():
            for name, cnt in store.counts_by_behavior().items():
                counts[name] = counts.get(name, 0) + cnt
        for b in self.vocab:
            row = _VocabRow(
                name=b.name,
                hotkey=b.hotkey,
                color=b.color,
                count=counts.get(b.name, 0),
                on_remove=self._remove_behavior,
            )
            self.vocab_list_box.addWidget(row)
        # Refresh the next-free hotkey hint.
        nf = _next_free_hotkey(self.vocab)
        self.add_hotkey.setPlaceholderText(nf if nf else "—")

        # Keep the merge controls in sync with the vocabulary.
        names = self.vocab.names()
        self.merge_sources.clear()
        self.merge_sources.addItems(names)
        cur = self.merge_target.currentText()
        self.merge_target.blockSignals(True)
        self.merge_target.clear()
        self.merge_target.addItems(names)
        if cur in names:
            self.merge_target.setCurrentText(cur)
        self.merge_target.blockSignals(False)

    def _refresh_chips(self) -> None:
        # Clear current chips.
        while self.chips_row.count():
            item = self.chips_row.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for b in self.vocab:
            btn = QPushButton(f"●  {b.name}    {b.hotkey}")
            btn.setObjectName(CHIP_ID)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(_chip_css(b.color))
            btn.clicked.connect(lambda _ch=False, name=b.name: self._apply_label(name))
            self.chips_row.addWidget(btn)
        self.chips_row.addStretch(1)

    def _update_caption(self) -> None:
        """Set the caption from the CURRENT trim bounds so the frame readout
        and duration track live as the user drags/nudges the trim bar."""
        if not self.clips or self.current >= len(self.clips):
            return
        in_f, out_f = self.trim_bar.bounds()
        span_ms = int((out_f - in_f) / self.fps * 1000)
        current_label = self._label_for_index(self.current) or "(unlabeled)"
        self.caption_label.setText(
            f"clip {self.current + 1} of {len(self.clips)}  ·  "
            f"frames {in_f:,}–{out_f:,}  ·  "
            f"{span_ms} ms  ·  current: {current_label}"
        )

    def _refresh_clip(self) -> None:
        # Keep the window-title chrome (filename in the header) in sync
        # with whichever video the active clip belongs to.
        self.title_label.setText(self._current_video_path().name)
        if not self.clips:
            self.caption_label.setText("(no clips proposed — nothing to label)")
            self._set_big_label(None)
            return
        clip = self.clips[self.current]
        label_now = self._label_for_index(self.current)
        self._set_big_label(label_now)
        # Configure the trim editor for this clip: a padded seekable window,
        # the proposed region shaded, and IN/OUT defaulting to the saved
        # zone (if already labeled) or the whole proposed clip.
        win_start, win_end = compute_window(clip.start_frame, clip.end_frame, self._trim_pad_frames)
        self.trim_bar.set_window(win_start, win_end)
        self.trim_bar.set_clip_region(clip.start_frame, clip.end_frame)
        prior = self._clip_zone.get(self.current)
        if prior is not None:
            self.trim_bar.set_bounds(prior.start_frame, prior.end_frame)
        else:
            self.trim_bar.set_bounds(clip.start_frame, clip.end_frame)
        in_f, out_f = self.trim_bar.bounds()
        self._update_caption()
        # Speed for this clip's video: start the load if this is the first
        # clip from it, then point the trace at the window either way.
        self._ensure_speed_loaded(self._current_video_path())
        self._refresh_speed_trace()
        # Switch the player to the (trimmed) clip span.
        ok = self.clip.set_clip(
            video_path=clip.video_path,
            start_frame=in_f,
            end_frame=out_f,
            fps=self.fps,
        )
        if not ok:
            self.statusBar().showMessage(f"clip player failed to open {clip.video_path}", 3000)

    def _on_trim_changed(self, in_frame: int, out_frame: int) -> None:
        """Handle bar drags/nudges: loop the new span so the preview tracks
        the trim live, and update the on-screen frame readout."""
        self.clip.set_loop_bounds(int(in_frame), int(out_frame))
        self._update_caption()
        self._refresh_speed_trace()

    def _set_big_label(self, label: str | None) -> None:
        """Show the current clip's label big in the top-right, tinted by its
        vocabulary color. ``None``/empty shows a muted placeholder."""
        if label:
            color = self.vocab.color_for(label)
            self.big_label.setText(label)
            self.big_label.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: 700;")
        else:
            self.big_label.setText("—")
            self.big_label.setStyleSheet(
                f"color: {colors.TEXT_DISABLED}; font-size: 26px; font-weight: 700;"
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        # Wait for any in-flight pose parse FIRST. Qt aborts the process when
        # a running QThread is destroyed, and this window owns them.
        self._stop_speed_threads()
        # Release any privately-owned capture in the player (no-op for
        # cache-owned captures — see ClipPlayer.release()).
        try:
            self.clip.release()
        except Exception:
            pass
        # And close every cache-owned capture.
        if self._capture_cache is not None:
            try:
                self._capture_cache.close_all()
            except Exception:
                pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Sidebar sub-widgets + chip styling helpers
# ---------------------------------------------------------------------------


def _section_header(text: str) -> QLabel:
    # Upper-case in Python — Qt Style Sheets have no text-transform, and no
    # letter-spacing either (both log "Unknown property"), so the tracking
    # that makes a run of capitals read as a label is set on the QFont.
    h = QLabel(text.upper())
    h.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {colors.TEXT_TERTIARY};")
    font = h.font()
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.9)
    h.setFont(font)
    return h


class _VocabRow(QFrame):
    """One row in the vocabulary list — color dot + name + hotkey + count + ×."""

    def __init__(
        self,
        name: str,
        hotkey: str,
        color: str,
        count: int,
        on_remove,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("VocabRow")  # qualified -- see LabelCard
        self.setStyleSheet(
            f"QFrame#VocabRow {{ background: {colors.BASE}; "
            f"border: 1px solid {colors.BORDER}; border-radius: 6px; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background: {color}; border-radius: 5px;")
        layout.addWidget(dot)
        name_label = QLabel(name)
        name_label.setStyleSheet(f"font-weight: 500; color: {colors.TEXT_SECONDARY};")
        layout.addWidget(name_label)
        layout.addStretch(1)
        kbd = QLabel(hotkey)
        kbd.setStyleSheet(
            f"color: {colors.TEXT_TERTIARY}; font-family: SF Mono, monospace; "
            f"font-size: 11px; background: {colors.CHROME}; "
            f"border: 1px solid {colors.BORDER}; border-radius: 4px; padding: 0 5px;"
        )
        layout.addWidget(kbd)
        count_label = QLabel(str(count))
        count_label.setMinimumWidth(24)
        count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # NOTE: Qt Style Sheets don't support `font-variant-numeric`
        # (it spams "Unknown property" warnings); the right-alignment +
        # min-width already keep the counts tidy.
        count_label.setStyleSheet(f"color: {colors.TEXT_TERTIARY};")
        layout.addWidget(count_label)
        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove_btn.setToolTip(f"Remove {name!r} from vocabulary")
        remove_btn.setObjectName("VocabRemove")
        # padding: 0 is load-bearing. The theme's QPushButton carries
        # "padding: 6px 12px", which a widget sheet does not reset -- 24px of
        # horizontal padding inside a 20px fixed width left no room to draw
        # the glyph, so the button was present, enabled and invisible.
        remove_btn.setStyleSheet(
            "QPushButton#VocabRemove { background: transparent; border: none; "
            f"padding: 0; margin: 0; color: {colors.TEXT_TERTIARY}; font-size: 15px; }}"
            f"QPushButton#VocabRemove:hover {{ color: {colors.ERROR}; }}"
        )
        remove_btn.clicked.connect(lambda _checked=False: on_remove(name))
        layout.addWidget(remove_btn)


# Object name shared by every chip. An unqualified `QPushButton` rule in a
# widget stylesheet only ties with the app theme's own `QPushButton` rule, and
# the theme wins the tie -- which is what flattened these pills back to 6px
# rectangles. An ID selector outranks a type selector, so it does not.
CHIP_ID = "Chip"

# A real radius, not a 999px sentinel. Qt does not clamp an oversized
# border-radius, it ignores the declaration outright -- so with the theme
# applied the pills came back as square-cornered rectangles. 14px is past half
# a chip's ~29px height, which is a full pill, and degrades to a rounded
# rectangle rather than to nothing if the chip ever gets taller.
CHIP_RADIUS = 14


def _chip_css(color: str) -> str:
    """A behaviour chip in that behaviour's own colour.

    The colour is vocabulary data, not theme, so it is used as given -- these
    are the same hues the ethogram and the annotated video draw the behaviour
    in, and re-mapping them here would break that correspondence.

    The *text* colour is picked per chip: the vocabulary colours are chosen by
    whoever set the project up, and one fixed text colour cannot be legible on
    both amber and crimson.
    """
    return (
        f"QPushButton#{CHIP_ID} {{ "
        f"background: {color}; color: {readable_text_on(color)}; border: none; "
        f"padding: 7px 16px; border-radius: {CHIP_RADIUS}px; "
        "font-weight: 600; font-size: 13px; "
        "}"
    )


def _chip_warn_css() -> str:
    return (
        f"QPushButton#{CHIP_ID} {{ background: {colors.with_alpha(colors.WARNING, 0.15)}; "
        f"color: {colors.WARNING}; "
        f"border: 1px solid {colors.with_alpha(colors.WARNING, 0.45)}; "
        f"padding: 6px 14px; border-radius: {CHIP_RADIUS}px; font-weight: 500; }}"
        f"QPushButton#{CHIP_ID}:hover {{ background: {colors.with_alpha(colors.WARNING, 0.24)}; }}"
    )


def _chip_unknown_css() -> str:
    return (
        f"QPushButton#{CHIP_ID} {{ background: {colors.SURFACE_2}; color: {colors.TEXT_TERTIARY}; "
        f"border: 1px solid {colors.BORDER}; "
        f"padding: 6px 14px; border-radius: {CHIP_RADIUS}px; font-weight: 500; }}"
        f"QPushButton#{CHIP_ID}:hover {{ background: {colors.BORDER}; color: {colors.TEXT_PRIMARY}; }}"
    )


def _chip_subtle_css() -> str:
    return (
        f"QPushButton#{CHIP_ID} {{ background: transparent; color: {colors.TEXT_TERTIARY}; "
        f"border: 1px solid transparent; padding: 6px 12px; border-radius: {CHIP_RADIUS}px; "
        "font-weight: 500; } "
        f"QPushButton#{CHIP_ID}:hover {{ background: {colors.SURFACE_2}; "
        f"color: {colors.TEXT_PRIMARY}; }}"
    )


# _nav_css is gone: prev/next now take the shared theme's default button,
# which is exactly the quiet surface control they were hand-rolling.
