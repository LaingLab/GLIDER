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

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
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
from glider.gui.behavior.annotator.trim_bar import TrimBar, compute_window

if TYPE_CHECKING:
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
        for video_path, ann_path in self.videos_meta.items():
            try:
                self.stores[video_path] = AnnotationStore.load_csv(ann_path)
            except (ValueError, OverlapError):
                self.stores[video_path] = AnnotationStore()
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
        self.title_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        self.save_indicator = QLabel("")
        self.save_indicator.setStyleSheet("color: #6b6b6b; font-size: 12px;")
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
        self.trim_hint = QLabel("trim: drag handles · [ ] in · { } out")
        self.trim_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
        self.trim_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.trim_hint)

        self.caption_label = QLabel("(no clip)")
        self.caption_label.setStyleSheet("color: #6b6b6b; font-size: 12px;")
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.caption_label)

        # Label buttons card.
        label_card = QFrame()
        label_card.setStyleSheet(
            "background: white; border: 1px solid #e5e5e5; border-radius: 6px;"
        )
        lcv = QVBoxLayout(label_card)
        lcv.setContentsMargins(16, 14, 16, 14)
        lcv.setSpacing(8)
        h3 = QLabel("LABEL THIS CLIP")
        h3.setStyleSheet("font-size: 12px; font-weight: 600; color: #6b6b6b;")
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
        self.btn_multi.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_multi.setStyleSheet(_chip_warn_css())
        self.btn_multi.clicked.connect(lambda: self._apply_label(MULTI_BEHAVIOR))
        self.btn_unclear = QPushButton("? unclear   U")
        self.btn_unclear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_unclear.setStyleSheet(_chip_unknown_css())
        self.btn_unclear.clicked.connect(lambda: self._apply_label(UNCLEAR))
        self.btn_skip = QPushButton("skip   Space")
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
        self.btn_prev.setStyleSheet(_nav_css())
        self.btn_prev.clicked.connect(lambda: self._go(-1))
        self.btn_next = QPushButton("next →")
        self.btn_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_next.setStyleSheet(_nav_css())
        self.btn_next.clicked.connect(lambda: self._go(+1))
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        main.addLayout(nav)

        self.setStatusBar(QStatusBar())
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
        sidebar.setStyleSheet("background: white; border: 1px solid #e5e5e5; border-radius: 6px;")
        v = QVBoxLayout(sidebar)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)

        # Progress card.
        v.addWidget(_section_header("Progress"))
        self.progress_label = QLabel("clip 0 / 0")
        self.progress_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        v.addWidget(self.progress_label)
        self.labeled_label = QLabel("0 labeled · 0 marked")
        self.labeled_label.setStyleSheet("color: #6b6b6b; font-size: 12px;")
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
        self.vocab_empty_hint.setStyleSheet("color: #9ca3af; font-size: 12px;")
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
    def _save_annotations_for_video(self, video: Path) -> None:
        """Persist the store for `video` to its annotations CSV."""
        store = self.stores[video]
        ann_path = self.videos_meta[video]
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

    def _set_big_label(self, label: str | None) -> None:
        """Show the current clip's label big in the top-right, tinted by its
        vocabulary color. ``None``/empty shows a muted placeholder."""
        if label:
            color = self.vocab.color_for(label)
            self.big_label.setText(label)
            self.big_label.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: 700;")
        else:
            self.big_label.setText("—")
            self.big_label.setStyleSheet("color: #cbd5e1; font-size: 26px; font-weight: 700;")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
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
    # Upper-case in Python — Qt Style Sheets don't support text-transform /
    # letter-spacing (they log "Unknown property" warnings).
    h = QLabel(text.upper())
    h.setStyleSheet("font-size: 12px; font-weight: 600; color: #6b6b6b;")
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
        self.setStyleSheet("background: #f6f6f6; border-radius: 4px; padding: 2px;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background: {color}; border-radius: 5px;")
        layout.addWidget(dot)
        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: 500;")
        layout.addWidget(name_label)
        layout.addStretch(1)
        kbd = QLabel(hotkey)
        kbd.setStyleSheet(
            "color: #6b6b6b; font-family: SF Mono, monospace; font-size: 11px;"
            "background: white; border: 1px solid #e5e5e5; "
            "border-radius: 3px; padding: 0 5px;"
        )
        layout.addWidget(kbd)
        count_label = QLabel(str(count))
        count_label.setMinimumWidth(24)
        count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # NOTE: Qt Style Sheets don't support `font-variant-numeric`
        # (it spams "Unknown property" warnings); the right-alignment +
        # min-width already keep the counts tidy.
        count_label.setStyleSheet("color: #6b6b6b;")
        layout.addWidget(count_label)
        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove_btn.setToolTip(f"Remove {name!r} from vocabulary")
        remove_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            "color: #9ca3af; font-size: 16px; }"
            "QPushButton:hover { color: #b91c1c; }"
        )
        remove_btn.clicked.connect(lambda _checked=False: on_remove(name))
        layout.addWidget(remove_btn)


def _chip_css(color: str) -> str:
    return (
        "QPushButton { "
        f"background: {color}; color: white; border: none; "
        "padding: 7px 14px; border-radius: 999px; "
        "font-weight: 500; font-size: 13px; "
        "}"
    )


def _chip_warn_css() -> str:
    return (
        "QPushButton { background: #fef3c7; color: #92400e; "
        "border: none; padding: 7px 14px; border-radius: 999px; "
        "font-weight: 500; }"
    )


def _chip_unknown_css() -> str:
    return (
        "QPushButton { background: #f1f5f9; color: #475569; "
        "border: none; padding: 7px 14px; border-radius: 999px; "
        "font-weight: 500; }"
    )


def _chip_subtle_css() -> str:
    return (
        "QPushButton { background: transparent; color: #6b6b6b; "
        "border: none; padding: 7px 12px; border-radius: 999px; "
        "font-weight: 500; } "
        "QPushButton:hover { background: #f6f6f6; }"
    )


def _nav_css() -> str:
    return (
        "QPushButton { background: transparent; color: #6b6b6b; "
        "border: 1px solid #e5e5e5; padding: 6px 12px; border-radius: 4px; } "
        "QPushButton:hover { color: #1a1a1a; border-color: #cbd5e1; }"
    )
