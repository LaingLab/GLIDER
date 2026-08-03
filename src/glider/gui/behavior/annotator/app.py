"""Annotator entry point.

Run with ``python -m glider.gui.behavior.annotator.app`` (or launch it
from the GLIDER behavior-analysis tools).

The annotator's flow is:

1. Load the pose CSV and run :func:`propose_clips_multi` to pick
   ``n_clips`` diverse, distinctive clips spread across the feature
   space of one or more sessions.
2. Open a window that plays each clip on loop while the user assigns
   a label (or marks it ``multi-behavior`` / ``unclear`` / skip).
3. Persist labels to ``<stem>_annotations.csv`` next to each video.
   GLIDER's behavior training pipeline (:mod:`glider.analysis.behavior.pipeline`)
   consumes those files directly.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from glider.analysis.behavior.annotations import AnnotationStore
from glider.analysis.behavior.features import FeatureSpec
from glider.analysis.behavior.vocabulary import Vocabulary
from glider.gui.behavior.annotator.capture_cache import VideoCaptureCache
from glider.gui.behavior.annotator.resume_cache import ResumeCache, resolve_cache_dir
from glider.gui.behavior.annotator.sampler import (
    ProposedClip,
    propose_clips_multi,
    zones_to_clips,
)


def annotation_path_for(pose_csv: Path) -> Path:
    """Annotation CSV location for a session: a sibling of the POSE CSV.

    This MUST match where the behavior training pipeline looks for
    annotations — training resolves them as ``pose_csv.parent / f"{pose_csv.stem}
    _annotations.csv"``. Deriving the annotator's path the same way keeps
    the two in sync even when videos and pose CSVs live in different
    directories (``--videos-dir`` ≠ ``--pose-csv-dir``); otherwise the
    annotator would write next to the video while train reads next to the
    pose CSV, and edits would silently never reach training.
    """
    pose_csv = Path(pose_csv)
    return pose_csv.parent / f"{pose_csv.stem}_annotations.csv"


def build_review_clips(videos_meta: dict[Path, Path], fps: float) -> list[ProposedClip]:
    """Build the review clip list: every saved zone across all videos
    becomes a replayable clip. Missing annotation files yield no clips."""
    clips: list[ProposedClip] = []
    for video, ann_path in videos_meta.items():
        store = AnnotationStore.load_csv(ann_path)
        clips.extend(zones_to_clips(store, video, fps))
    return clips


def make_more_sampler(
    pose_sessions: list[tuple[Path, Path]],  # (video, pose_csv) pairs
    *,
    spec: FeatureSpec | None = None,
    window: int = 30,
    fps: float = 30.0,
    spatial_weight: float = 1.0,
    min_frame_gap: int | None = None,
    base_seed: int = 1042,
):
    """Return a ``sample(n) -> list[ProposedClip]`` callable for the
    render-more button. Each call advances the seed so repeated presses
    surface different picks.

    Defaults match the annotator's own, so a caller that exposes none of the
    sampling knobs (the Behavior Analysis window's Annotate tab) still gets
    the button rather than having to restate them."""
    pairs = [(p, v) for v, p in pose_sessions]  # (pose_csv, video) for sampler
    state = {"seed": int(base_seed)}

    def sample(n: int) -> list[ProposedClip]:
        state["seed"] += 1
        n_total = max(int(n), len(pairs))
        return propose_clips_multi(
            sessions=pairs,
            n_clips_total=n_total,
            spec=spec,
            window=window,
            fps=fps,
            random_state=state["seed"],
            spatial_weight=spatial_weight,
            min_frame_gap=min_frame_gap,
        )

    return sample


def run(
    sessions: list[tuple[Path, Path]],  # (video_path, pose_csv) pairs
    vocab_path: str | Path | None = None,
    n_clips: int = 50,
    window: int = 30,
    fps: float = 30.0,
    body_axis: tuple[int, int] = (0, -1),
    random_state: int = 42,
    spatial_weight: float = 1.0,
    min_frame_gap: int | None = None,
    review: bool = False,
    exclude_labeled: bool = False,
    argv: Sequence[str] | None = None,
) -> int:
    """Launch the annotator GUI on one or more (video, pose_csv) sessions.

    When ``review`` is True, skip k-means sampling and load the existing
    annotation zones as the clip list (all already labelled). Pose CSVs
    that exist enable the in-GUI "render more" button; missing ones just
    disable it.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError as e:
        print(
            "PyQt6 isn't installed. Install with: pip install glider[pc]",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        print(
            "OpenCV isn't installed. Install with: pip install opencv-python",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    if not sessions:
        raise ValueError("run() requires at least one (video, pose_csv) session")
    sessions = [(Path(v), Path(p)) for v, p in sessions]
    primary_video = sessions[0][0]

    spec = FeatureSpec(body_axis=body_axis)

    # Annotations live next to the POSE CSV — the same place train reads
    # them — so edits made here always reach training even when videos and
    # pose CSVs are in different directories.
    videos_meta: dict[Path, Path] = {v: annotation_path_for(p) for v, p in sessions}

    clip_sampler = None
    if review:
        # Review mode: load existing zones as clips; sampler (if pose CSVs
        # exist) backs the render-more button. No resume cache.
        clips = build_review_clips(videos_meta, fps)
        print(f"  review mode: loaded {len(clips)} labelled clip(s)")
        pose_sessions = [(v, p) for v, p in sessions if Path(p).exists()]
        if pose_sessions:
            clip_sampler = make_more_sampler(
                pose_sessions,
                spec=spec,
                window=window,
                fps=fps,
                spatial_weight=spatial_weight,
                min_frame_gap=min_frame_gap,
                base_seed=random_state + 1000,
            )
        else:
            print("  (no pose CSVs found — 'render more' disabled)")
    else:
        # Resume-cache check, then sampler call.
        cache_inputs = {
            "videos": sorted(str(v) for v, _p in sessions),
            "n_clips": int(n_clips),
            "window": int(window),
            "fps": float(fps),
            "random_state": int(random_state),
            "spatial_weight": float(spatial_weight),
            "min_frame_gap": min_frame_gap,
            "exclude_labeled": bool(exclude_labeled),
        }
        cache_dir = resolve_cache_dir(videos=[v for v, _p in sessions], videos_dir=None)
        resume = ResumeCache(cache_dir)
        cached = resume.load(inputs=cache_inputs)
        if cached is not None:
            clips = [ProposedClip(**c) for c in cached["clips"]]
            print(f"  resumed {len(clips)} clips from {resume.path.name}")
        else:
            pairs = [(p, v) for v, p in sessions]  # (pose_csv, video) for sampler
            # When excluding labelled regions, load each session's existing
            # zones so the sampler won't re-cover already-annotated frames.
            exclude_zones_by_session = None
            if exclude_labeled:
                exclude_zones_by_session = []
                n_zones = 0
                for v, _p in sessions:
                    store = AnnotationStore.load_csv(videos_meta[v])
                    zlist = [(z.start_frame, z.end_frame) for z in store]
                    n_zones += len(zlist)
                    exclude_zones_by_session.append(zlist)
                print(f"  excluding {n_zones} already-labelled zone(s) from sampling")
            print(f"Sampling {n_clips} clips across {len(sessions)} videos ...")
            clips = propose_clips_multi(
                sessions=pairs,
                n_clips_total=n_clips,
                spec=spec,
                window=window,
                fps=fps,
                random_state=random_state,
                spatial_weight=spatial_weight,
                min_frame_gap=min_frame_gap,
                exclude_zones_by_session=exclude_zones_by_session,
            )
            resume.save(
                inputs=cache_inputs,
                clip_payload=[c.__dict__ for c in clips],
            )

    # Vocabulary lookup: prefer explicit --vocab, fall back to a sibling
    # of the FIRST video (primary). Note: the vocab fallback intentionally
    # ties the saved YAML's name to the lexically-first video in the
    # session list, so multi-video sessions get a single shared vocab file
    # named after that video.
    vocab = Vocabulary()
    resolved_vocab_path: Path | None = None
    if vocab_path is not None:
        resolved_vocab_path = Path(vocab_path)
        if resolved_vocab_path.exists():
            try:
                vocab = Vocabulary.load(resolved_vocab_path)
            except Exception as e:  # noqa: BLE001
                print(f"  warning: couldn't load vocabulary {vocab_path}: {e}")
    else:
        sibling = primary_video.parent / f"{primary_video.stem}_behaviors.yaml"
        if sibling.exists():
            try:
                vocab = Vocabulary.load(sibling)
                resolved_vocab_path = sibling
            except Exception:  # noqa: BLE001
                pass

    capture_cache = VideoCaptureCache(max_open=3)

    # Deferred so the PyQt6 ImportError check above can run first — importing
    # AnnotatorWindow at module top would pull PyQt6 and produce a less
    # friendly traceback for users who haven't installed the [ui] extra.
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow

    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("Behavior Annotator")
    app.setOrganizationName("GLIDER")
    window_widget = AnnotatorWindow(
        clips=clips,
        videos_meta=videos_meta,
        fps=fps,
        vocab=vocab,
        vocab_path=resolved_vocab_path,
        capture_cache=capture_cache,
        clip_sampler=clip_sampler,
    )
    window_widget.show()
    window_widget.warn_about_load_errors()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run([(Path("video.mp4"), Path("pose.csv"))]))
