"""Draw arena perimeters across a folder of videos and write centre zones.

Walks every video in a folder, opens :class:`ArenaDialog` on a frame from each,
and turns the four clicked floor corners into a centre zone of a fixed real
size. Corners are saved as they are collected, so a run can be stopped and
resumed - thirty videos is more than one sitting.

    python tools/arena_zones.py "Z:/.../videos"
    python tools/arena_zones.py "Z:/.../videos" --zone-cm 10 --arena-cm 30
    python tools/arena_zones.py "Z:/.../videos" --report

Zone files are written beside the videos in an ``arena_zones/`` subfolder rather
than over the existing ``*_zone.json``, so results computed against the old
hand-drawn zones stay reproducible. ``--report`` re-reads a finished
``arena_calibration.json`` and prints what it implies, including how far the new
scale moves from whatever ``pose_calibration.json`` holds - useful before
deciding to adopt it, since that scale carries into every cm/s figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glider.vision.arena import SCHEMA_VERSION, ArenaCalibration  # noqa: E402

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
CALIBRATION_NAME = "arena_calibration.json"

#: Fraction into the video to grab the frame the operator first sees. Not the
#: very start: rigs are often still being adjusted, and a hand in shot hides the
#: corner that has to be clicked.
_SEEK_FRACTION = 0.25


def find_videos(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda p: p.name.lower(),
    )


def read_frame(video: Path, index: int | None = None):
    """One frame from *video*, or None if it cannot be read."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            logger.error("Could not open %s", video)
            return None, 0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = int(count * _SEEK_FRACTION) if index is None else index
        if count:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(target, count - 1)))
        ok, frame = capture.read()
        return (frame if ok else None), count
    finally:
        capture.release()


def load_calibrations(path: Path) -> dict[str, ArenaCalibration]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: ArenaCalibration.from_dict(v) for k, v in data.get("arenas", {}).items()}


def save_calibrations(path: Path, arenas: dict[str, ArenaCalibration], zone_cm: float) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "created": datetime.now().isoformat(timespec="seconds"),
                "zone_cm": zone_cm,
                "arenas": {k: v.to_dict() for k, v in sorted(arenas.items())},
            },
            indent=2,
        )
    )


def zone_document(cal: ArenaCalibration, zone_cm: float, name: str = "Zone 1") -> dict:
    """A zone configuration holding just the centred square.

    Matches what :class:`glider.vision.zones.ZoneConfiguration` writes, so the
    result drops into anything that already reads these files. The shape is a
    polygon because under perspective a centred square is a quadrilateral, and
    forcing it to an axis-aligned rectangle would put back the error this whole
    exercise removes.
    """
    width, height = cal.frame_size
    return {
        "zones": [
            {
                "id": str(uuid.uuid4()),
                "name": name,
                "shape": "polygon",
                "vertices": [list(v) for v in cal.centre_zone_vertices(zone_cm)],
                "color": [255, 165, 0],
            }
        ],
        "config_width": width,
        "config_height": height,
    }


def write_zones(arenas: dict[str, ArenaCalibration], out_dir: Path, zone_cm: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, cal in sorted(arenas.items()):
        path = out_dir / f"{stem}_zone.json"
        path.write_text(json.dumps(zone_document(cal, zone_cm), indent=2))
        written.append(path)
    return written


def score_existing(
    folder: Path,
    arenas: dict[str, ArenaCalibration],
    zone_cm: float,
    *,
    keypoint: str = "body_center",
) -> None:
    """Score already-written pose CSVs against the drawn arenas.

    The batch window now scores zones as inference runs, but a cohort tracked
    before the arenas were drawn has CSVs and no zone files. Whether a point is
    inside a polygon needs no pixels, so those sessions can be scored from what
    is already on disk rather than by decoding every video again.
    """
    from glider.vision.pose.batch import find_pose_csv
    from glider.vision.zone_scoring import score_csv, write_zone_csvs, zone_output_dir
    from glider.vision.zones import ZoneConfiguration

    print()
    print(f"{'session':16}{'csv':>28}{'centre_s':>10}{'of_total':>10}{'cover':>8}")
    scored = 0
    for stem, cal in sorted(arenas.items()):
        video = next(
            (
                folder / f"{stem}{ext}"
                for ext in VIDEO_SUFFIXES
                if (folder / f"{stem}{ext}").exists()
            ),
            None,
        )
        csv_path = find_pose_csv(video) if video else None
        if csv_path is None:
            print(f"{stem:16}{'no pose CSV':>28}")
            continue
        config = ZoneConfiguration.from_dict(zone_document(cal, zone_cm))
        try:
            scoring = score_csv(csv_path, config, resolution=cal.frame_size, keypoint=keypoint)
        except Exception as e:
            print(f"{stem:16}{'FAILED':>28}  {e}")
            continue
        write_zone_csvs(scoring, zone_output_dir(video))
        seconds = next(iter(scoring.seconds_in_zone.values()))
        total = scoring.frames_total / scoring.fps if scoring.fps else 0
        pct = seconds / total * 100 if total else 0
        print(
            f"{stem:16}{csv_path.name[:28]:>28}{seconds:10.1f}{pct:9.1f}%{scoring.coverage * 100:7.1f}%"
        )
        scored += 1
    print()
    print(f"Scored {scored} session(s); zone CSVs written to <stem>_zones/ beside each video.")


def report(folder: Path, arenas: dict[str, ArenaCalibration], zone_cm: float) -> None:
    """Print what the drawn perimeters imply, next to the existing scale."""
    existing: dict[str, float] = {}
    pose = folder / "pose_calibration.json"
    if pose.exists():
        for entry in json.loads(pose.read_text()).get("videos", []):
            stem = Path(entry["video"].replace("\\", "/")).stem
            existing[stem] = entry["px_per_mm"] * 10.0

    print(
        f"\n{'session':16}{'px/cm new':>11}{'px/cm old':>11}{'change':>9}{'edge':>7}{'scale':>7}  flags"
    )
    scales, changes = [], []
    for stem, cal in sorted(arenas.items()):
        new = cal.px_per_cm_centre
        residuals = cal.residuals()
        scales.append(new)
        old = existing.get(stem)
        change = f"{(new / old - 1) * 100:+.1f}%" if old else "-"
        if old:
            changes.append(abs(new / old - 1))
        flags = []
        if residuals["suspect"]:
            flags.append("SUSPECT")
        if residuals["clipped"]:
            flags.append("clipped")
        print(
            f"{stem:16}{new:11.2f}{old if old else 0:11.2f}{change:>9}"
            f"{residuals['edge_ratio']:7.2f}{residuals['scale_ratio']:7.2f}  {' '.join(flags)}"
        )

    if not scales:
        return
    spread = max(scales) / min(scales) - 1
    print(f"\n{len(scales)} arenas, zone {zone_cm} cm")
    print(f"  new scale spread : {spread * 100:.1f}%  (glider warns above 10%)")
    if existing:
        old_values = [existing[s] for s in arenas if s in existing]
        if old_values:
            old_spread = max(old_values) / min(old_values) - 1
            print(f"  old scale spread : {old_spread * 100:.1f}%")
    if changes:
        biggest = max(changes) * 100
        print(f"  largest per-video scale change: {biggest:.1f}%")
        print("  Speeds in cm/s scale inversely with px/cm, so adopting this")
        print("  moves every speed figure and the cohort freeze/dart thresholds.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder of videos")
    parser.add_argument("--arena-cm", type=float, default=30.0, help="Arena side (default 30)")
    parser.add_argument("--zone-cm", type=float, default=10.0, help="Centre zone side (default 10)")
    parser.add_argument("--out", type=Path, default=None, help="Zone output dir")
    parser.add_argument("--report", action="store_true", help="Report only, no drawing")
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score existing pose CSVs against the drawn arenas and write zone CSVs",
    )
    parser.add_argument(
        "--keypoint", default="body_center", help="Tracked point deciding occupancy"
    )
    parser.add_argument("--redo", action="store_true", help="Re-draw arenas already saved")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    folder: Path = args.folder
    if not folder.is_dir():
        parser.error(f"Not a folder: {folder}")

    calibration_path = folder / CALIBRATION_NAME
    arenas = load_calibrations(calibration_path)
    out_dir = args.out or folder / "arena_zones"

    if args.report or args.score:
        if not arenas:
            print(f"No arenas saved yet in {calibration_path}")
            return 1
        if args.score:
            score_existing(folder, arenas, args.zone_cm, keypoint=args.keypoint)
        else:
            report(folder, arenas, args.zone_cm)
        return 0

    videos = find_videos(folder)
    if not videos:
        parser.error(f"No videos in {folder}")

    from PyQt6.QtWidgets import QApplication

    from glider.gui.dialogs.arena_dialog import ArenaDialog

    # Held for the lifetime of the loop: dropping the reference closes Qt.
    _app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    todo = [v for v in videos if args.redo or v.stem not in arenas]
    print(f"{len(videos)} videos, {len(arenas)} already drawn, {len(todo)} to do.")

    for i, video in enumerate(todo, 1):
        frame, count = read_frame(video)
        if frame is None:
            logger.warning("Skipping unreadable %s", video.name)
            continue

        dialog = ArenaDialog(
            frame,
            title=f"{video.stem}  ({i}/{len(todo)})",
            frame_count=count,
            on_scrub=lambda idx, v=video: read_frame(v, idx)[0],
        )
        dialog.arena_spin.setValue(args.arena_cm)
        dialog.zone_spin.setValue(args.zone_cm)
        if video.stem in arenas:
            dialog.canvas.set_corners(arenas[video.stem].corners)

        if dialog.exec() != dialog.DialogCode.Accepted:
            print("Stopped. Progress saved.")
            break

        cal = dialog.calibration()
        if cal is not None:
            arenas[video.stem] = cal
            save_calibrations(calibration_path, arenas, args.zone_cm)
            print(f"  {video.stem}: {cal.px_per_cm_centre:.2f} px/cm")

    if not arenas:
        print("Nothing drawn.")
        return 1

    written = write_zones(arenas, out_dir, args.zone_cm)
    print(f"\nWrote {len(written)} zone files to {out_dir}")
    print(f"Corners saved to {calibration_path}")
    report(folder, arenas, args.zone_cm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
