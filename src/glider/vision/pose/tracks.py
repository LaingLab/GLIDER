"""N animals over one video.

Deliberately dumb. It holds a mapping, a frame rate and provenance, and has no
analysis methods of its own -- everything that computes anything takes a single
``PoseData``, so there is never a question of which container to call something
on.

Every slot spans the whole video. A frame where that animal was not found is
NaN, which is already exactly what a dropout means to ``filtering.smooth``,
``arena_gate.gate_to_arena`` and ``zone_scoring``; that is what lets all three
work per-track with no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from glider.vision.pose.core import PoseData

__all__ = ["PoseTracks"]


@dataclass
class PoseTracks:
    """``{slot: PoseData}`` for one video, plus the rate they share."""

    tracks: dict[int, PoseData]
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tracks:
            raise ValueError("PoseTracks needs at least one slot")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive; got {self.fps}")

        slots = sorted(self.tracks)
        if slots != list(range(len(slots))):
            raise ValueError(
                f"slot ids must be contiguous from 0; got {slots}. A gap means an "
                f"animal was dropped upstream, and the individuals written to the "
                f"CSV would then not mean what their names say."
            )

        frame_counts = {p.n_frames for p in self.tracks.values()}
        if len(frame_counts) != 1:
            raise ValueError(
                f"every slot must span the whole video; got n_frames "
                f"{sorted(frame_counts)}. An absent animal is NaN, not a short track."
            )

        names = {tuple(p.keypoint_names) for p in self.tracks.values()}
        if len(names) != 1:
            raise ValueError(
                f"every slot must carry the same keypoint_names; got "
                f"{sorted(list(n) for n in names)}. Animals whose keypoints do not "
                f"line up cannot be compared, written to one DLC CSV, or fed to one "
                f"model -- and the columns would silently mean different body parts "
                f"for different animals."
            )

    @property
    def n_animals(self) -> int:
        return len(self.tracks)

    @property
    def n_frames(self) -> int:
        return next(iter(self.tracks.values())).n_frames

    @property
    def keypoint_names(self) -> list[str]:
        return list(next(iter(self.tracks.values())).keypoint_names)

    @property
    def individuals(self) -> list[str]:
        """DLC ``individuals`` header values, in slot order."""
        return [f"animal{slot}" for slot in sorted(self.tracks)]

    def __getitem__(self, slot: int) -> PoseData:
        return self.tracks[slot]

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self):
        """Slot ids, ascending -- so every consumer writes animals in one order."""
        return iter(sorted(self.tracks))
