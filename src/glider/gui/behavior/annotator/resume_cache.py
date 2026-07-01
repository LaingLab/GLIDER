"""JSON resume-cache for multi-video annotation sessions.

The annotator samples N clips up front. If the user closes the window and
relaunches with the same arguments, re-sampling would (a) waste a few seconds
and (b) produce a different clip queue if defaults shifted. Instead we write
the sampled queue to `<resume-dir>/.annotate_queue.json` along with a hash of
the sampling inputs.

On next launch:
- If the cache exists AND the input-hash matches -> reuse the cached queue.
- If the hash differs -> overwrite with a freshly sampled queue.
- If the file is missing/corrupt -> silently re-sample.

This module has no Qt / OpenCV / pandas deps so it's trivially testable.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_FILENAME = ".annotate_queue.json"


def resolve_cache_dir(
    *,
    videos: Iterable[Path] | None,
    videos_dir: Path | None,
) -> Path:
    """Pick the directory where the resume-cache file lives.

    Order:
    1. `videos_dir` if provided.
    2. `os.path.commonpath` of the positional videos, when that succeeds AND
       the result is an existing directory.
    3. The parent of the first positional video as a last resort.
    """
    if videos_dir is not None:
        return Path(videos_dir)
    video_list = [Path(v) for v in (videos or [])]
    if not video_list:
        raise ValueError("resolve_cache_dir needs either videos or videos_dir")
    try:
        common = os.path.commonpath([str(v.resolve()) for v in video_list])
        common_path = Path(common)
        if common_path.is_dir():
            return common_path
    except ValueError:
        pass
    return video_list[0].parent


def _hash_inputs(inputs: dict[str, Any]) -> str:
    """Stable SHA1 of the sampling inputs, used as the cache identity."""
    payload = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class ResumeCache:
    """Tiny JSON file that records the sampled queue + an input hash."""

    cache_dir: Path

    @property
    def path(self) -> Path:
        return self.cache_dir / CACHE_FILENAME

    def save(self, *, inputs: dict[str, Any], clip_payload: list[dict[str, Any]]) -> None:
        """Atomically write the cache file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "hash": _hash_inputs(inputs),
            "inputs": inputs,
            "clips": clip_payload,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def load(self, *, inputs: dict[str, Any]) -> dict[str, Any] | None:
        """Return the cached record only if its input-hash matches."""
        if not self.path.exists():
            return None
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if record.get("hash") != _hash_inputs(inputs):
            return None
        return record
