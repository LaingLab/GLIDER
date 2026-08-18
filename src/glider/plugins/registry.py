"""Resolve the plugin catalogue from the best source available.

Order is network, then cache, then the copy shipped inside the wheel. Each step
down is a degradation the user should be able to see, so the resolved result
carries which source won and how old it is; the Plugins window prints both.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/LaingLab/glider-plugins/main/index.json"
CACHE_FILENAME = "plugin_index.json"
FETCH_TIMEOUT_SECONDS = 3.0

Fetcher = Callable[[str, float], Awaitable[str]]


@dataclass(frozen=True)
class ResolvedIndex:
    """A catalogue plus the provenance the window has to display."""

    plugins: list[dict[str, Any]] = field(default_factory=list)
    updated: str = ""
    schema_version: str = ""
    source: str = "bundled"  # "network" | "cache" | "bundled"


def _parse(text: str) -> ResolvedIndex | None:
    """Parse an index, returning None rather than raising on anything malformed.

    Callers use None to mean "try the next source". A bad index is a reason to
    fall back, not a reason to fail: the alternative is that one broken file on
    a web server bricks the Plugins window for everyone.

    Entries that are not objects are dropped rather than passed on. Every
    consumer downstream reads an entry with ``entry.get(...)``, so a bare string
    in the list would fail somewhere far from here with nothing naming the
    index. One bad entry costs one row, not the whole catalogue.
    """
    try:
        data = json.loads(text)
        raw = list(data["plugins"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Ignoring malformed plugin index: %s", exc)
        return None

    plugins: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, Mapping):
            plugins.append(dict(entry))
        else:
            logger.warning("Ignoring a plugin index entry that is not an object: %r", entry)

    return ResolvedIndex(
        plugins=plugins,
        updated=str(data.get("updated", "")),
        schema_version=str(data.get("schema_version", "")),
    )


async def _default_fetcher(url: str, timeout: float) -> str:
    """Fetch over HTTP in a worker thread.

    urllib is blocking, and this runs on the Qt event loop via qasync -- calling
    it directly would freeze the UI for the whole timeout.
    """
    import urllib.request

    def _get() -> str:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")

    return await asyncio.to_thread(_get)


def _read_text(path: Path) -> str | None:
    """Read a file, or None if it is not there or cannot be read.

    One call rather than ``exists()`` then ``read_text()``: two trips to a file
    that may live on an SMB share cost twice as much and still race.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


class PluginRegistry:
    def __init__(
        self,
        cache_dir: Path,
        url: str = DEFAULT_INDEX_URL,
        fetcher: Fetcher | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._url = url
        self._fetch = fetcher or _default_fetcher

    @property
    def _cache_path(self) -> Path:
        return self._cache_dir / CACHE_FILENAME

    @staticmethod
    def load_bundled() -> ResolvedIndex:
        """Read the copy shipped in the wheel.

        Unlike the other two sources this one raises. A malformed bundled index
        is a packaging defect that shipped, not a runtime condition to absorb.
        """
        path = Path(__file__).with_name("index.json")
        parsed = _parse(path.read_text(encoding="utf-8"))
        if parsed is None:
            raise ValueError(f"bundled plugin index is malformed: {path}")
        return parsed

    async def resolve(self) -> ResolvedIndex:
        """Network, then cache, then the bundled copy.

        Every filesystem touch here goes through a worker thread for the same
        reason the fetch does: this runs on the Qt event loop via qasync, and
        both the cache and the wheel can sit on a network share where a read is
        not the instant operation ``Path.read_text`` looks like.
        """
        try:
            text = await self._fetch(self._url, FETCH_TIMEOUT_SECONDS)
            parsed = _parse(text)
            if parsed is not None:
                await asyncio.to_thread(self._write_cache, text)
                return ResolvedIndex(
                    plugins=parsed.plugins,
                    updated=parsed.updated,
                    schema_version=parsed.schema_version,
                    source="network",
                )
        except Exception as exc:
            logger.info("Plugin index fetch failed, falling back: %s", exc)

        cached = await asyncio.to_thread(_read_text, self._cache_path)
        if cached is not None:
            parsed = _parse(cached)
            if parsed is not None:
                return ResolvedIndex(
                    plugins=parsed.plugins,
                    updated=parsed.updated,
                    schema_version=parsed.schema_version,
                    source="cache",
                )

        return await asyncio.to_thread(self.load_bundled)

    def _write_cache(self, text: str) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            # A read-only home directory must not break resolution.
            logger.warning("Could not cache the plugin index: %s", exc)
