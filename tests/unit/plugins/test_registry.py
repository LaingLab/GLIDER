"""How the catalogue is resolved, and how it says so.

The resolution order exists because a lab machine may be offline, may have been
offline for a month, or may never have been online. Which source won is not a
debugging detail -- the window shows it, because "why isn't the new plugin
listed" is otherwise unanswerable.
"""

import json

import pytest

from glider.plugins.registry import PluginRegistry

GOOD = {
    "schema_version": "1.0",
    "updated": "2026-08-01",
    "plugins": [{"name": "a", "pypi": "a", "version": "1.0.0", "glider_requires": ">=1.0"}],
}


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


async def test_a_successful_fetch_wins_and_is_cached(cache_dir):
    async def fetch(url, timeout):
        return json.dumps(GOOD)

    reg = PluginRegistry(cache_dir=cache_dir, fetcher=fetch)
    result = await reg.resolve()

    assert result.source == "network"
    assert result.updated == "2026-08-01"
    assert [p["name"] for p in result.plugins] == ["a"]
    assert json.loads((cache_dir / "plugin_index.json").read_text())["updated"] == "2026-08-01"


async def test_cache_wins_when_the_fetch_fails(cache_dir):
    cache_dir.mkdir(parents=True)
    (cache_dir / "plugin_index.json").write_text(json.dumps(GOOD))

    async def fetch(url, timeout):
        raise TimeoutError("no network")

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "cache"
    assert result.updated == "2026-08-01"


async def test_bundled_wins_when_fetch_and_cache_both_fail(cache_dir):
    async def fetch(url, timeout):
        raise TimeoutError("no network")

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "bundled"
    assert any(p["name"] == "glider-harp" for p in result.plugins)


async def test_a_malformed_network_index_falls_through_rather_than_raising(cache_dir):
    """Garbage from the network must not take the app down, and must not be
    cached -- caching it would poison every later run."""

    async def fetch(url, timeout):
        return "{ this is not json"

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "bundled"
    assert not (cache_dir / "plugin_index.json").exists()


async def test_a_malformed_cache_falls_through_to_bundled(cache_dir):
    cache_dir.mkdir(parents=True)
    (cache_dir / "plugin_index.json").write_text("{ not json")

    async def fetch(url, timeout):
        raise TimeoutError("no network")

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "bundled"


async def test_an_index_entry_that_is_not_an_object_is_dropped(cache_dir):
    """`_parse` guarded the shape of the index but not the shape of its entries,
    and every consumer downstream calls `entry.get(...)`."""

    async def fetch(url, timeout):
        return json.dumps({**GOOD, "plugins": ["glider-harp", {"name": "a", "pypi": "a"}]})

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "network"
    assert [p["name"] for p in result.plugins] == ["a"]


def test_the_bundled_index_is_valid_json_and_lists_harp():
    """A packaging guard: the shipped file is the last line of defence, so a
    typo in it is not something to discover at runtime on a lab machine."""
    result = PluginRegistry.load_bundled()

    assert result.schema_version == "1.0"
    assert any(p["name"] == "glider-harp" for p in result.plugins)
