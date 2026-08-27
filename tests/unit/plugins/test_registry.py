"""How the catalogue is resolved, and how it says so.

The resolution order exists because a lab machine may be offline, may have been
offline for a month, or may never have been online. Which source won is not a
debugging detail -- the window shows it, because "why isn't the new plugin
listed" is otherwise unanswerable.
"""

import json
from pathlib import Path

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


# --- the fetch is capped ------------------------------------------------------
#
# A curated catalogue is kilobytes. `response.read()` unbounded meant a
# misconfigured or hostile server could hand back gigabytes and the whole
# thing was buffered before parsing. Over the cap is a failed fetch, which
# falls through to cache/bundled like every other malformed-network case.


def test_an_index_within_the_cap_passes_through_untouched():
    from glider.plugins.registry import _within_cap

    assert _within_cap(b'{"plugins": []}') == b'{"plugins": []}'


def test_an_oversize_index_is_refused():
    from glider.plugins.registry import INDEX_MAX_BYTES, _within_cap

    with pytest.raises(ValueError, match="plugin index"):
        _within_cap(b"x" * (INDEX_MAX_BYTES + 1))


async def test_the_default_fetcher_enforces_the_cap(tmp_path):
    """Through the real fetcher, via a file:// URL -- no network, no fake."""
    from glider.plugins.registry import INDEX_MAX_BYTES, _default_fetcher

    oversize = tmp_path / "index.json"
    oversize.write_bytes(b"x" * (INDEX_MAX_BYTES + 1))

    with pytest.raises(ValueError, match="plugin index"):
        await _default_fetcher(oversize.as_uri(), 1.0)


async def test_the_default_fetcher_still_reads_a_normal_index(tmp_path):
    from glider.plugins.registry import _default_fetcher

    path = tmp_path / "index.json"
    path.write_text(json.dumps(GOOD), encoding="utf-8")

    text = await _default_fetcher(path.as_uri(), 1.0)

    assert json.loads(text)["updated"] == "2026-08-01"


class TestTheBundledCatalogueMatchesTheRepository:
    """Every in-repo plugin must be offerable from the Plugins window.

    glider-sleap-nn shipped to PyPI and did not appear in the window, because
    adding a plugin means touching two places nothing connected: the `plugins/`
    directory and this catalogue. Publishing succeeded, CI was green, and the
    only symptom was a plugin nobody could find.
    """

    @staticmethod
    def _plugins_dir() -> Path:
        return Path(__file__).resolve().parents[3] / "plugins"

    @classmethod
    def _bundled(cls) -> dict:
        return {p["name"]: p for p in PluginRegistry.load_bundled().plugins}

    @classmethod
    def _in_repo(cls) -> set[str]:
        return {
            d.name
            for d in cls._plugins_dir().iterdir()
            if d.is_dir() and (d / "pyproject.toml").is_file()
        }

    def test_every_in_repo_plugin_is_offered(self):
        missing = self._in_repo() - set(self._bundled())
        assert not missing, (
            "these plugins exist in plugins/ but are not offered in "
            f"src/glider/plugins/index.json: {sorted(missing)}"
        )

    def test_the_catalogue_names_no_plugin_that_is_not_there(self):
        """A stale entry offers an install that cannot work.

        Only entries whose homepage points into this tree are expected on disk:
        glider-harp is also published from a repository of its own.
        """
        expected_here = {
            name
            for name, entry in self._bundled().items()
            if "/tree/main/plugins/" in (entry.get("homepage") or "")
        }
        assert not (expected_here - self._in_repo()), (
            "catalogue points at plugins/ folders that do not exist: "
            f"{sorted(expected_here - self._in_repo())}"
        )

    def test_catalogue_requirements_are_copied_verbatim_or_not_at_all(self):
        """A catalogue requirement is appended to the install command as-is.

        It exists for one reason (see `installer.install_command`): uv honours
        a pre-release marker only on a *direct* requirement. So it is a
        duplicate of a line in the plugin's own pyproject, and a duplicate that
        has drifted is worse than no duplicate:

        * dropping an environment marker installs on a platform the plugin
          deliberately excluded -- `tensorflow-cpu` has no macOS wheel, and the
          catalogue once asked pip for it there unconditionally;
        * keeping an old pin re-permits a version the plugin has since
          excluded, undoing from here a pin tightened over there.

        Verbatim or absent. Nothing in between.
        """
        import re
        import tomllib

        for name, entry in self._bundled().items():
            pyproject = self._plugins_dir() / name / "pyproject.toml"
            if not pyproject.is_file():
                continue
            declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"].get(
                "dependencies", []
            )
            by_package = {re.split(r"[<>=!~\[]", d, maxsplit=1)[0].strip(): d for d in declared}
            for requirement in entry.get("requirements") or []:
                package = re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip()
                if package in by_package:
                    assert requirement == by_package[package], (
                        f"{name}: the catalogue asks pip for {requirement!r} but its "
                        f"pyproject declares {by_package[package]!r}"
                    )
