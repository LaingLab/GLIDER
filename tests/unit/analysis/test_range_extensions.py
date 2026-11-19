"""Plugin range measures (glider.range_metrics) and how the epoch catalog offers them."""

from __future__ import annotations

from glider.analysis.epochs import Metric, metric_catalog
from glider.analysis.range_extensions import RangeMetricHost, load_range_metric_providers


class _Provider:
    def __init__(self, name="p", values=None, fail=False):
        self.name, self._values, self.fail = name, values or {}, fail

    def metrics(self):
        return [Metric("p_a", "A", 1), Metric("p_b", "B", 1)]

    def values(self, view, start, end):
        if self.fail:
            raise RuntimeError("boom")
        return self._values


class _EP:
    def __init__(self, name, factory):
        self.name, self._factory = name, factory

    def load(self):
        return self._factory


def test_loader_skips_broken_and_unusable_plugins():
    def broken():
        raise ImportError("x")

    eps = [_EP("b", broken), _EP("a", lambda: [_Provider(), object()])]
    assert [p.name for p in load_range_metric_providers(eps)] == ["p"]


def test_only_answered_metrics_are_offered_and_a_failing_provider_is_dropped():
    bad = _Provider("bad", fail=True)
    host = RangeMetricHost([bad, _Provider(values={"p_a": 2.0})])
    row = host.columns(None, 0, 10)
    assert row["p_a"] == 2.0 and "p_b" not in row
    assert [m.key for m in row["_extra_metrics"]] == ["p_a"]
    assert host.columns(None, 0, 10)["p_a"] == 2.0  # bad stays dropped, no re-raise
    assert RangeMetricHost([_Provider()]).columns(None, 0, 1) == {}


def test_catalog_offers_plugin_metrics_before_the_thresholds():
    rows = [
        {"session": "a", **RangeMetricHost([_Provider(values={"p_a": 1.0})]).columns(None, 0, 1)},
        {"session": "b"},
    ]
    keys = [m.key for m in metric_catalog(rows)]
    assert keys.index("p_a") == keys.index("freeze_threshold") - 1
    assert "p_a" not in [m.key for m in metric_catalog([{"session": "b"}])]
