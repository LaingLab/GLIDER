"""The JSON beside a retrieval video, and the protocol beside its cohort."""

from __future__ import annotations

import json

import pytest

from glider.analysis.behavior.retrieval import RetrievalSettings
from glider.analysis.behavior.retrieval_setup import (
    RetrievalProtocol,
    RetrievalSetup,
    SidecarError,
    load_protocol,
    load_setup,
    protocol_path,
    save_protocol,
    save_setup,
    setup_path,
)
from glider.vision.zones import Zone, ZoneShape


def _zone(x=0.1):
    return Zone("n", "nest", ZoneShape.RECTANGLE, [(x, 0.1), (x + 0.2, 0.3)])


def test_paths(tmp_path):
    assert setup_path(tmp_path / "litter1_pre.avi") == tmp_path / "litter1_pre.retrieval.json"
    assert protocol_path(tmp_path) == tmp_path / "retrieval_protocol.json"


def test_missing_files_mean_defaults(tmp_path):
    assert load_setup(tmp_path / "v.retrieval.json") == RetrievalSetup()
    p = load_protocol(tmp_path)
    assert (p.cutoff_s, p.nose_keypoint, p.in_nest_s, p.contact_cm, p.carry_speed_cm_s) == (
        300.0,
        "nose",
        2.0,
        1.0,
        3.0,
    )
    assert p.settings() == RetrievalSettings()


def test_setup_round_trips_and_keeps_unknown_keys(tmp_path):
    path = tmp_path / "v.retrieval.json"
    path.write_text(
        json.dumps(
            {"version": 1, "trial_start_s": 4.0, "nests": [], "px_per_mm": 2.0, "lab_note": "x"}
        )
    )
    setup = load_setup(path)
    setup.set_nest(0.0, _zone())
    setup.set_nest(252.0, _zone(0.5))
    save_setup(path, setup)
    back = load_setup(path)
    assert back.trial_start_s == 4.0 and back.px_per_mm == 2.0
    assert [p.from_s for p in back.nests] == [0.0, 252.0]
    assert json.loads(path.read_text())["lab_note"] == "x"


def test_the_first_nest_always_starts_at_zero():
    setup = RetrievalSetup()
    setup.set_nest(30.0, _zone())
    assert setup.nests[0].from_s == 0.0
    setup.set_nest(60.0, _zone(0.5))
    setup.remove_nest(0)
    assert [p.from_s for p in setup.nests] == [0.0]


def test_nest_at_picks_the_latest_period_started():
    setup = RetrievalSetup()
    setup.set_nest(0.0, _zone())
    setup.set_nest(100.0, _zone(0.5))
    assert setup.nest_at(99.9).from_s == 0.0
    assert setup.nest_at(100.0).from_s == 100.0
    assert RetrievalSetup().nest_at(5.0) is None


def test_redrawing_at_the_same_time_replaces_that_period():
    setup = RetrievalSetup()
    setup.set_nest(0.0, _zone())
    setup.set_nest(0.0, _zone(0.5))
    assert len(setup.nests) == 1 and setup.nests[0].zone.vertices[0][0] == 0.5


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        "[1, 2]",
        '{"version": 99}',
        '{"version": 1, "nests": [{"from_s": "x"}]}',
        '{"version": 1, "nests": [{"from_s": 0, "zone": "x"}]}',
        '{"version": 1, "nests": [{"from_s": 0, "zone": null}]}',
        '{"version": 1, "nests": [{"from_s": 0, "zone": []}]}',
        '{"version": 1, "trial_start_s": 1' + "0" * 400 + "}",
        # Parseable, but no GLIDER wrote it: each would score nonsense or raise.
        '{"version": 1, "trial_start_s": -1}',
        '{"version": 1, "trial_start_s": NaN}',
        '{"version": 1, "px_per_mm": NaN}',
        '{"version": 1, "px_per_mm": -2}',
        '{"version": 1, "px_per_mm": 0}',
        '{"version": 1, "nests": [{"from_s": Infinity, "zone": {"vertices": [[0, 0], [1, 1]]}}]}',
        '{"version": 1, "nests": [{"from_s": -3, "zone": {"vertices": [[0, 0], [1, 1]]}}]}',
        '{"version": 1, "nests": [{"from_s": 0, "zone": {"vertices": [[NaN, 0], [1, 1]]}}]}',
        '{"version": 1, "nests": [{"from_s": 0, "zone": {"vertices": [["a", 0], [1, 1]]}}]}',
        '{"version": 1, "nests": [{"from_s": 0, "zone": {"vertices": [[0, 0, 0], [1, 1]]}}]}',
        # Past Python's 4300-digit int limit, and nested past its recursion limit.
        pytest.param('{"version": 1, "trial_start_s": ' + "1" * 5000 + "}", id="5000-digits"),
        pytest.param('{"version": 1, "x": ' + "[" * 10**5 + "]" * 10**5 + "}", id="deep"),
    ],
)
def test_an_unreadable_setup_raises_and_is_not_touched(tmp_path, text):
    path = tmp_path / "v.retrieval.json"
    path.write_text(text)
    with pytest.raises(SidecarError, match="v.retrieval.json"):
        load_setup(path)
    assert path.read_text() == text


def test_protocol_round_trip(tmp_path):
    save_protocol(tmp_path, RetrievalProtocol(cutoff_s=600.0, nose_keypoint="0", contact_cm=1.5))
    p = load_protocol(tmp_path)
    assert (p.cutoff_s, p.nose_keypoint, p.contact_cm) == (600.0, "0", 1.5)
    assert p.settings().contact_cm == 1.5


def test_nests_are_read_in_time_order_and_the_first_starts_at_zero(tmp_path):
    path = tmp_path / "v.retrieval.json"
    zone = _zone().to_dict()
    nests = [{"from_s": 90.0, "zone": zone}, {"from_s": 5.0, "zone": zone}]
    path.write_text(json.dumps({"version": 1, "nests": nests}))
    assert [p.from_s for p in load_setup(path).nests] == [0.0, 90.0]


@pytest.mark.parametrize(
    "entry",
    [
        '"cutoff_s": 0',
        '"cutoff_s": -5',
        '"cutoff_s": NaN',
        '"in_nest_s": 0',
        '"contact_cm": -1',
        '"carry_speed_cm_s": Infinity',
    ],
)
def test_an_unusable_protocol_raises_and_is_not_touched(tmp_path, entry):
    text = '{"version": 1, ' + entry + "}"
    protocol_path(tmp_path).write_text(text)
    with pytest.raises(SidecarError, match="retrieval_protocol.json"):
        load_protocol(tmp_path)
    assert protocol_path(tmp_path).read_text() == text
