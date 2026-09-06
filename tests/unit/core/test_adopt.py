"""Adopt: moving a real cohort into the canonical layout without losing it.

These folders are irreplaceable and live on network shares. So the tests that
matter are not "does it move files" - they are the refusals: a plan with a
collision moves nothing, a destination is never overwritten, an interrupted run
resumes, and the whole thing reverses.

The sidecar case has its own test. A pose CSV and its .meta.json travelling to
different folders is what silently erased freezing from eleven recordings, and
adoption is a moment when it could happen again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from glider.core.adopt import (
    REVERSAL_NAME,
    apply_plan,
    plan_adopt,
    revert,
    supersede,
)


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def flat(tmp_path):
    """A cohort folder shaped like the ones that exist: four spellings of one
    session, outputs grouped by kind, pose CSVs loose at the top."""
    _touch(tmp_path / "Test 1.mp4")
    _touch(tmp_path / "Test 1DLC_train-6.csv")
    _touch(tmp_path / "Test 1DLC_train-6.meta.json")
    _touch(tmp_path / "test1_zone.json")
    _touch(tmp_path / "Test_1_arena.json")
    _touch(tmp_path / "Test 1" / "final_outputs" / "stats.csv")
    _touch(tmp_path / "Test 1" / "final_outputs" / "run.json")
    _touch(tmp_path / "cohort_speed.json")
    return tmp_path


def _moved_to(plan, name: str) -> Path | None:
    for move in plan.moves:
        if move.source.name == name:
            return move.destination
    return None


class TestPlanning:
    def test_it_plans_without_touching_anything(self, flat):
        before = sorted(p.name for p in flat.rglob("*"))
        plan = plan_adopt(flat)
        assert plan.moves
        assert sorted(p.name for p in flat.rglob("*")) == before

    def test_the_recording_lands_in_its_session_folder(self, flat):
        plan = plan_adopt(flat)
        assert _moved_to(plan, "Test 1.mp4") == flat / "sessions" / "Test 1" / "Test 1.mp4"

    def test_spelling_drift_is_canonicalised(self, flat):
        plan = plan_adopt(flat)
        assert _moved_to(plan, "test1_zone.json").name == "Test 1_zone.json"
        assert _moved_to(plan, "Test_1_arena.json").name == "Test 1_arena.json"

    def test_analysis_outputs_are_attributed_by_their_folder(self, flat):
        # stats.csv is named for what it is, never for the session, so it is
        # only reachable through the folder it sits in.
        plan = plan_adopt(flat)
        assert (
            _moved_to(plan, "stats.csv") == flat / "sessions" / "Test 1" / "analysis" / "stats.csv"
        )

    def test_cohort_files_stay_at_the_root(self, flat):
        plan = plan_adopt(flat)
        assert _moved_to(plan, "cohort_speed.json") is None
        assert flat / "cohort_speed.json" in plan.unclassified

    def test_an_empty_folder_is_refused_rather_than_planned(self, tmp_path):
        plan = plan_adopt(tmp_path)
        assert not plan.is_safe
        assert plan.moves == []


class TestTheSidecarTravelsWithItsCsv:
    """The relationship whose breakage cost eleven recordings their scores."""

    def test_they_land_in_the_same_folder(self, flat):
        plan = plan_adopt(flat)
        csv = _moved_to(plan, "Test 1DLC_train-6.csv")
        meta = _moved_to(plan, "Test 1DLC_train-6.meta.json")
        assert csv is not None and meta is not None
        assert csv.parent == meta.parent

    def test_they_keep_a_matching_stem(self, flat):
        plan = plan_adopt(flat)
        csv = _moved_to(plan, "Test 1DLC_train-6.csv")
        meta = _moved_to(plan, "Test 1DLC_train-6.meta.json")
        assert meta.name == csv.stem + ".meta.json"

    def test_the_pair_is_still_resolvable_after_the_move(self, flat):
        from glider.core.session import Session

        apply_plan(plan_adopt(flat))
        session = Session(flat, "Test 1")
        assert session.pose_csv is not None
        assert session.pose_meta is not None
        assert session.pose_meta.parent == session.pose_csv.parent


class TestRefusals:
    def test_a_collision_between_two_sources_refuses_the_plan(self, flat):
        # Two spellings of the same artifact would become one file.
        _touch(flat / "Test 1_zone.json")
        plan = plan_adopt(flat)
        assert not plan.is_safe
        assert any("would both become" in p for p in plan.problems)

    def test_an_existing_destination_refuses_the_plan(self, flat):
        _touch(flat / "sessions" / "Test 1" / "Test 1.mp4", "something else")
        plan = plan_adopt(flat)
        assert not plan.is_safe
        assert any("already exists" in p for p in plan.problems)

    def test_an_unsafe_plan_moves_nothing(self, flat):
        _touch(flat / "Test 1_zone.json")
        plan = plan_adopt(flat)
        before = sorted(str(p) for p in flat.rglob("*"))
        result = apply_plan(plan)
        assert not result.ok
        assert result.moved == []
        assert sorted(str(p) for p in flat.rglob("*")) == before

    def test_a_refused_plan_writes_no_reversal(self, flat):
        _touch(flat / "Test 1_zone.json")
        apply_plan(plan_adopt(flat))
        assert not (flat / REVERSAL_NAME).exists()

    def test_test_15_is_not_filed_under_test_1(self, tmp_path):
        # The prefix trap. Filing half a cohort under the wrong animal is
        # exactly the class of error that mislabels an experiment.
        _touch(tmp_path / "Test 1.mp4")
        _touch(tmp_path / "Test 15.mp4")
        _touch(tmp_path / "Test 15_zone.json")
        plan = plan_adopt(tmp_path)
        assert _moved_to(plan, "Test 15_zone.json").parent.name == "Test 15"


class TestApplying:
    def test_it_moves_the_files(self, flat):
        result = apply_plan(plan_adopt(flat))
        assert result.ok
        assert (flat / "sessions" / "Test 1" / "Test 1.mp4").exists()
        assert not (flat / "Test 1.mp4").exists()

    def test_it_writes_the_reversal_before_moving(self, flat):
        result = apply_plan(plan_adopt(flat))
        assert result.reversal_path.exists()
        data = json.loads(result.reversal_path.read_text())
        assert len(data["moves"]) == len(result.moved)

    def test_content_survives(self, tmp_path):
        _touch(tmp_path / "Test 1.mp4")
        _touch(tmp_path / "Test 1DLC_train-6.csv", "frame,x\n1,2\n")
        apply_plan(plan_adopt(tmp_path))
        moved = tmp_path / "sessions" / "Test 1" / "Test 1DLC_train-6.csv"
        assert moved.read_text() == "frame,x\n1,2\n"

    def test_adoption_is_idempotent(self, flat):
        apply_plan(plan_adopt(flat))
        second = plan_adopt(flat)
        assert second.moves == [] or not second.is_safe


class TestResumable:
    def test_an_already_done_move_is_not_an_error(self, flat):
        plan = plan_adopt(flat)
        # Simulate an interrupted run: one move already happened.
        first = plan.moves[0]
        first.destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(first.source, first.destination)

        result = apply_plan(plan)
        assert result.ok
        assert first in result.already_done
        assert first not in result.moved

    def test_a_vanished_source_with_no_destination_fails_loudly(self, flat):
        plan = plan_adopt(flat)
        plan.moves[0].source.unlink()
        result = apply_plan(plan)
        assert not result.ok
        assert result.failed == plan.moves[0]

    def test_it_stops_at_the_first_failure(self, flat):
        plan = plan_adopt(flat)
        plan.moves[1].source.unlink()
        result = apply_plan(plan)
        assert not result.ok
        assert len(result.moved) == 1


class TestRevert:
    def test_it_puts_everything_back(self, flat):
        before = sorted(str(p.relative_to(flat)) for p in flat.rglob("*") if p.is_file())
        result = apply_plan(plan_adopt(flat))
        revert(result.reversal_path)
        after = sorted(
            str(p.relative_to(flat))
            for p in flat.rglob("*")
            if p.is_file() and p.name != REVERSAL_NAME
        )
        assert after == before

    def test_it_refuses_to_overwrite_something_recreated_since(self, flat):
        result = apply_plan(plan_adopt(flat))
        _touch(flat / "Test 1.mp4", "a different recording")
        reverted = revert(result.reversal_path)
        assert not reverted.ok
        assert (flat / "Test 1.mp4").read_text() == "a different recording"

    def test_reverting_twice_is_harmless(self, flat):
        result = apply_plan(plan_adopt(flat))
        assert revert(result.reversal_path).ok
        second = revert(result.reversal_path)
        assert second.ok
        assert second.moved == []

    def test_an_unreadable_reversal_is_reported_not_raised(self, tmp_path):
        (tmp_path / REVERSAL_NAME).write_text("{ not json")
        result = revert(tmp_path / REVERSAL_NAME)
        assert not result.ok
        assert "could not read" in result.error


class TestSupersede:
    def test_it_moves_aside_rather_than_deleting(self, tmp_path):
        target = _touch(tmp_path / "old_zone.json", "old")
        moved = supersede(target, tmp_path)
        assert not target.exists()
        assert moved.read_text() == "old"
        assert "superseded" in moved.parts

    def test_two_files_of_one_name_both_survive(self, tmp_path):
        first = supersede(_touch(tmp_path / "a" / "stats.csv", "one"), tmp_path)
        second = supersede(_touch(tmp_path / "b" / "stats.csv", "two"), tmp_path)
        assert first != second
        assert {first.read_text(), second.read_text()} == {"one", "two"}

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert supersede(tmp_path / "nope.json", tmp_path) is None


class TestDescribe:
    def test_the_plan_reads_as_a_mapping(self, flat):
        text = plan_adopt(flat).describe()
        assert "Test 1.mp4" in text
        assert "->" in text

    def test_a_refused_plan_says_so(self, flat):
        _touch(flat / "Test 1_zone.json")
        assert "REFUSED" in plan_adopt(flat).describe()
