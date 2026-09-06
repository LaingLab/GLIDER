"""Project: the manifest that says which mouse a recording is.

Nothing on disk records subject identity, and it cannot be recovered from the
recordings: inferring it from distance rank agreed with the truth on 0 of 15
videos and would have mislabelled a whole cohort. So the manifest is the only
source, and the tests that matter are about it being trustworthy - round-trips
without loss, refuses to half-say things, and models a crossover correctly.
"""

from __future__ import annotations

import json

import pytest

from glider.core.experiment_session import Subject
from glider.core.project import (
    MANIFEST_NAME,
    PROJECT_SCHEMA_VERSION,
    Project,
    ProjectError,
    Provenance,
)


@pytest.fixture
def cohort(tmp_path):
    """Two mice, two days each - the crossover this was written for."""
    project = Project(root=tmp_path, name="TRH open field")
    for subject_id, sex in (("M001", "Male"), ("M002", "Male")):
        subject = Subject(subject_id=subject_id, sex=sex, strain="C57BL/6J")
        project.set_session(f"{subject_id} d1", subject=subject, group="saline", day="1")
        project.set_session(f"{subject_id} d2", subject=subject, group="TRH", day="2")
    # Counterbalanced: the second animal gets the arms the other way round.
    project.set_session("M002 d1", group="TRH")
    project.set_session("M002 d2", group="saline")
    return project


class TestRoundTrip:
    def test_a_saved_manifest_reloads_identically(self, cohort, tmp_path):
        cohort.save()
        again = Project.load(tmp_path)
        assert again.name == cohort.name
        assert again.sessions == cohort.sessions
        assert {k: s.subject_id for k, s in again.subjects.items()} == {
            "M001": "M001",
            "M002": "M002",
        }

    def test_it_writes_where_load_looks(self, cohort, tmp_path):
        assert cohort.save() == tmp_path / MANIFEST_NAME

    def test_the_manifest_is_readable_json(self, cohort, tmp_path):
        cohort.save()
        data = json.loads((tmp_path / MANIFEST_NAME).read_text())
        assert data["schema_version"] == PROJECT_SCHEMA_VERSION
        assert data["sessions"]["M001 d1"]["group"] == "saline"

    def test_saving_twice_keeps_the_original_creation_time(self, cohort):
        first = json.loads(cohort.save().read_text())["created_at"]
        second = json.loads(cohort.save().read_text())["created_at"]
        assert first == second


class TestCrossover:
    """One mouse, both arms. Getting this wrong makes the design
    unrepresentable - or worse, representable incorrectly."""

    def test_one_subject_spans_both_arms(self, cohort):
        assert cohort.sessions_for_subject("M001") == ["M001 d1", "M001 d2"]
        assert cohort.group_for("M001 d1") == "saline"
        assert cohort.group_for("M001 d2") == "TRH"

    def test_the_subject_is_the_same_object_both_days(self, cohort):
        assert cohort.subject_for("M001 d1") is cohort.subject_for("M001 d2")

    def test_group_is_not_read_off_the_subject(self, tmp_path):
        # A Subject carries a `group` field, but in a crossover it cannot be
        # the answer: the same animal is in both arms.
        project = Project(root=tmp_path)
        subject = Subject(subject_id="M001", group="saline")
        project.set_session("d1", subject=subject)
        project.set_session("d2", subject=subject, group="TRH")
        assert project.group_for("d1") == "saline"
        assert project.group_for("d2") == "TRH"

    def test_groups_lists_both_arms(self, cohort):
        assert cohort.groups() == {
            "TRH": ["M001 d2", "M002 d1"],
            "saline": ["M001 d1", "M002 d2"],
        }

    def test_per_session_treatment_survives_the_round_trip(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("d2", group="TRH", treatment={"solution": "TRH", "dose": "1 mg/kg"})
        project.save()
        assert Project.load(tmp_path).record("d2").treatment == {
            "solution": "TRH",
            "dose": "1 mg/kg",
        }

    def test_a_subjects_drug_fields_land_on_the_session(self, tmp_path):
        # Treatment named on the Subject describes that day, not the animal,
        # so it must not end up in the shared subject record.
        project = Project(root=tmp_path)
        project.set_session("d1", subject=Subject(subject_id="M001", solution="saline"))
        project.save()
        again = Project.load(tmp_path)
        assert again.record("d1").treatment == {"solution": "saline"}
        assert again.subjects["M001"].solution == ""


class TestNoManifest:
    """Every existing cohort is in this state and must still load."""

    def test_a_folder_with_no_manifest_is_an_anonymous_project(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        project = Project.load(tmp_path)
        assert project.session_ids() == ["Test 1"]
        assert project.subject_for("Test 1") is None
        assert project.group_for("Test 1") is None

    def test_an_unlisted_session_still_resolves_its_files(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        assert Project.load(tmp_path).session("Test 1").video is not None

    def test_a_corrupt_manifest_is_refused_not_ignored(self, tmp_path):
        (tmp_path / MANIFEST_NAME).write_text("{ not json")
        with pytest.raises(ProjectError):
            Project.load(tmp_path)

    def test_a_newer_schema_is_refused(self, tmp_path):
        # Reading it would silently drop whatever the newer version added,
        # and the operator would never know which fields went missing.
        (tmp_path / MANIFEST_NAME).write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(ProjectError, match="newer"):
            Project.load(tmp_path)


class TestDiscovery:
    def test_it_finds_sessions_in_the_canonical_layout(self, tmp_path):
        (tmp_path / "sessions" / "Test 1").mkdir(parents=True)
        (tmp_path / "sessions" / "Test 2").mkdir()
        assert Project.load(tmp_path).session_ids() == ["Test 1", "Test 2"]

    def test_it_finds_sessions_in_a_media_subfolder(self, tmp_path):
        (tmp_path / "males").mkdir()
        (tmp_path / "males" / "Test 17.mp4").touch()
        assert Project.load(tmp_path).session_ids() == ["Test 17"]

    def test_output_folders_are_not_scanned_for_recordings(self, tmp_path):
        # annotated.mp4 in analysis/ is a rendering of a session, not one.
        (tmp_path / "analysis").mkdir()
        (tmp_path / "analysis" / "annotated.mp4").touch()
        assert Project.load(tmp_path).session_ids() == []

    def test_a_loose_sidecar_does_not_invent_a_session(self, tmp_path):
        # Discovery is by recording. A stray zone file with no video would
        # otherwise put an empty row in every cohort table.
        (tmp_path / "Test 3_zone.json").write_text("{}")
        assert Project.load(tmp_path).session_ids() == []

    def test_a_manifest_session_with_no_files_is_still_listed(self, tmp_path):
        # Precisely what doctor needs to report, so it must not be hidden.
        project = Project(root=tmp_path)
        project.set_session("Test 9", group="saline")
        project.save()
        assert Project.load(tmp_path).session_ids() == ["Test 9"]


class TestProvenance:
    def test_it_round_trips(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session(
            "Test 1",
            provenance=Provenance(
                pose_model="train-6", classifier="postural-4class", thresholds={"freeze_cm_s": 3.0}
            ),
        )
        project.save()
        recovered = Project.load(tmp_path).record("Test 1").provenance
        assert recovered.pose_model == "train-6"
        assert recovered.thresholds == {"freeze_cm_s": 3.0}

    def test_an_empty_provenance_is_falsey_and_not_written(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("Test 1", group="saline")
        project.save()
        assert (
            "provenance"
            not in json.loads((tmp_path / MANIFEST_NAME).read_text())["sessions"]["Test 1"]
        )


class TestPartialUpdates:
    def test_setting_one_field_leaves_the_others_alone(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("Test 1", subject="M001", group="saline", day="1")
        project.set_session("Test 1", provenance=Provenance(pose_model="train-6"))
        record = project.record("Test 1")
        assert (record.subject, record.group, record.day) == ("M001", "saline", "1")
        assert record.provenance.pose_model == "train-6"

    def test_a_group_can_be_cleared_explicitly(self, tmp_path):
        # None means "leave it"; "" means "it is not assigned". Without the
        # distinction a wrong assignment could never be taken back.
        project = Project(root=tmp_path)
        project.set_session("Test 1", group="saline")
        project.set_session("Test 1", group="")
        assert project.group_for("Test 1") is None


class TestSessionIntegration:
    def test_session_subject_resolves_through_the_manifest(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("Test 1", subject=Subject(subject_id="M001", sex="Male"), group="TRH")
        project.save()
        session = project.session("Test 1")
        assert session.subject.subject_id == "M001"
        assert session.group == "TRH"

    def test_session_subject_is_none_without_a_manifest(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        session = Project.load(tmp_path).session("Test 1")
        assert session.subject is None
        assert session.group is None
