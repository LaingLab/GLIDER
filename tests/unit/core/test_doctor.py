"""Doctor: catching, mechanically, the failures that cost a cohort its scores.

The headline case is the pose sidecar. It sat in a different folder from its
CSV; resolution came back None; classify computed no speed axis; freezing and
darting were never scored for eleven recordings and nothing raised. If doctor
does not catch that, it is not worth having.

Every check here is a warning. A cohort mid-analysis is legitimately
inconsistent, and a checker that refuses to run on real folders is one nobody
runs.
"""

from __future__ import annotations

import json

import pytest

from glider.core.doctor import Finding, check_session, doctor, format_report
from glider.core.experiment_session import Subject
from glider.core.project import Project, Provenance


def _pose_csv(path, *, sidecar: bool = True, resolution=(640, 480)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ["nose", "body_center"]
    path.write_text(
        "scorer," + ",".join(["yolo"] * len(parts) * 3) + "\n"
        "bodyparts," + ",".join(p for p in parts for _ in range(3)) + "\n"
        "coords," + ",".join(["x", "y", "likelihood"] * len(parts)) + "\n"
        "0," + ",".join(["1.0"] * len(parts) * 3) + "\n"
    )
    if not sidecar:
        return
    payload = {"schema_version": 1, "fps": 30.0, "source": "yolo", "n_frames": 1}
    if resolution is not None:
        payload["resolution"] = list(resolution)
    (path.parent / (path.stem + ".meta.json")).write_text(json.dumps(payload))


def _calibration(corners=None) -> dict:
    """A square arena filling most of a 640x480 frame - 30 cm across."""
    return {
        "corners": corners or [[100, 60], [540, 60], [540, 420], [100, 420]],
        "width_cm": 30.0,
        "height_cm": 30.0,
        "frame_size": [640, 480],
    }


def _arena(path, *, corners=None) -> None:
    """A bare calibration, the simplest shape an arena file takes."""
    path.write_text(json.dumps(_calibration(corners)))


def _arena_document(path, *, corners=None, **extra) -> None:
    """What tools/arena_zones.py actually writes: the calibration wrapped in a
    document, with its derived numbers recorded alongside."""
    from glider.vision.arena import ArenaCalibration

    calibration = ArenaCalibration.from_dict(_calibration(corners))
    payload = {
        "schema_version": 1,
        "session": path.stem.replace("_arena", ""),
        "zone_cm": 10.0,
        "arena": _calibration(corners),
        "px_per_cm_centre": calibration.px_per_cm_centre,
        "residuals": calibration.residuals(),
    }
    payload.update(extra)
    path.write_text(json.dumps(payload))


def _checks(findings: list[Finding]) -> set[str]:
    return {f.check for f in findings}


@pytest.fixture
def healthy(tmp_path):
    """A session with nothing wrong with it."""
    (tmp_path / "Test 1.mp4").touch()
    _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
    _arena(tmp_path / "Test 1_arena.json")
    project = Project(root=tmp_path)
    project.set_session("Test 1", subject=Subject(subject_id="M001"), group="saline")
    project.save()
    return project


class TestTheSidecarBug:
    """The one that erased freezing from eleven recordings."""

    def test_a_pose_csv_with_no_sidecar_is_reported(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv", sidecar=False)
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        assert "pose_sidecar_missing" in _checks(findings)

    def test_the_finding_says_what_it_costs(self, tmp_path):
        # "No sidecar" is a fact about the disk. The reason to act on it is
        # that freezing and darting silently stop being scored.
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv", sidecar=False)
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        message = next(f.message for f in findings if f.check == "pose_sidecar_missing")
        assert "freezing" in message and "darting" in message

    def test_a_sidecar_with_no_resolution_is_reported_too(self, tmp_path):
        # Present but useless: this is what the parked-metadata folder actually
        # produced for the readers that did find a sidecar.
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv", resolution=None)
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        assert "no_resolution" in _checks(findings)

    def test_a_healthy_session_reports_neither(self, healthy):
        findings = check_session(healthy.session("Test 1"), healthy)
        assert not _checks(findings) & {"pose_sidecar_missing", "no_resolution"}


class TestHealthy:
    def test_a_complete_session_has_nothing_to_report(self, healthy):
        assert check_session(healthy.session("Test 1"), healthy) == []

    def test_the_report_says_so(self, healthy):
        assert "nothing to report" in format_report(doctor(healthy), project=healthy)


class TestSubjectsAndGroups:
    def test_a_session_with_no_subject_is_reported(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        _arena(tmp_path / "Test 1_arena.json")
        project = Project.load(tmp_path)
        assert "no_subject" in _checks(check_session(project.session("Test 1"), project))

    def test_a_subject_with_no_group_is_reported(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        _arena(tmp_path / "Test 1_arena.json")
        project = Project(root=tmp_path)
        project.set_session("Test 1", subject=Subject(subject_id="M001"))
        assert "no_group" in _checks(check_session(project.session("Test 1"), project))

    def test_a_dangling_subject_reference_is_reported(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("Test 1", subject="M404", group="saline")
        assert "unknown_subject" in _checks(check_session(project.session("Test 1"), project))

    def test_subject_checks_are_skipped_without_a_manifest(self, tmp_path):
        # A folder that was never adopted has no subjects by construction, and
        # saying so once per session is noise, not a finding.
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        _arena(tmp_path / "Test 1_arena.json")
        session = Project.load(tmp_path).session("Test 1")
        assert check_session(session) == []


class TestStaleAnalysis:
    def test_outputs_from_a_superseded_model_are_reported(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        _arena(tmp_path / "Test 1_arena.json")
        analysis = tmp_path / "Test 1" / "analysis"
        analysis.mkdir(parents=True)
        (analysis / "stats.csv").write_text("state,fraction\n")
        (analysis / "run.json").write_text(
            json.dumps({"pose_csv": str(tmp_path / "Test 1DLC_train-2-4.csv")})
        )
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        assert "stale_analysis" in _checks(findings)

    def test_outputs_from_the_current_model_are_not(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        _arena(tmp_path / "Test 1_arena.json")
        analysis = tmp_path / "Test 1" / "analysis"
        analysis.mkdir(parents=True)
        (analysis / "run.json").write_text(
            json.dumps({"pose_csv": str(tmp_path / "Test 1DLC_train-6.csv")})
        )
        assert "stale_analysis" not in _checks(
            check_session(Project.load(tmp_path).session("Test 1"))
        )

    def test_an_unreadable_run_manifest_is_not_a_crash(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        analysis = tmp_path / "Test 1" / "analysis"
        analysis.mkdir(parents=True)
        (analysis / "run.json").write_text("{ not json")
        check_session(Project.load(tmp_path).session("Test 1"))  # must not raise


class TestArena:
    def test_a_missing_arena_is_reported(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        assert "no_arena" in _checks(check_session(Project.load(tmp_path).session("Test 1")))

    def test_a_cohort_calibration_master_downgrades_the_warning(self, tmp_path):
        # One real cohort calibrates from a single master and draws no
        # per-session arenas. Warning once per session buries the real findings.
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        (tmp_path / "pose_calibration.json").write_text("{}")
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        assert [f.severity for f in findings if f.check == "no_arena"] == ["info"]

    def test_it_is_still_a_warning_with_no_calibration_at_all(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        assert [f.severity for f in findings if f.check == "no_arena"] == ["warning"]

    def test_a_badly_drawn_arena_is_reported(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        # A wildly non-square quad: one corner clicked on the wrong feature.
        _arena(
            tmp_path / "Test 1_arena.json",
            corners=[[100, 60], [540, 60], [620, 460], [110, 300]],
        )
        assert "suspect_arena" in _checks(check_session(Project.load(tmp_path).session("Test 1")))

    def test_a_corrupt_arena_file_is_reported_not_raised(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        (tmp_path / "Test 1_arena.json").write_text("{ not json")
        assert "suspect_arena" in _checks(check_session(Project.load(tmp_path).session("Test 1")))

    def test_the_wrapped_document_shape_is_understood(self, tmp_path):
        # What arena_zones.py writes: the calibration nested under "arena".
        # Reading only the inner shape reported all thirty correctly-drawn
        # arenas in a real cohort as unreadable.
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        _arena_document(tmp_path / "Test 1_arena.json")
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        assert "suspect_arena" not in _checks(findings)
        assert "no_arena" not in _checks(findings)

    def test_a_wrapped_document_that_is_off_is_still_reported(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        _arena_document(
            tmp_path / "Test 1_arena.json",
            corners=[[100, 60], [540, 60], [620, 460], [110, 300]],
        )
        assert "suspect_arena" in _checks(check_session(Project.load(tmp_path).session("Test 1")))

    def test_the_recorded_scale_is_used_when_present(self, tmp_path):
        from glider.core.doctor import _arena_scale

        path = tmp_path / "Test 1_arena.json"
        _arena_document(path)
        expected = json.loads(path.read_text())["px_per_cm_centre"]
        assert _arena_scale(path) == pytest.approx(expected)


class TestScaleOutlier:
    def _cohort(self, tmp_path, scales_px_per_cm):
        """Sessions whose arenas are drawn at the given pixel scales."""
        project = Project(root=tmp_path)
        for i, px_per_cm in enumerate(scales_px_per_cm, start=1):
            sid = f"Test {i}"
            (tmp_path / f"{sid}.mp4").touch()
            _pose_csv(tmp_path / f"{sid}DLC_train-6.csv")
            side = 30.0 * px_per_cm
            _arena(
                tmp_path / f"{sid}_arena.json",
                corners=[[50, 20], [50 + side, 20], [50 + side, 20 + side], [50, 20 + side]],
            )
            project.set_session(sid, subject=Subject(subject_id=f"M{i:03d}"), group="saline")
        project.save()
        return Project.load(tmp_path)

    def test_one_mis_drawn_arena_stands_out(self, tmp_path):
        project = self._cohort(tmp_path, [14.0, 14.2, 13.9, 14.1, 7.0])
        outliers = [f.session_id for f in doctor(project) if f.check == "scale_outlier"]
        assert outliers == ["Test 5"]

    def test_a_real_cohorts_camera_spread_does_not_fire(self, tmp_path):
        # One real cohort spans 17% across its cameras and every arena in it is
        # correctly drawn. A check that flags all thirty teaches the operator
        # to ignore it.
        project = self._cohort(tmp_path, [13.0, 13.6, 14.2, 14.8, 15.2])
        assert [f for f in doctor(project) if f.check == "scale_outlier"] == []

    def test_too_few_sessions_to_have_a_median(self, tmp_path):
        project = self._cohort(tmp_path, [14.0, 7.0])
        assert [f for f in doctor(project) if f.check == "scale_outlier"] == []


class TestNamingDrift:
    def test_an_artifact_under_another_spelling_is_noted(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        _arena(tmp_path / "Test 1_arena.json")
        (tmp_path / "test1_zone.json").write_text("{}")
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        drift = [f for f in findings if f.check == "naming_drift"]
        assert len(drift) == 1
        assert drift[0].severity == "info"

    def test_matching_names_are_not_noted(self, healthy):
        (healthy.root / "Test 1_zone.json").write_text("{}")
        findings = check_session(healthy.session("Test 1"), healthy)
        assert "naming_drift" not in _checks(findings)


class TestSeverity:
    def test_an_untracked_session_is_a_note_not_a_warning(self, tmp_path):
        # Nothing is wrong with a recording that has not been tracked yet.
        (tmp_path / "Test 1.mp4").touch()
        _arena(tmp_path / "Test 1_arena.json")
        findings = check_session(Project.load(tmp_path).session("Test 1"))
        assert [f.severity for f in findings if f.check == "no_pose"] == ["info"]

    def test_nothing_doctor_reports_is_an_error(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("Test 9", subject="M404")
        project.save()
        assert {f.severity for f in doctor(project)} <= {"warning", "info"}


class TestReport:
    def test_it_groups_by_session_and_counts(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("Test 9", group="saline")
        project.save()
        report = format_report(doctor(project), project=project)
        assert "Test 9" in report
        assert "warning" in report

    def test_warnings_come_before_notes_within_a_session(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        _pose_csv(tmp_path / "Test 1DLC_train-6.csv")
        (tmp_path / "test1_zone.json").write_text("{}")
        report = format_report(check_session(Project.load(tmp_path).session("Test 1")))
        assert report.index("no_arena") < report.index("naming_drift")


class TestCompetingAnalyses:
    """Two full sets of outputs for one recording, disagreeing with each other.

    Found in a real cohort: females/Test 1/ and Test 1/ were both produced from
    males/Test 1.mp4 and reported different bout counts. Whichever a tool found
    first became the answer.
    """

    def _two_runs(self, tmp_path, video="males/Test 1.mp4"):
        (tmp_path / "males").mkdir(exist_ok=True)
        (tmp_path / "males" / "Test 1.mp4").touch()
        for folder in ("Test 1", "females/Test 1"):
            out = tmp_path / folder
            out.mkdir(parents=True, exist_ok=True)
            (out / "run.json").write_text(json.dumps({"video": str(tmp_path / video)}))
            (out / "stats.csv").write_text("state,fraction\n")
        return Project.load(tmp_path)

    def test_two_sets_of_outputs_for_one_recording_are_reported(self, tmp_path):
        findings = doctor(self._two_runs(tmp_path))
        assert "competing_analyses" in _checks(findings)

    def test_the_finding_names_both_folders(self, tmp_path):
        findings = doctor(self._two_runs(tmp_path))
        message = next(f.message for f in findings if f.check == "competing_analyses")
        assert "Test 1" in message and "females" in message

    def test_a_set_kept_in_its_own_batch_folder_is_only_a_note(self, tmp_path):
        # rescored_filtered/<id>/ is a deliberate alternate run. GLIDER resolves
        # only one of them, so nothing is ambiguous today - and a warning per
        # session is how a report gets scrolled past.
        (tmp_path / "Test 1.mp4").touch()
        for folder in ("Test 1", "rescored_filtered/Test 1"):
            out = tmp_path / folder
            out.mkdir(parents=True, exist_ok=True)
            (out / "run.json").write_text(json.dumps({"video": str(tmp_path / "Test 1.mp4")}))
        finding = next(f for f in doctor(Project.load(tmp_path)) if f.check == "competing_analyses")
        assert finding.severity == "info"

    def test_two_sets_glider_would_both_resolve_is_a_warning(self, tmp_path):
        # analysis/ and final_outputs/ are both places Session looks, so which
        # one answers is down to ordering. That bites today.
        (tmp_path / "Test 1.mp4").touch()
        session_dir = tmp_path / "Test 1"
        (session_dir / "Test 1.mp4").parent.mkdir(parents=True, exist_ok=True)
        (session_dir / "Test 1.mp4").touch()
        for folder in ("analysis", "final_outputs"):
            out = session_dir / folder
            out.mkdir(parents=True, exist_ok=True)
            (out / "run.json").write_text(json.dumps({"video": str(tmp_path / "Test 1.mp4")}))
        finding = next(f for f in doctor(Project.load(tmp_path)) if f.check == "competing_analyses")
        assert finding.severity == "warning"

    def test_the_session_id_keeps_its_case(self, tmp_path):
        # Lower-casing it splits one session into two entries in the report.
        findings = doctor(self._two_runs(tmp_path))
        assert any(f.session_id == "Test 1" for f in findings if f.check == "competing_analyses")

    def test_one_set_of_outputs_is_not_reported(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        out = tmp_path / "Test 1"
        out.mkdir()
        (out / "run.json").write_text(json.dumps({"video": str(tmp_path / "Test 1.mp4")}))
        assert "competing_analyses" not in _checks(doctor(Project.load(tmp_path)))

    def test_runs_for_different_recordings_are_not_reported(self, tmp_path):
        for sid in ("Test 1", "Test 2"):
            (tmp_path / f"{sid}.mp4").touch()
            out = tmp_path / sid
            out.mkdir()
            (out / "run.json").write_text(json.dumps({"video": str(tmp_path / f"{sid}.mp4")}))
        assert "competing_analyses" not in _checks(doctor(Project.load(tmp_path)))

    def test_a_manifest_naming_no_video_is_skipped(self, tmp_path):
        (tmp_path / "Test 1.mp4").touch()
        out = tmp_path / "Test 1"
        out.mkdir()
        (out / "run.json").write_text(json.dumps({"speed_only": True}))
        doctor(Project.load(tmp_path))  # must not raise


class TestUnlistedSessions:
    def test_a_manifest_session_with_no_files_is_reported(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("Test 9", subject=Subject(subject_id="M001"), group="saline")
        project.save()
        assert "no_video" in _checks(doctor(Project.load(tmp_path)))

    def test_provenance_alone_does_not_satisfy_the_subject_check(self, tmp_path):
        project = Project(root=tmp_path)
        project.set_session("Test 9", provenance=Provenance(pose_model="train-6"))
        assert "no_subject" in _checks(check_session(project.session("Test 9"), project))
