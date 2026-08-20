"""Classifying a dropped path by kind.

``classify_drop`` is pure and Qt-free on purpose: routing a drop is a decision
about a path, and keeping it out of the widget makes it exhaustively testable
without a running application.
"""

from __future__ import annotations

import json

import pytest

from glider.gui.panels.model_drop import DropKind, classify_drop


def test_pt_is_a_pose_model(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    assert classify_drop(p) is DropKind.POSE_MODEL


def test_onnx_is_a_pose_model(tmp_path):
    p = tmp_path / "model.onnx"
    p.write_bytes(b"x")
    assert classify_drop(p) is DropKind.POSE_MODEL


def test_pkl_is_a_behavior_model(tmp_path):
    p = tmp_path / "model.pkl"
    p.write_bytes(b"x")
    assert classify_drop(p) is DropKind.BEHAVIOR_MODEL


@pytest.mark.parametrize("ext", [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".webm"])
def test_video_extensions(tmp_path, ext):
    p = tmp_path / f"clip{ext}"
    p.write_bytes(b"x")
    assert classify_drop(p) is DropKind.VIDEO


def test_sidecar_folder_is_a_pose_model(tmp_path):
    (tmp_path / "glider_pose.json").write_text(json.dumps({"kind": "dlc"}))
    assert classify_drop(tmp_path) is DropKind.POSE_MODEL


def test_dlc_project_folder_is_a_pose_model(tmp_path):
    (tmp_path / "pose_cfg.yaml").write_text("all_joints_names: [a]\n")
    assert classify_drop(tmp_path) is DropKind.POSE_MODEL


def test_sleap_folder_is_a_pose_model(tmp_path):
    (tmp_path / "training_config.json").write_text("{}")
    assert classify_drop(tmp_path) is DropKind.POSE_MODEL


def test_folder_holding_only_an_onnx_is_still_a_pose_model(tmp_path):
    """Routed as a pose model so the drop handler can explain what's missing.

    Calling it UNKNOWN would silently ignore the drop; the operator needs to be
    told the folder has no sidecar, not left wondering why nothing happened.
    """
    (tmp_path / "model.onnx").write_bytes(b"x")
    assert classify_drop(tmp_path) is DropKind.POSE_MODEL


def test_unrelated_file_is_unknown(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hi")
    assert classify_drop(p) is DropKind.UNKNOWN


def test_empty_folder_is_unknown(tmp_path):
    assert classify_drop(tmp_path) is DropKind.UNKNOWN


def test_missing_path_is_unknown(tmp_path):
    assert classify_drop(tmp_path / "nope") is DropKind.UNKNOWN


def test_video_extensions_match_the_batch_module(tmp_path):
    """The panel and the batch picker must not drift apart on what a video is."""
    from glider.vision.pose.batch import VIDEO_EXTS

    for ext in VIDEO_EXTS:
        p = tmp_path / f"clip{ext}"
        p.write_bytes(b"x")
        assert classify_drop(p) is DropKind.VIDEO, ext
