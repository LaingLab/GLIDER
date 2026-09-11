"""The four-row DeepLabCut header. This is the interop artifact -- SimBA, DLC
and Keypoint-MoSeq read it directly -- so its shape is not ours to improvise."""

import json

import numpy as np

from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import meta_path, to_dlc_csv_multi
from glider.vision.pose.tracks import PoseTracks

NAMES = ["snout", "tail"]


def pose(fill):
    return PoseData(
        xy=np.full((3, 2, 2), float(fill)),
        confidence=np.full((3, 2), 0.9),
        keypoint_names=list(NAMES),
        fps=30.0,
        source="yolo_test",
        metadata={"resolution": (640, 480)},
    )


def two_animals():
    return PoseTracks(tracks={0: pose(1), 1: pose(2)}, fps=30.0)


def test_the_header_is_four_rows_named_correctly(tmp_path):
    path = to_dlc_csv_multi(two_animals(), tmp_path / "out.csv")
    head = path.read_text().splitlines()[:4]
    assert head[0].startswith("scorer")
    assert head[1].startswith("individuals")
    assert head[2].startswith("bodyparts")
    assert head[3].startswith("coords")


def test_both_animals_appear_under_their_slot_names(tmp_path):
    path = to_dlc_csv_multi(two_animals(), tmp_path / "out.csv")
    individuals = path.read_text().splitlines()[1]
    assert "animal0" in individuals and "animal1" in individuals


def test_columns_are_x_y_likelihood_per_bodypart_per_animal(tmp_path):
    import pandas as pd

    path = to_dlc_csv_multi(two_animals(), tmp_path / "out.csv")
    df = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
    assert df.shape == (3, 2 * 2 * 3)  # animals x bodyparts x (x, y, likelihood)
    assert np.allclose(df[("yolo_test", "animal1", "snout", "x")], 2.0)


def test_the_sidecar_records_the_animal_count_and_names(tmp_path):
    path = to_dlc_csv_multi(two_animals(), tmp_path / "out.csv")
    meta = json.loads(meta_path(path).read_text())
    assert meta["n_animals"] == 2
    assert meta["individuals"] == ["animal0", "animal1"]
    assert meta["fps"] == 30.0
    assert meta["keypoint_names"] == NAMES


def test_one_animal_still_writes_a_four_row_header(tmp_path):
    # A PoseTracks of one is still multi-animal output. The single-animal
    # three-row path is to_dlc_csv, and which one ran must stay legible.
    single = PoseTracks(tracks={0: pose(1)}, fps=30.0)
    path = to_dlc_csv_multi(single, tmp_path / "out.csv")
    assert path.read_text().splitlines()[1].startswith("individuals")
