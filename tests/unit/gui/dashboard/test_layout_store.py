import json

from glider.gui.dashboard.layout import default_layout
from glider.gui.dashboard.layout_store import from_dict, load_layout, save_layout, to_dict


def test_roundtrip_through_dict():
    layout = default_layout()
    restored = from_dict(to_dict(layout))
    assert restored.assignment == layout.assignment


def test_load_missing_file_returns_default(tmp_path):
    path = tmp_path / "dashboard_layout.json"
    assert load_layout(path).assignment == default_layout().assignment


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "dashboard_layout.json"
    layout = default_layout().with_assignment(
        {
            "top_left": "camera",
            "top_right": "run_control",
            "bottom_left": "device_states",
            "bottom_right": "manual_controls",
        }
    )
    save_layout(layout, path)
    loaded = load_layout(path)
    assert loaded.assignment == layout.assignment


def test_load_corrupt_json_returns_default(tmp_path):
    path = tmp_path / "dashboard_layout.json"
    path.write_text("{not valid json")
    assert load_layout(path).assignment == default_layout().assignment


def test_load_assignment_with_unknown_key_returns_default(tmp_path):
    path = tmp_path / "dashboard_layout.json"
    path.write_text(
        json.dumps(
            {
                "assignment": {
                    "top_left": "bogus",
                    "top_right": "device_states",
                    "bottom_left": "camera",
                    "bottom_right": "experiment_info",
                }
            }
        )
    )
    assert load_layout(path).assignment == default_layout().assignment


def test_load_assignment_with_duplicate_panels_returns_default(tmp_path):
    path = tmp_path / "dashboard_layout.json"
    path.write_text(
        json.dumps(
            {
                "assignment": {
                    "top_left": "camera",
                    "top_right": "camera",
                    "bottom_left": "run_control",
                    "bottom_right": "device_states",
                }
            }
        )
    )
    assert load_layout(path).assignment == default_layout().assignment


def test_load_non_string_assignment_values_returns_default(tmp_path):
    path = tmp_path / "dashboard_layout.json"
    path.write_text(
        json.dumps(
            {
                "assignment": {
                    "top_left": ["x"],
                    "top_right": 1,
                    "bottom_left": None,
                    "bottom_right": {},
                }
            }
        )
    )
    assert load_layout(path).assignment == default_layout().assignment


def test_save_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "dashboard_layout.json"
    save_layout(default_layout(), path)
    assert path.exists()
