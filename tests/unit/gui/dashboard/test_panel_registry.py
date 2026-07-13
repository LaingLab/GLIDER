from glider.gui.dashboard.panel_registry import PANEL_KEYS, PANEL_NAMES


def test_five_panels_registered():
    assert PANEL_KEYS == (
        "run_control",
        "device_states",
        "camera",
        "manual_controls",
        "experiment_info",
    )


def test_every_key_has_a_display_name():
    for key in PANEL_KEYS:
        assert PANEL_NAMES[key]
        assert isinstance(PANEL_NAMES[key], str)


def test_names_have_no_extra_keys():
    assert set(PANEL_NAMES) == set(PANEL_KEYS)
