from glider.gui.panels.node_editor_controller import merge_behavior_setting


def test_merge_behavior_setting_creates_nested_namespace():
    state = {}
    out = merge_behavior_setting(state, "revolution", "turns", 3)
    assert out == {"behavior_settings": {"revolution": {"turns": 3}}}


def test_merge_behavior_setting_preserves_siblings():
    state = {"behavior_settings": {"revolution": {"turns": 3}}, "timeout": 5}
    out = merge_behavior_setting(state, "revolution", "ramp_device", "m")
    assert out["behavior_settings"]["revolution"] == {"turns": 3, "ramp_device": "m"}
    assert out["timeout"] == 5


def test_merge_behavior_setting_isolates_behaviors():
    state = {"behavior_settings": {"revolution": {"turns": 3}}}
    out = merge_behavior_setting(state, "counts", "counts_target", 800)
    assert out["behavior_settings"]["revolution"] == {"turns": 3}
    assert out["behavior_settings"]["counts"] == {"counts_target": 800}
