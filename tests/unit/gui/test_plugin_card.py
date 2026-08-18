"""What a plugin row shows, per state.

These assert on *which controls exist*, not on pixels: the spec fixes the
control set per state, and that is the part a user depends on.
"""

import pytest

from glider.gui.widgets.plugin_card import PluginCard

ENTRY = {
    "name": "glider-harp",
    "display_name": "Harp Devices",
    "version": "0.1.0",
    "pypi": "glider-harp",
    "description": "Harp-protocol instruments.",
    "author": "Laing Lab",
    "provides": ["driver", "device"],
}


@pytest.mark.parametrize(
    "state,expected",
    [
        ("enabled", ["Disable", "Reload"]),
        ("disabled", ["Enable"]),
        ("available", ["Install"]),
        ("incompatible", ["Install"]),
        ("failed", ["Retry"]),
    ],
)
def test_each_state_offers_its_controls(qtbot, state, expected):
    card = PluginCard(ENTRY, state=state)
    qtbot.addWidget(card)

    assert [b.text() for b in card.buttons()] == expected


def test_an_incompatible_plugin_cannot_be_installed(qtbot):
    card = PluginCard(ENTRY, state="incompatible", message="Needs GLIDER >=2.0. Running 1.0.0.")
    qtbot.addWidget(card)

    assert card.buttons()[0].isEnabled() is False
    assert "2.0" in card.message_text()


def test_the_package_name_and_version_are_shown_verbatim(qtbot):
    """These are what you type into pip and what a bug report needs."""
    card = PluginCard(ENTRY, state="available")
    qtbot.addWidget(card)

    assert "glider-harp" in card.identity_text()
    assert "0.1.0" in card.identity_text()


def test_pip_output_is_shown_when_an_install_fails(qtbot):
    card = PluginCard(
        ENTRY,
        state="failed",
        message="pip exited with code 1.",
        output="ERROR: Could not find a version that satisfies the requirement zmq>=26",
    )
    qtbot.addWidget(card)

    assert "zmq>=26" in card.output_text()


def test_clicking_install_emits_the_plugin_name(qtbot):
    card = PluginCard(ENTRY, state="available")
    qtbot.addWidget(card)

    with qtbot.waitSignal(card.install_requested) as blocker:
        card.buttons()[0].click()

    assert blocker.args == ["glider-harp"]
