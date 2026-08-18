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
        ("broken", ["Reload"]),
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


def test_a_plugin_on_disk_that_did_not_load_is_not_called_an_install_failure(qtbot):
    """It is installed, so it belongs under the Installed filter -- and the
    install did not fail, the import did. The pill has to say which."""
    card = PluginCard(ENTRY, state="broken", message="Module not found: serial")
    qtbot.addWidget(card)

    assert "install failed" not in card._pill.text().lower()
    assert "serial" in card.message_text()


def test_the_version_can_be_corrected_after_an_install(qtbot):
    """pip resolves the version; the catalogue only advertises one. The label was
    written once in `__init__` and never again."""
    card = PluginCard(ENTRY, state="available")
    qtbot.addWidget(card)

    card.set_version("0.2.5")

    assert "0.2.5" in card.identity_text()
    assert "0.1.0" not in card.identity_text()


def test_disable_says_what_it_actually_does(qtbot):
    """Disable marks the plugin so it is not loaded again. It does not unregister
    what is already registered, and it does not survive a restart -- so the
    control must not imply otherwise."""
    card = PluginCard(ENTRY, state="enabled")
    qtbot.addWidget(card)

    tip = card.buttons()[0].toolTip().lower()

    assert "restart" in tip
    assert "next" in tip or "future" in tip


def test_clicking_install_emits_the_plugin_name(qtbot):
    card = PluginCard(ENTRY, state="available")
    qtbot.addWidget(card)

    with qtbot.waitSignal(card.install_requested) as blocker:
        card.buttons()[0].click()

    assert blocker.args == ["glider-harp"]
