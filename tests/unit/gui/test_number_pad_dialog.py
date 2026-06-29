"""Tests for the touch number-pad dialog's value logic."""

from __future__ import annotations

from glider.gui.dialogs.number_pad_dialog import NumberPadDialog


def test_digits_build_value(qtbot):
    pad = NumberPadDialog("Revolutions", value=1, minimum=1, maximum=1000)
    qtbot.addWidget(pad)
    pad._on_key("C")
    pad._on_key("2")
    pad._on_key("5")
    assert pad.value() == 25


def test_value_is_clamped_to_max(qtbot):
    pad = NumberPadDialog("Revolutions", value=1, minimum=1, maximum=100)
    qtbot.addWidget(pad)
    for k in "9999":
        pad._on_key(k)
    assert pad.value() == 100


def test_backspace_and_clear(qtbot):
    pad = NumberPadDialog("N", value=0, minimum=0, maximum=1000)
    qtbot.addWidget(pad)
    pad._on_key("1")
    pad._on_key("2")
    pad._on_key("3")
    pad._on_key("⌫")
    assert pad.value() == 12
    pad._on_key("C")
    assert pad.value() == 0
