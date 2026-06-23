"""
Tests for WaitForInputNode's "revolution" mode.

Revolution mode counts full turns of a wrap-around sensor such as the
AS5600 magnetic encoder, whose 12-bit raw angle sawtooths 0..4095 and
back. A poll-to-poll jump larger than half the full-scale range can only
be the 4095<->0 boundary being crossed, i.e. one completed turn. Either
direction counts, and the node fires after ``turns_target`` of them.
"""

from __future__ import annotations

import pytest

from glider.nodes.control_nodes import WaitForInputNode


class _ScriptedDevice:
    """Async-readable device that yields a fixed sequence of values.

    Once the sequence is exhausted it keeps returning the final value, so
    a poll loop that never triggers stalls on a steady reading (and is
    caught by the test's timeout) rather than erroring out.
    """

    def __init__(self, values: list[int]):
        self._values = list(values)
        self._i = 0
        self.id = "scripted"

    async def read(self) -> int:
        value = self._values[self._i]
        if self._i < len(self._values) - 1:
            self._i += 1
        return value


def _make_node(values: list[int], turns: int = 1, counts: int = 4096) -> WaitForInputNode:
    node = WaitForInputNode()
    node._threshold_mode = "revolution"
    node._turns_target = turns
    node._counts_per_turn = counts
    node._poll_interval = 0.0  # don't slow the test down
    node._device = _ScriptedDevice(values)
    node._waiting = True
    return node


# A forward sweep (0..4000), a wrap down to ~100 (= turn 1), another
# sweep, then a wrap down to ~200 (= turn 2).
_TWO_TURN_SWEEP = [0, 1000, 2000, 3000, 4000, 100, 1100, 2100, 3100, 4090, 200]


async def test_fires_after_one_completed_turn():
    node = _make_node(_TWO_TURN_SWEEP, turns=1)
    await node._poll_device(timeout=0.0)
    # First wrap is 4000 -> 100, so the trigger value is the post-wrap reading.
    assert node._trigger_value == 100


async def test_waits_for_target_turn_count():
    node = _make_node(_TWO_TURN_SWEEP, turns=2)
    await node._poll_device(timeout=0.0)
    # Should not fire on the first wrap; the second wrap is 4090 -> 200.
    assert node._trigger_value == 200


async def test_counts_either_direction():
    # Decreasing sweep then a jump UP across 0->4095 is also one turn.
    node = _make_node([4000, 3000, 2000, 1000, 4090], turns=1)
    await node._poll_device(timeout=0.0)
    assert node._trigger_value == 4090


async def test_no_phantom_trigger_on_first_read_or_small_steps():
    # Starts high and only ever takes small steps — never wraps, so the
    # node must keep waiting and ultimately time out (no phantom turn from
    # the very first reading having no predecessor).
    node = _make_node([4095, 4090, 4080, 4070], turns=1)
    with pytest.raises(TimeoutError):
        await node._poll_device(timeout=0.2)
    assert node._trigger_value is None


def test_revolution_state_round_trips():
    node = WaitForInputNode()
    node.set_state(
        {
            "threshold_mode": "revolution",
            "turns_target": 5,
            "counts_per_turn": 1024,
        }
    )
    assert node._threshold_mode == "revolution"
    assert node._turns_target == 5
    assert node._counts_per_turn == 1024

    state = node.get_state()
    assert state["threshold_mode"] == "revolution"
    assert state["turns_target"] == 5
    assert state["counts_per_turn"] == 1024
