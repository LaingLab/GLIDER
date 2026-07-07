# tests/unit/hal/test_input_behavior.py
"""Tests for the InputBehavior base poll loop and BehaviorContext."""

from __future__ import annotations

import pytest

from glider.hal.input_behavior import BehaviorContext, InputBehavior


class _ScriptedDevice:
    """Yields a fixed sequence; repeats the last value once exhausted.

    Supports both read() (read_action=None) and execute_action(name) so the
    same fake drives raw-angle behaviors (read_action="angle") too.
    """

    def __init__(self, values):
        self._values = list(values)
        self._i = 0
        self.id = "scripted"

    async def read(self):
        v = self._values[self._i]
        if self._i < len(self._values) - 1:
            self._i += 1
        return v

    async def execute_action(self, name, *a, **k):
        return await self.read()


class _FailingDevice:
    id = "flaky"

    async def read(self):
        raise OSError("read failed")


class _ThresholdBehavior(InputBehavior):
    key = "threshold"
    label = "Threshold"
    settings = [{"key": "limit", "label": "Limit", "type": "int", "default": 5}]

    def check(self, value, settings, ctx):
        return value >= settings["limit"]


async def _ctx(device, poll=0.0):
    return BehaviorContext(device=device, hardware_manager=None, poll_interval=poll)


async def test_triggers_and_returns_value():
    beh = _ThresholdBehavior()
    ctx = await _ctx(_ScriptedDevice([0, 2, 4, 6, 8]))
    value = await beh.wait_for_input({"limit": 5}, ctx, timeout=1.0)
    assert value == 6


async def test_scratch_is_fresh_per_wait():
    # Two waits on the same behavior instance must not share scratch state.
    beh = _CountingBehavior()
    v1 = await beh.wait_for_input({"target": 3}, await _ctx(_ScriptedDevice([1, 1, 1, 1])), 1.0)
    v2 = await beh.wait_for_input({"target": 3}, await _ctx(_ScriptedDevice([1, 1, 1, 1])), 1.0)
    assert v1 == 1 and v2 == 1  # each counted from zero, not 3 then 6


class _CountingBehavior(InputBehavior):
    key = "count"
    label = "Count"
    settings = []

    def check(self, value, settings, ctx):
        ctx.scratch["n"] = ctx.scratch.get("n", 0) + 1
        return ctx.scratch["n"] >= settings["target"]


async def test_times_out():
    beh = _ThresholdBehavior()
    ctx = await _ctx(_ScriptedDevice([0, 1, 2]))  # never reaches 5
    with pytest.raises(TimeoutError):
        await beh.wait_for_input({"limit": 5}, ctx, timeout=0.2)


async def test_read_error_raises_runtime_error():
    beh = _ThresholdBehavior()
    ctx = await _ctx(_FailingDevice())
    with pytest.raises(RuntimeError):
        await beh.wait_for_input({"limit": 5}, ctx, timeout=1.0)


async def test_cleanup_runs_on_trigger_timeout_and_error():
    calls = []

    class _CleanupBehavior(_ThresholdBehavior):
        async def cleanup(self, ctx):
            calls.append("cleanup")

    # trigger
    await _CleanupBehavior().wait_for_input({"limit": 1}, await _ctx(_ScriptedDevice([2])), 1.0)
    # timeout
    with pytest.raises(TimeoutError):
        await _CleanupBehavior().wait_for_input(
            {"limit": 9}, await _ctx(_ScriptedDevice([0, 0])), 0.2
        )
    # error
    with pytest.raises(RuntimeError):
        await _CleanupBehavior().wait_for_input({"limit": 1}, await _ctx(_FailingDevice()), 1.0)
    assert calls == ["cleanup", "cleanup", "cleanup"]


async def test_read_action_uses_execute_action():
    seen = []

    class _ActionDevice(_ScriptedDevice):
        async def execute_action(self, name, *a, **k):
            seen.append(name)
            return await self.read()

    class _AngleBehavior(_ThresholdBehavior):
        read_action = "angle"

    ctx = await _ctx(_ActionDevice([9]))
    await _AngleBehavior().wait_for_input({"limit": 1}, ctx, 1.0)
    assert seen == ["angle"]


async def test_on_sample_skipped_on_triggering_sample():
    samples = []

    class _SideEffectBehavior(_ThresholdBehavior):
        async def on_sample(self, value, settings, ctx):
            samples.append(value)

    ctx = await _ctx(_ScriptedDevice([0, 1, 9]))  # triggers on 9 (limit 5)
    await _SideEffectBehavior().wait_for_input({"limit": 5}, ctx, 1.0)
    assert samples == [0, 1]  # on_sample not called for the triggering value 9
