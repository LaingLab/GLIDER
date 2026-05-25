"""Tests for flow control nodes: DelayNode, TimerNode."""

from unittest.mock import AsyncMock, patch

import pytest

from glider.nodes.logic.flow_nodes import DelayNode, TimerNode


class TestDelayNode:
    """Tests for DelayNode unit-aware duration handling.

    DelayNode runs its sleep via ``asyncio.to_thread(event.wait, duration)``
    so the wait happens off the qasync event loop (see the docstring on
    ``DelayNode`` for why). These tests patch ``asyncio.to_thread`` to
    capture the resolved duration without actually sleeping.
    """

    async def _capture_resolved_duration(self, node: DelayNode) -> float:
        """Run ``node.execute()`` and return the duration value handed
        to ``asyncio.to_thread`` (which is the second arg to that call —
        ``to_thread(event.wait, duration)``).
        """
        # _fire_exec_output requires no real wiring; stub it out so the
        # call doesn't fail when execute() returns from the (mocked) sleep.
        node._fire_exec_output = AsyncMock()  # type: ignore[assignment]

        with patch("glider.nodes.logic.flow_nodes.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = False  # event.wait returns False on timeout
            await node.execute()
            assert mock_to_thread.await_count == 1, (
                f"Expected asyncio.to_thread to be awaited once, "
                f"got {mock_to_thread.await_count} times"
            )
            # to_thread(callable, *args) — duration is the second positional
            args, _kwargs = mock_to_thread.await_args
            return args[1]

    @pytest.mark.asyncio
    async def test_execute_seconds_default_unit(self):
        """Duration without unit defaults to seconds."""
        node = DelayNode()
        node._state["duration"] = 0.5
        assert await self._capture_resolved_duration(node) == 0.5

    @pytest.mark.asyncio
    async def test_execute_milliseconds_unit_converts_to_seconds(self):
        """unit='milliseconds' divides duration by 1000 before sleeping."""
        node = DelayNode()
        node._state["duration"] = 500
        node._state["unit"] = "milliseconds"
        assert await self._capture_resolved_duration(node) == 0.5

    @pytest.mark.asyncio
    async def test_execute_milliseconds_unit_with_port_input(self):
        """When no state duration, port input is interpreted under the unit."""
        node = DelayNode()
        node._state["unit"] = "milliseconds"
        # Input port index 1 is Duration
        node._inputs[1] = 250
        assert await self._capture_resolved_duration(node) == 0.25

    @pytest.mark.asyncio
    async def test_execute_negative_clamped_to_zero(self):
        """Negative durations are clamped to 0 regardless of unit."""
        node = DelayNode()
        node._state["duration"] = -5
        node._state["unit"] = "milliseconds"
        assert await self._capture_resolved_duration(node) == 0


class TestTimerNode:
    """Tests for TimerNode unit-aware interval handling."""

    def test_effective_interval_seconds_default(self):
        """Default interval unit is seconds (no conversion)."""
        node = TimerNode()
        node._inputs[0] = 2.0  # Interval port
        assert node._effective_interval() == 2.0

    def test_effective_interval_milliseconds(self):
        """unit='milliseconds' divides interval by 1000."""
        node = TimerNode()
        node._state["unit"] = "milliseconds"
        node._inputs[0] = 500
        assert node._effective_interval() == 0.5

    def test_effective_interval_none_returns_default_seconds(self):
        """When input is None and no state, falls back to 1.0 seconds."""
        node = TimerNode()
        node._inputs[0] = None
        assert node._effective_interval() == 1.0

    def test_effective_interval_enforces_minimum(self):
        """Interval below 10 ms (0.01 s) is clamped upward."""
        node = TimerNode()
        node._state["unit"] = "milliseconds"
        node._inputs[0] = 1  # 1 ms -> below 10 ms floor
        assert node._effective_interval() == 0.01

    def test_effective_interval_state_overrides_port(self):
        """State 'interval' takes precedence over port input."""
        node = TimerNode()
        node._inputs[0] = 5.0  # would be seconds via port
        node._state["interval"] = 250
        node._state["unit"] = "milliseconds"
        assert node._effective_interval() == 0.25
