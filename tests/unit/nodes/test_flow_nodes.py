"""Tests for flow control nodes: DelayNode, TimerNode."""

import asyncio
from unittest.mock import patch

import pytest

from glider.nodes.logic.flow_nodes import DelayNode, TimerNode


class TestDelayNode:
    """Tests for DelayNode unit-aware duration handling."""

    @pytest.mark.asyncio
    async def test_execute_seconds_default_unit(self):
        """Duration without unit defaults to seconds."""
        node = DelayNode()
        node._state["duration"] = 0.5

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            await node.execute()

        # Called once with seconds value
        mock_sleep.assert_awaited_once_with(0.5)

    @pytest.mark.asyncio
    async def test_execute_milliseconds_unit_converts_to_seconds(self):
        """unit='milliseconds' divides duration by 1000 before sleeping."""
        node = DelayNode()
        node._state["duration"] = 500
        node._state["unit"] = "milliseconds"

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            await node.execute()

        mock_sleep.assert_awaited_once_with(0.5)

    @pytest.mark.asyncio
    async def test_execute_milliseconds_unit_with_port_input(self):
        """When no state duration, port input is interpreted under the unit."""
        node = DelayNode()
        node._state["unit"] = "milliseconds"
        # Input port index 1 is Duration
        node._inputs[1] = 250

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            await node.execute()

        mock_sleep.assert_awaited_once_with(0.25)

    @pytest.mark.asyncio
    async def test_execute_negative_clamped_to_zero(self):
        """Negative durations are clamped to 0 regardless of unit."""
        node = DelayNode()
        node._state["duration"] = -5
        node._state["unit"] = "milliseconds"

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            await node.execute()

        mock_sleep.assert_awaited_once_with(0)
