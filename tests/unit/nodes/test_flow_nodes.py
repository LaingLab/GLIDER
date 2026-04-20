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
