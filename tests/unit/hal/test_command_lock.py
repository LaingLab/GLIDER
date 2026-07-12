"""Per-device command serialization (D13) + the value_spec accessor precedence.

The lock test proves that two commands to the SAME device never interleave their
hardware steps: an action that yields mid-body must still run start-to-end before
the next command begins.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from glider.hal.base_device import BaseDevice, DeviceConfig
from glider.hal.value_spec import KIND_WHOLE, ActionValueSpec


class _ProbeDevice(BaseDevice):
    """Minimal concrete device: one action that yields control mid-execution."""

    def __init__(self, board, config, name=None, declared=None):
        super().__init__(board, config, name)
        self.log: list[str] = []
        self._declared = declared  # optional ActionValueSpec for "go"

    @property
    def device_type(self) -> str:
        return "Probe"

    @property
    def actions(self):
        return {"go": self._go, "noop": self._noop}

    async def _go(self, tag: str) -> None:
        self.log.append(f"{tag}-start")
        # Two yields: without the per-device lock, a second command would slip
        # in here and the log would interleave.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.log.append(f"{tag}-end")

    def _noop(self) -> str:  # sync action → runs via to_thread
        return "ok"

    def _declared_value_spec(self, action_name: str):
        return self._declared if action_name == "go" else None

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    @classmethod
    def from_dict(cls, data, board):
        return cls(board, DeviceConfig())


def _make(**kw) -> _ProbeDevice:
    dev = _ProbeDevice(MagicMock(), DeviceConfig(), **kw)
    dev._initialized = True
    return dev


# --- D13 per-device serialization -------------------------------------------


async def test_concurrent_commands_to_one_device_do_not_interleave():
    dev = _make()
    await asyncio.gather(dev.execute_action("go", "A"), dev.execute_action("go", "B"))
    # Serialized in arrival order — not ["A-start", "B-start", "A-end", "B-end"].
    assert dev.log == ["A-start", "A-end", "B-start", "B-end"]


async def test_lock_released_after_each_command():
    dev = _make()
    await dev.execute_action("go", "A")
    await dev.execute_action("go", "B")
    assert dev.log == ["A-start", "A-end", "B-start", "B-end"]
    assert not dev._command_lock.locked()


async def test_sync_action_also_serializes_through_the_lock():
    dev = _make()
    result = await dev.execute_action("noop")
    assert result == "ok"
    assert not dev._command_lock.locked()


# --- value_spec accessor precedence -----------------------------------------


def test_value_spec_returns_none_when_nothing_declared():
    dev = _make()
    assert dev.value_spec("go") is None


def test_value_spec_prefers_declared_over_fallback():
    spec = ActionValueSpec(KIND_WHOLE, 0, 100, unit="mL/min")
    dev = _make(declared=spec)
    assert dev.value_spec("go") is spec
    assert dev.value_spec("noop") is None
