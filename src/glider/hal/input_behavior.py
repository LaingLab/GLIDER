# src/glider/hal/input_behavior.py
"""Plugin-declared input behaviors for the WaitForInput node.

A device advertises ``input_behaviors`` (a sibling of ``actions``): named
things the WaitForInput node can wait on, each with its own settings schema and
trigger logic. A behavior instance is a STATELESS descriptor — the same instance
may be reused across runs — so all per-wait scratch state lives on the
``BehaviorContext`` the node creates fresh for each wait.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_READ_ERRORS = 3  # consecutive read failures before aborting the wait


@dataclass
class BehaviorContext:
    """Per-wait context passed to every behavior call.

    ``scratch`` is a fresh, mutable store for cross-sample state (last value,
    accumulated counts, resolved helper devices, ...). ``hardware_manager`` lets
    a behavior resolve a ``device_ref`` setting to a second device (e.g. a ramp
    PWM motor distinct from the bound sensor).
    """

    device: Any
    hardware_manager: Any = None
    poll_interval: float = 0.05
    scratch: dict[str, Any] = field(default_factory=dict)


class InputBehavior:
    """Base class: a stateless descriptor + the shared poll loop.

    Subclasses set ``key``/``label``/``settings`` and implement ``check``.
    Optional: ``read_action`` (which quantity to sample), ``on_sample`` (per-poll
    side effects, e.g. ramp a motor), ``cleanup`` (always-runs teardown).
    Exotic devices override ``wait_for_input`` entirely but MUST still run
    cleanup on every exit (trigger, timeout, error).
    """

    key: str = ""
    label: str = ""
    settings: list[dict] = []
    read_action: str | None = None  # None -> device.read(); else execute_action(name)

    def check(self, value: Any, settings: dict, ctx: BehaviorContext) -> bool:
        """Return True to trigger. Keep cross-sample state in ctx.scratch."""
        raise NotImplementedError

    async def on_sample(self, value: Any, settings: dict, ctx: BehaviorContext) -> None:
        """Per-poll side effects on non-triggering samples. Default no-op."""

    async def cleanup(self, ctx: BehaviorContext) -> None:
        """Teardown that runs on every exit. Default no-op."""

    async def _read(self, ctx: BehaviorContext) -> Any:
        if self.read_action is None:
            return await ctx.device.read()
        return await ctx.device.execute_action(self.read_action)

    async def wait_for_input(self, settings: dict, ctx: BehaviorContext, timeout: float) -> Any:
        """Standard poll loop: sample -> check -> (on_sample) -> sleep.

        Raises TimeoutError when ``timeout`` (>0) elapses, or RuntimeError after
        ``_MAX_READ_ERRORS`` consecutive read failures. ``cleanup`` runs in
        ``finally`` so a ramp motor is stopped on trigger, timeout, and error.
        """
        start = time.monotonic()
        errors = 0
        try:
            while True:
                if timeout > 0 and (time.monotonic() - start) >= timeout:
                    raise TimeoutError(f"Behavior '{self.key}' timed out after {timeout}s")
                try:
                    value = await self._read(ctx)
                    errors = 0
                except Exception as e:
                    errors += 1
                    logger.error("Behavior read error (%d/%d): %s", errors, _MAX_READ_ERRORS, e)
                    if errors >= _MAX_READ_ERRORS:
                        raise RuntimeError(f"Device read failed: {e}") from e
                    await asyncio.sleep(ctx.poll_interval)
                    continue

                if self.check(value, settings, ctx):
                    return value
                await self.on_sample(value, settings, ctx)
                await asyncio.sleep(ctx.poll_interval)
        finally:
            await self.cleanup(ctx)
