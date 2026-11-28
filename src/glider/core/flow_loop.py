"""The flow loop: one asyncio loop thread that owns the flow and the hardware.

With the setting on, FlowEngine, HardwareManager and every board and device
coroutine run here instead of on the qasync GUI loop. Paint events, video
frames and widget work then stop delaying flow steps: the flow-clock benchmark
put the worst hand-off on the GUI loop at ~27 ms against ~4 ms on a loop of its
own.

The GUI does not change shape. Its coroutines still run on the GUI loop and
still ``await core.start_experiment()`` or ``await device.set_state(True)``;
those methods carry :func:`on_flow_loop`, which hops the call onto this loop
and awaits the result. A caller already on the flow loop, or any caller while
the loop is not running, goes straight through. So nested calls never hop
twice, and with the setting off (the default) nothing changes at all.

Callbacks go the other way through ``pyqtSignal``s the GUI already uses, which
Qt queues onto the GUI thread whichever thread emits them.

Board drivers and the zone/behavior nodes capture ``get_running_loop()`` when
they connect or start, and post their callbacks to it. Because connect and
start hop, that captured loop is this one.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import functools
import inspect
import logging
import os
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Env var that overrides the ``timing.flow_loop_thread`` setting: "1"/"0".
ENV_VAR = "GLIDER_FLOW_THREAD"

#: How long the e-stop waits for the flow loop to answer before running the
#: stop on the caller's loop instead. A wedged loop must not block an e-stop.
ESTOP_PING_TIMEOUT_S = 0.25

# Set while a call must run on the caller's loop even though a flow loop is
# active: the e-stop fallback, when the flow loop has stopped answering.
_bypass: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "glider_flow_loop_bypass", default=False
)

_active: FlowLoopThread | None = None
# The loop that started the flow loop: the GUI's qasync loop in the app.
_home_loop: asyncio.AbstractEventLoop | None = None
_active_lock = threading.Lock()


class FlowLoopThread:
    """A daemon thread running one asyncio event loop until stopped."""

    def __init__(self, name: str = "glider-flow-loop") -> None:
        self._name = name
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def start(self, timeout: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("Flow loop thread did not start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.call_soon(self._ready.set)
        try:
            loop.run_forever()
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    def stop(self, timeout: float = 5.0) -> None:
        loop, thread = self._loop, self._thread
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("Flow loop thread did not stop within %.1fs", timeout)
        self._thread = None
        self._loop = None

    # -- state -------------------------------------------------------------
    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    @property
    def is_running(self) -> bool:
        loop = self._loop
        return (
            loop is not None
            and not loop.is_closed()
            and loop.is_running()
            and self._thread is not None
            and self._thread.is_alive()
        )

    def is_current(self) -> bool:
        """True when called from code running on this loop's thread."""
        return self._thread is not None and threading.current_thread() is self._thread

    # -- hand-offs ---------------------------------------------------------
    def submit(self, coro: Coroutine[Any, Any, T]):
        """Schedule *coro* on the flow loop; returns a concurrent Future."""
        loop = self._loop
        if loop is None or loop.is_closed():
            coro.close()
            raise RuntimeError("Flow loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, loop)

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run *coro* on the flow loop and await it from another loop.

        Cancelling the awaiting task cancels the task on the flow loop.
        """
        return await asyncio.wrap_future(self.submit(coro))

    def call_soon(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run a plain callable on the flow loop's thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("Flow loop is not running")
        loop.call_soon_threadsafe(fn, *args)

    def ping(self, timeout: float) -> bool:
        """True if the loop runs a callback within *timeout* seconds."""
        if self.is_current():
            return True
        if not self.is_running:
            return False
        answered = threading.Event()
        try:
            self.call_soon(answered.set)
        except RuntimeError:
            return False
        return answered.wait(timeout)


# -- the active flow loop ------------------------------------------------------
def active() -> FlowLoopThread | None:
    """The running flow loop, or None when the flow runs on the caller's loop."""
    fl = _active
    if fl is None or not fl.is_running:
        return None
    return fl


def start(home_loop: asyncio.AbstractEventLoop | None = None) -> FlowLoopThread:
    """Start the flow loop thread and make it the active one. Idempotent.

    *home_loop* is the GUI loop, where :func:`call_on_gui` sends Qt work.
    """
    global _active, _home_loop
    with _active_lock:
        if home_loop is not None:
            _home_loop = home_loop
        if _active is not None and _active.is_running:
            return _active
        fl = FlowLoopThread()
        fl.start()
        _active = fl
        logger.info("Flow loop thread started: flow and hardware run off the GUI thread")
        return fl


def stop() -> None:
    """Stop the active flow loop thread, if any."""
    global _active, _home_loop
    with _active_lock:
        fl, _active = _active, None
        _home_loop = None
    if fl is not None:
        fl.stop()
        logger.info("Flow loop thread stopped")


def enabled_by_setting() -> bool:
    """Whether the flow loop thread should run: env var first, then config."""
    env = os.environ.get(ENV_VAR, "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    try:
        from glider.core.config import get_config

        return bool(get_config().timing.flow_loop_thread)
    except Exception:  # noqa: BLE001 - a bad config must not stop startup
        logger.debug("Could not read flow_loop_thread setting", exc_info=True)
        return False


def _should_hop() -> FlowLoopThread | None:
    if _bypass.get():
        return None
    fl = active()
    if fl is None or fl.is_current():
        return None
    return fl


# -- decorators ----------------------------------------------------------------
def on_flow_loop(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Run the decorated coroutine function on the flow loop.

    Goes straight through when no flow loop is active, when the caller is
    already on it, or inside :func:`bypass_flow_loop`.
    """
    if getattr(fn, "__glider_flow_loop__", False):
        return fn

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        fl = _should_hop()
        if fl is None:
            return await fn(*args, **kwargs)
        return await fl.run(fn(*args, **kwargs))

    wrapper.__glider_flow_loop__ = True  # type: ignore[attr-defined]
    return wrapper


def call_on_flow_loop(fn: Callable[..., Any]) -> Callable[..., None]:
    """Make a sync, fire-and-forget method run on the flow loop's thread.

    For methods such as ``start_link_supervisor`` that create a task on the
    running loop: called from the GUI thread, the task must still land on the
    flow loop.
    """
    if getattr(fn, "__glider_flow_loop__", False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        fl = _should_hop()
        if fl is None:
            fn(*args, **kwargs)
            return
        fl.call_soon(functools.partial(fn, *args, **kwargs))

    wrapper.__glider_flow_loop__ = True  # type: ignore[attr-defined]
    return wrapper


def wrap_public_coroutines(cls: type) -> None:
    """Apply :func:`on_flow_loop` to every public coroutine method *cls* defines.

    Only names defined on *cls* itself are touched; inherited ones were wrapped
    on the class that defined them. Names starting with ``_`` are left alone, and
    so is anything already wrapped.
    """
    for name, attr in list(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(attr, staticmethod | classmethod):
            continue
        if inspect.iscoroutinefunction(attr) and not getattr(attr, "__glider_flow_loop__", False):
            setattr(cls, name, on_flow_loop(attr))


@contextlib.contextmanager
def bypass_flow_loop():
    """Run the calls made inside this block on the caller's loop."""
    token = _bypass.set(True)
    try:
        yield
    finally:
        _bypass.reset(token)


async def run_estop(make_coro: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an e-stop on the flow loop, or locally if the flow loop is wedged.

    *make_coro* is called once, on whichever loop the stop ends up on, so a
    coroutine is never built and then dropped.
    """
    fl = _should_hop()
    if fl is None:
        return await make_coro()
    if fl.ping(ESTOP_PING_TIMEOUT_S):
        return await fl.run(make_coro())
    logger.error(
        "Flow loop did not answer within %.0f ms: running EMERGENCY STOP on the caller's loop",
        ESTOP_PING_TIMEOUT_S * 1000,
    )
    with bypass_flow_loop():
        return await make_coro()


def estop_entry(fn: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Awaitable[T]]:
    """Mark an e-stop coroutine method: it goes through :func:`run_estop`."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        return await run_estop(lambda: fn(*args, **kwargs))

    wrapper.__glider_flow_loop__ = True  # type: ignore[attr-defined]
    return wrapper


def post_to_loop(
    loop: asyncio.AbstractEventLoop | None, fn: Callable[..., Any], *args: Any
) -> bool:
    """Run *fn* on *loop*'s thread: inline if already there. False if it can't."""
    if loop is None or loop.is_closed():
        return False
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        fn(*args)
        return True
    try:
        loop.call_soon_threadsafe(fn, *args)
    except RuntimeError:
        return False
    return True


async def call_on_gui(fn: Callable[..., T], *args: Any) -> T:
    """Run a plain callable on the GUI thread and await its result.

    For node code that has to build or drive Qt objects. Inline when the flow
    loop is off or the caller is not on it.
    """
    fl = active()
    home = _home_loop
    if fl is None or home is None or not fl.is_current():
        return fn(*args)
    result: concurrent.futures.Future = concurrent.futures.Future()

    def _run() -> None:
        if not result.set_running_or_notify_cancel():
            return
        try:
            result.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001 - handed to the awaiting side
            result.set_exception(exc)

    home.call_soon_threadsafe(_run)
    return await asyncio.wrap_future(result)
