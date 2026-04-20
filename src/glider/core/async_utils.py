"""Helpers for fire-and-forget asyncio tasks."""

import asyncio
import logging

logger = logging.getLogger(__name__)


def log_task_exception(task: asyncio.Task) -> None:
    """
    Done-callback that surfaces exceptions from fire-and-forget tasks.

    Use as: `task.add_done_callback(log_task_exception)`. Without this,
    Python's default behavior only prints a "Task exception was never retrieved"
    warning when the task is garbage-collected, which happens non-deterministically
    and loses the traceback.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Unhandled error in background task: %s", exc, exc_info=exc)
