"""glider-mcp: lets AI agents design GLIDER experiments and analyze recordings.

stdio transport. stdout is the protocol channel, so logging goes to stderr and
nothing in glider.mcp prints. v1 never connects hardware or runs a flow.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import sys
from collections.abc import Callable

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from glider.mcp import analysis, experiments

INSTRUCTIONS = """\
GLIDER designs and analyzes behavioral neuroscience experiments.

Designing: call list_node_types and list_device_types, then new_experiment or
read_experiment, edit the JSON, run validate_experiment until it is valid, and
save_experiment. Files use GLIDER's session format. Ask the user before
overwriting their files. This server never connects hardware or runs an
experiment; the user runs it in GLIDER.

Analyzing: find_recordings, then session_summary on a recording folder; it
names the states, zones and event sources the other tools accept. Pass out_csv
for long tables and keep the preview in context.
"""

server = MCPServer("glider", instructions=INSTRUCTIONS)
# The headless core is shared state; tools run one at a time.
_lock = asyncio.Lock()


def _register(fn: Callable) -> None:
    @functools.wraps(fn)
    async def tool(*args, **kwargs):
        async with _lock:
            try:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except (ValueError, OSError) as e:
                # The agent can fix these; without ToolError it would see only
                # "Error executing tool".
                raise ToolError(str(e)) from e
        return result

    server.tool()(tool)


async def plot(path: str, kind: str, out_path: str, object_id: int = 0) -> Image:
    """Render a recording plot to a PNG at out_path and show it.

    kind: "ethogram", "trajectory", "occupancy", "zone_dwell" or "velocity".
    out_path: absolute path ending in .png; must not exist yet.
    """
    png = analysis.plot(path, kind, out_path, object_id)
    return Image(data=png.read_bytes(), format="png")


for _fn in (
    experiments.list_node_types,
    experiments.list_device_types,
    experiments.new_experiment,
    experiments.read_experiment,
    experiments.validate_experiment,
    experiments.save_experiment,
    analysis.find_recordings,
    analysis.session_summary,
    analysis.ethogram,
    analysis.trajectory,
    analysis.kinematics,
    analysis.events,
    analysis.project_doctor,
    plot,
):
    _register(_fn)


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    server.run()


if __name__ == "__main__":
    main()
