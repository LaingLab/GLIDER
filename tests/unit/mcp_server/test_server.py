"""The server end to end: tool list, error mapping, and a clean stdout."""

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.types import LATEST_PROTOCOL_VERSION  # noqa: E402

from glider.mcp.server import server  # noqa: E402

TOOLS = {
    "list_node_types",
    "list_device_types",
    "new_experiment",
    "read_experiment",
    "validate_experiment",
    "save_experiment",
    "find_recordings",
    "session_summary",
    "ethogram",
    "trajectory",
    "kinematics",
    "events",
    "project_doctor",
    "plot",
}
EXAMPLE = Path(__file__).parents[3] / "examples" / "pi_only.glider"


async def test_every_tool_is_registered_with_a_description():
    tools = await server.list_tools()
    assert {t.name for t in tools} == TOOLS
    assert all(t.description for t in tools)


async def test_tool_call_returns_json_text():
    result = await server.call_tool("validate_experiment", {"path": str(EXAMPLE)})
    assert not result.is_error
    assert json.loads(result.content[0].text)["valid"] is True


async def test_agent_fixable_errors_reach_the_agent():
    # In process, call_tool raises; over the wire the same ToolError becomes
    # an is_error result whose text is this message.
    with pytest.raises(ToolError, match="absolute"):
        await server.call_tool("new_experiment", {"path": "relative.glider", "name": "x"})


def test_stdout_carries_only_protocol_frames(tmp_path):
    """Core init, plugin loading and a tool call must not write to stdout.

    stdin stays open until the reply arrives: on EOF the server cancels any
    request still running and answers "Connection closed".
    """
    frames = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "list_node_types", "arguments": {}},
        },
    ]
    # stderr goes to a file: an unread pipe fills (~4 KB on Windows) and blocks the server.
    stderr_log = tmp_path / "stderr.log"
    with stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "glider.mcp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            encoding="utf-8",
        )
    watchdog = threading.Timer(120, proc.kill)
    watchdog.start()
    try:
        proc.stdin.write("".join(json.dumps(f) + "\n" for f in frames))
        proc.stdin.flush()
        reply = None
        for line in proc.stdout:
            if not line.strip():
                continue
            message = json.loads(line)  # any stray print fails here
            if message.get("id") == 2:
                reply = message
                break
        assert reply is not None, stderr_log.read_text(encoding="utf-8")[-2000:]
        assert (
            "error" not in reply
        ), f"{reply['error']}\n{stderr_log.read_text(encoding='utf-8')[-2000:]}"
        assert "StartExperiment" in reply["result"]["content"][0]["text"]
    finally:
        watchdog.cancel()
        proc.stdin.close()
        proc.wait(timeout=30)
