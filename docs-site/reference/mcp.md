# AI Agents (MCP)

GLIDER includes an [MCP](https://modelcontextprotocol.io) server, `glider-mcp`,
so an AI agent (Claude Desktop, Claude Code, Cursor, or any MCP client) can
design experiments and analyze your recordings.

!!! warning "Agents cannot run hardware"
    The server never connects a board or starts an experiment. An agent writes
    `.glider` files; you open and run them in GLIDER.

## Install

```bash
pip install "glider[mcp]"
```

The installer builds do not include `glider-mcp` yet.

## Connect an agent

Claude Code:

```bash
claude mcp add glider -- glider-mcp
```

Claude Desktop and most other clients take a JSON entry:

```json
{"mcpServers": {"glider": {"command": "glider-mcp"}}}
```

If `glider-mcp` is not on the client's `PATH`, use its full path
(`which glider-mcp` on macOS/Linux, `where glider-mcp` on Windows).

## Tools

| Designing experiments | |
|---|---|
| `list_node_types` | Node types and their ports |
| `list_device_types` | Device types (with required pins) and board drivers |
| `new_experiment` | A minimal Start → End experiment to build on |
| `read_experiment` | An experiment's JSON |
| `validate_experiment` | Everything that would stop GLIDER opening or starting it |
| `save_experiment` | Validates, lays out new nodes, writes the file |

| Analyzing recordings | |
|---|---|
| `find_recordings` | Every recording under a folder, with its group |
| `session_summary` | What a recording holds; start here |
| `ethogram` | Bouts and per-state totals, or state transitions |
| `trajectory` | Zone dwell, zone transitions, or the position trace |
| `kinematics` | Speed statistics and distance |
| `events` | Events, or tracking around them |
| `project_doctor` | Problems in a project that would cost a result |
| `plot` | Ethogram, trajectory, occupancy, zone dwell or velocity as a PNG |

Agents only write `.glider`, `.csv` and `.png` files, only at absolute paths
they name, and never replace an existing file unless told to with
`save_experiment(..., overwrite=true)`.

## Try it

> Make me an experiment at ~/glider/blink.glider: an Arduino Uno with an LED on
> pin 13 that turns on, waits two seconds, and ends.

> Summarize the recording in ~/data/mouse3 and plot its ethogram.
