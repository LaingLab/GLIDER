# AI Agents (MCP)

GLIDER includes an [MCP](https://modelcontextprotocol.io) server, `glider-mcp`,
so an AI agent (Claude Desktop, Claude Code, Cursor, or any MCP client) can
design experiments and analyze your recordings.

!!! warning "Agents cannot run hardware"
    The server never connects a board or starts an experiment. An agent writes
    `.glider` files; you open and run them in GLIDER.

## Install

From your GLIDER clone (see [Installation](../getting-started/installation.md)):

```bash
uv sync --extra pc --extra mcp
```

`uv sync` installs only the extras you list, so keep any others you already
use (for example `--extra vision`). This puts `glider-mcp` in the clone's
virtual environment: `.venv/bin/glider-mcp` on macOS/Linux,
`.venv\Scripts\glider-mcp.exe` on Windows.

The installer builds do not include `glider-mcp` yet.

## Connect an agent

Claude Code:

```bash
claude mcp add glider -- /path/to/glider/.venv/bin/glider-mcp
```

On Windows, use `C:\path\to\glider\.venv\Scripts\glider-mcp.exe`.

Claude Desktop and most other clients take a JSON entry:

```json
{"mcpServers": {"glider": {"command": "/path/to/glider/.venv/bin/glider-mcp"}}}
```

Always give the full path to the clone's script; `glider-mcp` is not on your
`PATH` unless you activate the virtual environment first.

## Tools

| Designing experiments | |
|---|---|
| `list_node_types` | Node types, their ports, and the settings each accepts |
| `list_device_types` | Device types (required pins, settings) and board drivers |
| `list_examples` | The shipped example experiments, to read as templates |
| `new_experiment` | A minimal Start → End experiment to build on |
| `read_experiment` | An experiment's JSON |
| `validate_experiment` | Everything that would stop GLIDER opening or starting it |
| `save_experiment` | Validates, lays out new nodes, writes the file |

| Analyzing recordings | |
|---|---|
| `find_recordings` | Every recording under a folder, with its group |
| `session_summary` | What a recording holds, including time in each state; start here |
| `ethogram` | Bouts and per-state totals, or state transitions |
| `trajectory` | Zone dwell, zone transitions, or the position trace |
| `kinematics` | Speed statistics and distance |
| `events` | Events, or tracking around them |
| `project_doctor` | Problems in a project that would cost a result |
| `plot` | Ethogram, trajectory, occupancy, zone dwell or velocity as a PNG |

Agents only write `.glider`, `.csv` and `.png` files, only at absolute paths
they name, and never replace an existing file unless told to with
`save_experiment(..., overwrite=true)`.

## Seeing an agent's changes in GLIDER

If the experiment an agent edits is open in GLIDER, GLIDER picks the change up
on its own: with no unsaved edits it reloads straight away; with unsaved edits
a bar offers **Reload** (take the agent's version) or **Keep mine** (your next
save overwrites it). During a run nothing changes until the run ends. GLIDER
also checks the file about once a second, so this works for experiments on a
network share too.

## Try it

> Make me an experiment at ~/glider/blink.glider: an Arduino Uno with an LED on
> pin 13 that turns on, waits two seconds, and ends.

> Summarize the recording in ~/data/mouse3 and plot its ethogram.
