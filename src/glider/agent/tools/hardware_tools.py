"""
Hardware Tools

Tools for configuring boards and devices.
"""

import logging
from typing import TYPE_CHECKING, Any

from glider.agent.actions import ActionType, AgentAction
from glider.agent.llm_backend import ToolDefinition

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


# Tool Definitions
HARDWARE_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="list_boards",
        description="List all configured hardware boards",
        parameters={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="add_board",
        description="Add a new hardware board (Arduino or Raspberry Pi)",
        parameters={
            "type": "object",
            "properties": {
                "board_type": {
                    "type": "string",
                    "description": "Type of board",
                    "enum": ["arduino", "telemetrix", "raspberry_pi", "pigpio"],
                },
                "name": {"type": "string", "description": "Friendly name for the board"},
                "port": {
                    "type": "string",
                    "description": "Serial port (e.g., 'COM3', '/dev/ttyUSB0') - required for Arduino",
                },
            },
            "required": ["board_type", "name"],
        },
    ),
    ToolDefinition(
        name="remove_board",
        description="Remove a board and all its devices (DESTRUCTIVE)",
        parameters={
            "type": "object",
            "properties": {
                "board_id": {"type": "string", "description": "ID of the board to remove"}
            },
            "required": ["board_id"],
        },
    ),
    ToolDefinition(
        name="connect_board",
        description="Connect to a configured board",
        parameters={
            "type": "object",
            "properties": {
                "board_id": {"type": "string", "description": "ID of the board to connect"}
            },
            "required": ["board_id"],
        },
    ),
    ToolDefinition(
        name="disconnect_board",
        description="Disconnect from a board",
        parameters={
            "type": "object",
            "properties": {
                "board_id": {"type": "string", "description": "ID of the board to disconnect"}
            },
            "required": ["board_id"],
        },
    ),
    ToolDefinition(
        name="list_devices",
        description="List all configured devices",
        parameters={
            "type": "object",
            "properties": {
                "board_id": {"type": "string", "description": "Filter by board ID (optional)"}
            },
        },
    ),
    ToolDefinition(
        name="add_device",
        description="Add a new device to a board",
        parameters={
            "type": "object",
            "properties": {
                "board_id": {"type": "string", "description": "ID of the board to add device to"},
                "device_type": {
                    "type": "string",
                    "description": "Type of device",
                    "enum": ["DigitalOutput", "DigitalInput", "AnalogInput", "PWMOutput", "Servo"],
                },
                "name": {
                    "type": "string",
                    "description": "Friendly name for the device (e.g., 'led', 'button')",
                },
                "pin": {"type": "integer", "description": "Pin number on the board"},
                "settings": {
                    "type": "object",
                    "description": "Device-specific settings",
                    "additionalProperties": True,
                },
            },
            "required": ["board_id", "device_type", "name", "pin"],
        },
    ),
    ToolDefinition(
        name="remove_device",
        description="Remove a device",
        parameters={
            "type": "object",
            "properties": {
                "device_id": {"type": "string", "description": "ID of the device to remove"}
            },
            "required": ["device_id"],
        },
    ),
    ToolDefinition(
        name="configure_device",
        description="Update device settings",
        parameters={
            "type": "object",
            "properties": {
                "device_id": {"type": "string", "description": "ID of the device to configure"},
                "settings": {
                    "type": "object",
                    "description": "Settings to update",
                    "additionalProperties": True,
                },
            },
            "required": ["device_id", "settings"],
        },
    ),
    ToolDefinition(
        name="scan_ports",
        description="Scan for available serial ports (for Arduino)",
        parameters={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="test_device",
        description="Test a device (e.g., blink LED, read sensor)",
        parameters={
            "type": "object",
            "properties": {
                "device_id": {"type": "string", "description": "ID of the device to test"},
                "action": {
                    "type": "string",
                    "description": "Test action to perform",
                    "enum": ["blink", "on", "off", "read", "toggle"],
                },
                "duration_ms": {
                    "type": "integer",
                    "description": "Duration for blink test (default: 1000ms)",
                },
            },
            "required": ["device_id"],
        },
    ),
    ToolDefinition(
        name="get_pin_capabilities",
        description="Get information about a board's pins and their capabilities",
        parameters={
            "type": "object",
            "properties": {"board_id": {"type": "string", "description": "ID of the board"}},
            "required": ["board_id"],
        },
    ),
]


class HardwareToolExecutor:
    """Executes hardware-related tools."""

    def __init__(self, core: "GliderCore"):
        self._core = core

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        method = getattr(self, f"_execute_{tool_name}", None)
        if method is None:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            result = await method(args)
            return {"success": True, "result": result}
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return {"success": False, "error": str(e)}

    def create_action(self, tool_name: str, args: dict[str, Any]) -> AgentAction:
        """Create an action for a tool call."""
        action_types = {
            "list_boards": ActionType.GET_STATE,
            "add_board": ActionType.ADD_BOARD,
            "remove_board": ActionType.REMOVE_BOARD,
            "connect_board": ActionType.ADD_BOARD,
            "disconnect_board": ActionType.REMOVE_BOARD,
            "list_devices": ActionType.GET_STATE,
            "add_device": ActionType.ADD_DEVICE,
            "remove_device": ActionType.REMOVE_DEVICE,
            "configure_device": ActionType.CONFIGURE_DEVICE,
            "scan_ports": ActionType.SCAN_PORTS,
            "test_device": ActionType.TEST_DEVICE,
            "get_pin_capabilities": ActionType.GET_STATE,
        }

        descriptions = {
            "list_boards": "List configured boards",
            "add_board": f"Add {args.get('board_type', 'board')} board '{args.get('name', '')}'",
            "remove_board": f"Remove board {args.get('board_id', '')}",
            "connect_board": f"Connect to board {args.get('board_id', '')}",
            "disconnect_board": f"Disconnect from board {args.get('board_id', '')}",
            "list_devices": "List configured devices",
            "add_device": f"Add {args.get('device_type', 'device')} '{args.get('name', '')}' on pin {args.get('pin', '?')}",
            "remove_device": f"Remove device {args.get('device_id', '')}",
            "configure_device": f"Configure device {args.get('device_id', '')}",
            "scan_ports": "Scan for serial ports",
            "test_device": f"Test device {args.get('device_id', '')}",
            "get_pin_capabilities": f"Get pin info for board {args.get('board_id', '')}",
        }

        return AgentAction(
            action_type=action_types.get(tool_name, ActionType.GET_STATE),
            tool_name=tool_name,
            parameters=args,
            description=descriptions.get(tool_name, tool_name),
        )

    async def _execute_list_boards(self, args: dict[str, Any]) -> dict[str, Any]:
        """List all boards."""
        hw_manager = self._core.hardware_manager

        boards = []
        for board_id, board in hw_manager.boards.items():
            boards.append(
                {
                    "id": board_id,
                    "name": board.name,
                    "type": board.board_type,
                    "port": getattr(board, "port", None),
                    "connected": board.is_connected,
                }
            )

        return {"boards": boards, "count": len(boards)}

    async def _execute_add_board(self, args: dict[str, Any]) -> dict[str, Any]:
        """Add a new board."""
        board_type = args["board_type"]
        name = args["name"]
        port = args.get("port")

        # Normalize board type
        if board_type in ("arduino", "telemetrix"):
            board_type = "telemetrix"
            if not port:
                raise ValueError("Port is required for Arduino boards")
        elif board_type in ("raspberry_pi", "pigpio"):
            board_type = "pigpio"

        hw_manager = self._core.hardware_manager

        board_id = hw_manager.add_board(
            board_type=board_type,
            name=name,
            port=port,
        )

        logger.info(f"Added board: {name} ({board_type}) - ID: {board_id}")

        return {
            "board_id": board_id,
            "name": name,
            "type": board_type,
            "port": port,
        }

    async def _execute_remove_board(self, args: dict[str, Any]) -> dict[str, Any]:
        """Remove a board."""
        board_id = args["board_id"]

        hw_manager = self._core.hardware_manager
        success = hw_manager.remove_board(board_id)

        if success:
            logger.info(f"Removed board: {board_id}")
            return {"removed": board_id}
        else:
            raise ValueError(f"Board not found: {board_id}")

    async def _execute_connect_board(self, args: dict[str, Any]) -> dict[str, Any]:
        """Connect to a board."""
        board_id = args["board_id"]

        hw_manager = self._core.hardware_manager
        board = hw_manager.get_board(board_id)

        if board is None:
            raise ValueError(f"Board not found: {board_id}")

        await board.connect()

        logger.info(f"Connected to board: {board_id}")

        return {"connected": board_id, "success": board.is_connected}

    async def _execute_disconnect_board(self, args: dict[str, Any]) -> dict[str, Any]:
        """Disconnect from a board."""
        board_id = args["board_id"]

        hw_manager = self._core.hardware_manager
        board = hw_manager.get_board(board_id)

        if board is None:
            raise ValueError(f"Board not found: {board_id}")

        await board.disconnect()

        logger.info(f"Disconnected from board: {board_id}")

        return {"disconnected": board_id}

    async def _execute_list_devices(self, args: dict[str, Any]) -> dict[str, Any]:
        """List all devices."""
        hw_manager = self._core.hardware_manager
        board_filter = args.get("board_id")

        devices = []
        for device_id, device in hw_manager.devices.items():
            if board_filter and device.board_id != board_filter:
                continue

            devices.append(
                {
                    "id": device_id,
                    "name": device.name,
                    "type": device.device_type,
                    "board": device.board_id,
                    "pin": device.pin,
                }
            )

        return {"devices": devices, "count": len(devices)}

    async def _execute_add_device(self, args: dict[str, Any]) -> dict[str, Any]:
        """Add a new device."""
        board_id = args["board_id"]
        device_type = args["device_type"]
        name = args["name"]
        pin = args["pin"]
        settings = args.get("settings", {})

        hw_manager = self._core.hardware_manager

        device_id = hw_manager.add_device(
            board_id=board_id,
            device_type=device_type,
            name=name,
            pin=pin,
            **settings,
        )

        logger.info(f"Added device: {name} ({device_type}) on pin {pin} - ID: {device_id}")

        return {
            "device_id": device_id,
            "name": name,
            "type": device_type,
            "board": board_id,
            "pin": pin,
        }

    async def _execute_remove_device(self, args: dict[str, Any]) -> dict[str, Any]:
        """Remove a device."""
        device_id = args["device_id"]

        hw_manager = self._core.hardware_manager
        success = hw_manager.remove_device(device_id)

        if success:
            logger.info(f"Removed device: {device_id}")
            return {"removed": device_id}
        else:
            raise ValueError(f"Device not found: {device_id}")

    # Properties that can be set on devices via the agent tool.
    _CONFIGURABLE_PROPERTIES = frozenset(
        {
            "name",
            "pin",
            "enabled",
            "inverted",
            "pull_up",
            "debounce_ms",
            "frequency",
            "duty_cycle",
            "min_angle",
            "max_angle",
            "initial_value",
            "min_value",
            "max_value",
            "smoothing",
            "sample_rate",
            "threshold",
            "settings",
        }
    )

    async def _execute_configure_device(self, args: dict[str, Any]) -> dict[str, Any]:
        """Configure a device."""
        device_id = args["device_id"]
        settings = args["settings"]

        hw_manager = self._core.hardware_manager
        device = hw_manager.get_device(device_id)

        if device is None:
            raise ValueError(f"Device not found: {device_id}")

        # Apply settings (reject private attributes and non-allowlisted properties)
        applied = {}
        for key, value in settings.items():
            if key.startswith("_"):
                logger.warning(f"Rejected private property '{key}' on device {device_id}")
                continue
            if key not in self._CONFIGURABLE_PROPERTIES:
                logger.warning(f"Rejected unknown property '{key}' on device {device_id}")
                continue
            if hasattr(device, key):
                setattr(device, key, value)
                applied[key] = value

        logger.info(f"Configured device: {device_id} with {applied}")

        return {"configured": device_id, "settings": applied}

    async def _execute_scan_ports(self, args: dict[str, Any]) -> dict[str, Any]:
        """Scan for serial ports."""
        import serial.tools.list_ports

        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(
                {
                    "port": port.device,
                    "description": port.description,
                    "hwid": port.hwid,
                }
            )

        return {"ports": ports, "count": len(ports)}

    async def _execute_test_device(self, args: dict[str, Any]) -> dict[str, Any]:
        """Test a device."""
        device_id = args["device_id"]
        action = args.get("action", "blink")
        duration_ms = args.get("duration_ms", 1000)

        hw_manager = self._core.hardware_manager
        device = hw_manager.get_device(device_id)

        if device is None:
            raise ValueError(f"Device not found: {device_id}")

        result = {}

        if action == "on":
            await device.execute_action("on")
            result = {"action": "on", "success": True}
        elif action == "off":
            await device.execute_action("off")
            result = {"action": "off", "success": True}
        elif action == "toggle":
            await device.execute_action("toggle")
            result = {"action": "toggle", "success": True}
        elif action == "read":
            value = await device.execute_action("read")
            result = {"action": "read", "value": value}
        elif action == "blink":
            import asyncio

            await device.execute_action("on")
            await asyncio.sleep(duration_ms / 1000)
            await device.execute_action("off")
            result = {"action": "blink", "duration_ms": duration_ms, "success": True}

        logger.info(f"Tested device: {device_id} with action {action}")

        return result

    async def _execute_get_pin_capabilities(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get pin capabilities for a board."""
        board_id = args["board_id"]

        hw_manager = self._core.hardware_manager
        board = hw_manager.get_board(board_id)

        if board is None:
            raise ValueError(f"Board not found: {board_id}")

        capabilities = board.get_capabilities() if hasattr(board, "get_capabilities") else {}

        return {
            "board_id": board_id,
            "capabilities": capabilities,
        }
