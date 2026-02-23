"""
Undo/Redo Command Pattern Implementation for GLIDER.

This module contains command classes for undoable operations in the
node graph editor.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from glider.gui.main_window import MainWindow


class Command:
    """Base class for undoable commands."""

    def execute(self) -> None:
        """Execute the command."""
        raise NotImplementedError

    def undo(self) -> None:
        """Undo the command."""
        raise NotImplementedError

    def description(self) -> str:
        """Human-readable description."""
        return "Unknown command"


class CreateNodeCommand(Command):
    """Command for creating a node."""

    def __init__(self, main_window: "MainWindow", node_id: str, node_type: str, x: float, y: float):
        self._main_window = main_window
        self._node_id = node_id
        self._node_type = node_type
        self._x = x
        self._y = y

    def execute(self) -> None:
        """Create the node."""
        # Node is already created when command is recorded
        pass

    def undo(self) -> None:
        """Delete the node."""
        self._main_window._graph_view.remove_node(self._node_id)
        if self._main_window._core.session:
            self._main_window._core.session.remove_node(self._node_id)

    def description(self) -> str:
        return f"Create {self._node_type}"


class DeleteNodeCommand(Command):
    """Command for deleting a node."""

    def __init__(self, main_window: "MainWindow", node_id: str, node_data: dict):
        self._main_window = main_window
        self._node_id = node_id
        self._node_data = node_data  # Saved node state for restoration

    def execute(self) -> None:
        """Delete is already done when command is recorded."""
        pass

    def undo(self) -> None:
        """Restore the node."""
        data = self._node_data
        node_item = self._main_window._graph_view.add_node(
            data["id"], data["node_type"], data["x"], data["y"]
        )
        self._main_window._setup_node_ports(node_item, data["node_type"])
        self._main_window._graph_view._connect_port_signals(node_item)

        if self._main_window._core.session:
            from glider.core.experiment_session import NodeConfig

            node_config = NodeConfig(
                id=data["id"],
                node_type=data["node_type"],
                position=(data["x"], data["y"]),
                state=data.get("state", {}),
                device_id=data.get("device_id"),
                visible_in_runner=data.get("visible_in_runner", False),
            )
            self._main_window._core.session.add_node(node_config)

    def description(self) -> str:
        return f"Delete {self._node_data.get('node_type', 'node')}"


class MoveNodeCommand(Command):
    """Command for moving a node."""

    def __init__(
        self,
        main_window: "MainWindow",
        node_id: str,
        old_x: float,
        old_y: float,
        new_x: float,
        new_y: float,
    ):
        self._main_window = main_window
        self._node_id = node_id
        self._old_x = old_x
        self._old_y = old_y
        self._new_x = new_x
        self._new_y = new_y

    def execute(self) -> None:
        """Move is already done when command is recorded."""
        pass

    def undo(self) -> None:
        """Move node back to original position."""
        node_item = self._main_window._graph_view.nodes.get(self._node_id)
        if node_item:
            node_item.setPos(self._old_x, self._old_y)
        if self._main_window._core.session:
            self._main_window._core.session.update_node_position(
                self._node_id, self._old_x, self._old_y
            )

    def description(self) -> str:
        return "Move node"


class CreateConnectionCommand(Command):
    """Command for creating a connection."""

    def __init__(
        self,
        main_window: "MainWindow",
        conn_id: str,
        from_node: str,
        from_port: int,
        to_node: str,
        to_port: int,
        conn_type: str,
    ):
        self._main_window = main_window
        self._conn_id = conn_id
        self._from_node = from_node
        self._from_port = from_port
        self._to_node = to_node
        self._to_port = to_port
        self._conn_type = conn_type

    def execute(self) -> None:
        """Connection is already created when command is recorded."""
        pass

    def undo(self) -> None:
        """Remove the connection."""
        self._main_window._graph_view.remove_connection(self._conn_id)
        if self._main_window._core.session:
            self._main_window._core.session.remove_connection(self._conn_id)

    def description(self) -> str:
        return "Create connection"


class DeleteConnectionCommand(Command):
    """Command for deleting a connection."""

    def __init__(self, main_window: "MainWindow", conn_id: str, conn_data: dict):
        self._main_window = main_window
        self._conn_id = conn_id
        self._conn_data = conn_data

    def execute(self) -> None:
        """Deletion is already done when command is recorded."""
        pass

    def undo(self) -> None:
        """Restore the connection."""
        data = self._conn_data
        self._main_window._graph_view.add_connection(
            data["id"], data["from_node"], data["from_port"], data["to_node"], data["to_port"]
        )
        if self._main_window._core.session:
            from glider.core.experiment_session import ConnectionConfig

            conn_config = ConnectionConfig(
                id=data["id"],
                from_node=data["from_node"],
                from_output=data["from_port"],
                to_node=data["to_node"],
                to_input=data["to_port"],
                connection_type=data.get("conn_type", "data"),
            )
            self._main_window._core.session.add_connection(conn_config)

    def description(self) -> str:
        return "Delete connection"


class PropertyChangeCommand(Command):
    """Command for changing a node property."""

    def __init__(
        self, main_window: "MainWindow", node_id: str, prop_name: str, old_value, new_value
    ):
        self._main_window = main_window
        self._node_id = node_id
        self._prop_name = prop_name
        self._old_value = old_value
        self._new_value = new_value

    def execute(self) -> None:
        """Property is already changed when command is recorded."""
        pass

    def undo(self) -> None:
        """Restore old property value."""
        if self._main_window._core.session:
            self._main_window._core.session.update_node_state(
                self._node_id, {self._prop_name: self._old_value}
            )

    def description(self) -> str:
        return f"Change {self._prop_name}"


class UndoStack:
    """Manages undo/redo history."""

    def __init__(self, max_size: int = 100):
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []
        self._max_size = max_size

    def push(self, command: Command) -> None:
        """Add a command to the undo stack."""
        self._undo_stack.append(command)
        # Clear redo stack when new command is added
        self._redo_stack.clear()
        # Limit stack size
        if len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)

    def undo(self) -> Command | None:
        """Undo the last command."""
        if not self._undo_stack:
            return None
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)
        return command

    def redo(self) -> Command | None:
        """Redo the last undone command."""
        if not self._redo_stack:
            return None
        command = self._redo_stack.pop()
        # Re-execute by undoing the undo (for property changes, we need to swap values)
        # For most commands, we need to re-apply the action
        self._undo_stack.append(command)
        return command

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def clear(self) -> None:
        """Clear both stacks."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def undo_description(self) -> str:
        """Get description of next undo action."""
        if self._undo_stack:
            return self._undo_stack[-1].description()
        return ""

    def redo_description(self) -> str:
        """Get description of next redo action."""
        if self._redo_stack:
            return self._redo_stack[-1].description()
        return ""
