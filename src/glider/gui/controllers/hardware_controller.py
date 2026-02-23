"""
Hardware Tree Controller for GLIDER.

This controller encapsulates the hardware tree widget functionality,
extracted from MainWindow to improve modularity.
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


class HardwareTreeController(QWidget):
    """
    Controller for the hardware tree widget.

    Manages the display and interaction with boards and devices
    in the hardware tree view.

    Signals:
        hardware_changed: Emitted when hardware configuration changes
        device_selected: Emitted when a device is selected (device_id)
    """

    hardware_changed = pyqtSignal()
    device_selected = pyqtSignal(str)
    _connection_changed_signal = pyqtSignal(str, object)

    def __init__(self, core: "GliderCore", run_async: Callable, parent: Optional[QWidget] = None):
        """
        Initialize the hardware tree controller.

        Args:
            core: GliderCore instance
            run_async: Function to run async coroutines
            parent: Parent widget
        """
        super().__init__(parent)
        self._core = core
        self._run_async = run_async

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Hardware tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Type", "Status"])
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree)

        # Buttons
        btn_layout = QHBoxLayout()
        add_board_btn = QPushButton("+ Board")
        add_board_btn.clicked.connect(self._on_add_board)
        btn_layout.addWidget(add_board_btn)

        add_device_btn = QPushButton("+ Device")
        add_device_btn.clicked.connect(self._on_add_device)
        btn_layout.addWidget(add_device_btn)

        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Listen for hardware connection changes
        self._core.hardware_manager.on_connection_change(self._on_connection_change)
        self._connection_changed_signal.connect(self._handle_connection_change)

    @property
    def tree(self) -> QTreeWidget:
        """Get the tree widget."""
        return self._tree

    def refresh(self) -> None:
        """Refresh the hardware tree display."""
        self._tree.clear()

        # Add boards
        for board_id, board in self._core.hardware_manager.boards.items():
            board_item = QTreeWidgetItem(
                [
                    board_id,
                    getattr(board, "name", type(board).__name__),
                    board.state.name if hasattr(board, "state") else "Unknown",
                ]
            )
            board_item.setData(0, Qt.ItemDataRole.UserRole, ("board", board_id))

            # Add devices under this board
            for device_id, device in self._core.hardware_manager.devices.items():
                if hasattr(device, "board") and device.board is board:
                    pins = getattr(device, "_pins", [])
                    pin_str = f"Pin {pins[0]}" if pins else ""
                    device_item = QTreeWidgetItem(
                        [
                            getattr(device, "name", device_id),
                            f"{getattr(device, 'device_type', 'unknown')} ({pin_str})",
                            (
                                "Ready"
                                if getattr(device, "_initialized", False)
                                else "Not initialized"
                            ),
                        ]
                    )
                    device_item.setData(0, Qt.ItemDataRole.UserRole, ("device", device_id))
                    board_item.addChild(device_item)

            self._tree.addTopLevelItem(board_item)
            board_item.setExpanded(True)

        self._tree.resizeColumnToContents(0)
        self._tree.resizeColumnToContents(1)

        # Emit signal for other components to refresh
        self.hardware_changed.emit()

    def _on_connection_change(self, board_id: str, state) -> None:
        """Handle board connection state changes (may be called from non-GUI thread)."""
        self._connection_changed_signal.emit(board_id, state)

    def _handle_connection_change(self, board_id: str, state) -> None:
        """Handle board connection state changes (main thread)."""
        self.refresh()

    def _on_selection_changed(self) -> None:
        """Handle tree selection changes."""
        items = self._tree.selectedItems()
        if not items:
            return

        data = items[0].data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "device":
            self.device_selected.emit(data[1])

    def _on_context_menu(self, position) -> None:
        """Show context menu for hardware tree."""
        item = self._tree.itemAt(position)
        if item is None:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return

        item_type, item_id = data
        menu = QMenu(self)

        if item_type == "board":
            connect_action = menu.addAction("Connect")
            connect_action.triggered.connect(lambda: self._connect_board(item_id))

            disconnect_action = menu.addAction("Disconnect")
            disconnect_action.triggered.connect(lambda: self._disconnect_board(item_id))

            menu.addSeparator()

            remove_action = menu.addAction("Remove Board")
            remove_action.triggered.connect(lambda: self._remove_board(item_id))

        elif item_type == "device":
            remove_action = menu.addAction("Remove Device")
            remove_action.triggered.connect(lambda: self._remove_device(item_id))

        menu.exec(self._tree.viewport().mapToGlobal(position))

    def _connect_board(self, board_id: str) -> None:
        """Connect to a specific board and initialize its devices."""

        async def connect():
            try:
                success = await self._core.hardware_manager.connect_board(board_id)
                if success:
                    # Initialize devices for this board
                    for device_id, device in self._core.hardware_manager.devices.items():
                        if hasattr(device, "board") and device.board is not None:
                            if device.board.id == board_id:
                                try:
                                    await self._core.hardware_manager.initialize_device(device_id)
                                except Exception as e:
                                    logger.warning(f"Failed to initialize device {device_id}: {e}")
                else:
                    QMessageBox.warning(
                        self, "Connection Failed", f"Could not connect to {board_id}"
                    )
                self.refresh()
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", str(e))

        self._run_async(connect())

    def _disconnect_board(self, board_id: str) -> None:
        """Disconnect from a specific board."""

        async def disconnect():
            try:
                await self._core.hardware_manager.disconnect_board(board_id)
                self.refresh()
            except Exception as e:
                QMessageBox.critical(self, "Disconnect Error", str(e))

        self._run_async(disconnect())

    def _remove_board(self, board_id: str) -> None:
        """Remove a board."""
        reply = QMessageBox.question(
            self,
            "Remove Board",
            f"Remove board '{board_id}' and all its devices?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:

            async def remove():
                await self._core.hardware_manager.remove_board(board_id)
                if self._core.session:
                    self._core.session.remove_board(board_id)
                self.refresh()

            self._run_async(remove())

    def _remove_device(self, device_id: str) -> None:
        """Remove a device."""
        reply = QMessageBox.question(
            self,
            "Remove Device",
            f"Remove device '{device_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:

            async def remove():
                await self._core.hardware_manager.remove_device(device_id)
                if self._core.session:
                    self._core.session.remove_device(device_id)
                self.refresh()

            self._run_async(remove())

    def _on_add_board(self) -> None:
        """Handle add board button click."""
        # This will be connected to MainWindow's add board dialog
        # For now, emit a signal or call parent method
        parent = self.parent()
        if hasattr(parent, "_on_add_board"):
            parent._on_add_board()

    def _on_add_device(self) -> None:
        """Handle add device button click."""
        # This will be connected to MainWindow's add device dialog
        parent = self.parent()
        if hasattr(parent, "_on_add_device"):
            parent._on_add_device()

    def connect_all(self) -> None:
        """Connect to all hardware."""

        async def connect():
            try:
                await self._core.setup_hardware()
                results = await self._core.connect_hardware()
                self.refresh()
                failed = [k for k, v in results.items() if not v]
                if failed:
                    QMessageBox.warning(
                        self, "Connection Warning", f"Failed to connect: {', '.join(failed)}"
                    )
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", str(e))

        self._run_async(connect())

    def disconnect_all(self) -> None:
        """Disconnect all hardware."""
        self._run_async(self._core.hardware_manager.disconnect_all())
