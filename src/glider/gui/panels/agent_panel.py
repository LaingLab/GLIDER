"""
Agent Panel

Chat interface for AI agent interaction.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.agent.agent_controller import AgentController

logger = logging.getLogger(__name__)


class MessageBubble(QFrame):
    """A single message bubble in the chat."""

    def __init__(self, role: str, content: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._role = role
        self._content = content
        self._content_label: Optional[QLabel] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Role label
        role_label = QLabel(self._role.capitalize())
        role_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #888;")
        layout.addWidget(role_label)

        # Content - use QLabel with proper wrapping
        self._content_label = QLabel(self._content)
        self._content_label.setWordWrap(True)
        self._content_label.setTextFormat(Qt.TextFormat.PlainText)
        self._content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Use Ignored horizontal policy to force word wrap
        self._content_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self._content_label.setMinimumWidth(100)
        layout.addWidget(self._content_label)

        # Allow the bubble itself to shrink
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Styling
        if self._role == "user":
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #2a5298;
                    border-radius: 8px;
                    margin-left: 40px;
                }
                QLabel {
                    color: white;
                }
            """)
        else:
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #3c3c3c;
                    border-radius: 8px;
                    margin-right: 40px;
                }
                QLabel {
                    color: #e0e0e0;
                }
            """)

    def update_content(self, content: str) -> None:
        """Update the message content."""
        self._content = content
        if self._content_label:
            self._content_label.setText(content)


class ActionConfirmWidget(QFrame):
    """Widget for confirming/rejecting agent actions."""

    confirmed = pyqtSignal()
    rejected = pyqtSignal()

    def __init__(self, actions: list, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._actions = actions
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("Agent wants to perform actions:")
        header.setStyleSheet("font-weight: bold; color: #ffa726;")
        layout.addWidget(header)

        # Action list
        for action in self._actions:
            action_label = QLabel(f"  - {action.description}")
            action_label.setStyleSheet("color: #e0e0e0; margin-left: 8px;")
            layout.addWidget(action_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        confirm_btn = QPushButton("Execute")
        confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        confirm_btn.clicked.connect(self.confirmed.emit)
        button_layout.addWidget(confirm_btn)

        reject_btn = QPushButton("Cancel")
        reject_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        reject_btn.clicked.connect(self.rejected.emit)
        button_layout.addWidget(reject_btn)

        layout.addLayout(button_layout)

        # Frame styling
        self.setStyleSheet("""
            ActionConfirmWidget {
                background-color: #2d2d2d;
                border: 1px solid #ffa726;
                border-radius: 8px;
            }
        """)


class ChatInput(QLineEdit):
    """Custom input field with Enter to send."""

    submitted = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setPlaceholderText("Type a message...")
        self.returnPressed.connect(self._on_submit)

    def _on_submit(self) -> None:
        text = self.text().strip()
        if text:
            self.submitted.emit(text)
            self.clear()


class AgentPanel(QWidget):
    """
    Chat interface for AI agent interaction.

    Provides:
    - Scrollable chat history
    - Message input with send button
    - Action confirmation dialogs
    - Status indicators
    """

    message_sent = pyqtSignal(str)
    settings_requested = pyqtSignal()

    def __init__(
        self, controller: Optional["AgentController"] = None, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._controller = controller
        self._current_response_bubble: Optional[MessageBubble] = None
        self._action_widget: Optional[ActionConfirmWidget] = None
        self._setup_ui()

        if controller:
            self._connect_controller()

    def set_controller(self, controller: "AgentController") -> None:
        """Set the agent controller."""
        self._controller = controller
        self._connect_controller()

    def _connect_controller(self) -> None:
        """Connect controller callbacks."""
        pass  # Callbacks connected in main window

    def _setup_ui(self) -> None:
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet("background-color: #2d2d2d; border-bottom: 1px solid #444;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        title = QLabel("AI Assistant")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Status indicator
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        header_layout.addWidget(self._status_label)

        header_layout.addSpacing(8)

        # Settings button
        settings_btn = QPushButton("Settings")
        settings_btn.setMinimumWidth(70)
        settings_btn.clicked.connect(self.settings_requested.emit)
        header_layout.addWidget(settings_btn)

        layout.addWidget(header)

        # Chat area (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")

        self._chat_container = QWidget()
        self._chat_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._chat_layout = QVBoxLayout(self._chat_container)
        self._chat_layout.setContentsMargins(8, 8, 8, 8)
        self._chat_layout.setSpacing(8)
        self._chat_layout.addStretch()

        scroll_area.setWidget(self._chat_container)
        self._scroll_area = scroll_area
        layout.addWidget(scroll_area, 1)

        # Quick prompts
        quick_prompts = QFrame()
        quick_prompts.setStyleSheet("background-color: #2d2d2d;")
        quick_layout = QHBoxLayout(quick_prompts)
        quick_layout.setContentsMargins(8, 4, 8, 4)

        prompts = [
            ("Blink LED", "Create an experiment that blinks an LED"),
            ("Add Arduino", "Help me set up an Arduino board"),
            ("Explain", "What types of nodes are available?"),
        ]

        for label, prompt in prompts:
            btn = QPushButton(label)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    border: 1px solid #555;
                    border-radius: 12px;
                    padding: 4px 12px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #4c4c4c;
                }
            """)
            btn.clicked.connect(lambda checked, p=prompt: self._send_message(p))
            quick_layout.addWidget(btn)

        quick_layout.addStretch()
        layout.addWidget(quick_prompts)

        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #2d2d2d; border-top: 1px solid #444;")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 8, 8, 8)

        self._input = ChatInput()
        self._input.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                color: white;
            }
            QLineEdit:focus {
                border-color: #2196f3;
            }
        """)
        self._input.submitted.connect(self._send_message)
        input_layout.addWidget(self._input)

        send_btn = QPushButton("Send")
        send_btn.setFixedWidth(60)
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        send_btn.clicked.connect(lambda: self._send_message(self._input.text()))
        self._send_btn = send_btn
        input_layout.addWidget(send_btn)

        layout.addWidget(input_frame)

        # Add welcome message
        self._add_message(
            "assistant",
            "Hello! I'm your AI assistant. I can help you:\n\n"
            "- Create experiments with visual flow graphs\n"
            "- Configure Arduino or Raspberry Pi hardware\n"
            "- Explain concepts and troubleshoot issues\n\n"
            "What would you like to do?",
        )

    def _send_message(self, text: str) -> None:
        """Send a message."""
        text = text.strip()
        if not text:
            return

        self._input.clear()
        self._add_message("user", text)
        self.message_sent.emit(text)

        # If we have a controller, process the message
        if self._controller:
            self._process_message(text)

    def _process_message(self, text: str) -> None:
        """Process message through the controller."""
        self._set_status("Thinking...")
        self._input.setEnabled(False)
        self._send_btn.setEnabled(False)

        # Create response bubble for streaming
        self._current_response_bubble = self._add_message("assistant", "...")

        # Process asynchronously
        try:
            asyncio.create_task(self._async_process(text))
        except RuntimeError:
            logger.debug("No event loop available for async task")

    async def _async_process(self, text: str) -> None:
        """Async message processing."""
        try:
            async for response in self._controller.process_message(text):
                # Update the response bubble
                if self._current_response_bubble and response.content:
                    self._current_response_bubble.update_content(response.content)
                    self._scroll_to_bottom()

                if response.is_complete:
                    # Show action confirmation if needed
                    if response.pending_actions:
                        self._show_action_confirmation(response.pending_actions)

                    if response.error:
                        self._add_message("assistant", f"Error: {response.error}")

        except Exception as e:
            logger.exception("Error processing message")
            self._add_message("assistant", f"Sorry, an error occurred: {str(e)}")

        finally:
            self._set_status("Ready")
            self._input.setEnabled(True)
            self._send_btn.setEnabled(True)
            self._input.setFocus()
            self._current_response_bubble = None

    def _add_message(self, role: str, content: str) -> MessageBubble:
        """Add a message to the chat."""
        bubble = MessageBubble(role, content)

        # Insert before the stretch
        count = self._chat_layout.count()
        self._chat_layout.insertWidget(count - 1, bubble)

        # Scroll to bottom
        QTimer.singleShot(50, self._scroll_to_bottom)

        return bubble

    def _scroll_to_bottom(self) -> None:
        """Scroll chat to bottom."""
        scrollbar = self._scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _show_action_confirmation(self, actions: list) -> None:
        """Show action confirmation widget."""
        if self._action_widget:
            self._action_widget.deleteLater()

        self._action_widget = ActionConfirmWidget(actions)
        self._action_widget.confirmed.connect(self._on_confirm_actions)
        self._action_widget.rejected.connect(self._on_reject_actions)

        count = self._chat_layout.count()
        self._chat_layout.insertWidget(count - 1, self._action_widget)
        self._scroll_to_bottom()

    def _on_confirm_actions(self) -> None:
        """Handle action confirmation."""
        if self._action_widget:
            self._action_widget.deleteLater()
            self._action_widget = None

        if self._controller:
            try:
                asyncio.create_task(self._execute_confirmed())
            except RuntimeError:
                logger.debug("No event loop available for async task")

    async def _execute_confirmed(self) -> None:
        """Execute confirmed actions."""
        self._set_status("Executing...")

        try:
            response = await self._controller.confirm_actions()
            self._add_message("assistant", response.content)
        except Exception as e:
            self._add_message("assistant", f"Error executing actions: {str(e)}")
        finally:
            self._set_status("Ready")

    def _on_reject_actions(self) -> None:
        """Handle action rejection."""
        if self._action_widget:
            self._action_widget.deleteLater()
            self._action_widget = None

        if self._controller:
            try:
                asyncio.create_task(self._execute_rejected())
            except RuntimeError:
                logger.debug("No event loop available for async task")

    async def _execute_rejected(self) -> None:
        """Handle rejected actions."""
        try:
            response = await self._controller.reject_actions()
            self._add_message("assistant", response.content)
        except Exception:
            logger.exception("Error rejecting actions")

    def _set_status(self, status: str) -> None:
        """Update status label."""
        self._status_label.setText(status)

    def clear_chat(self) -> None:
        """Clear chat history."""
        # Remove all message bubbles
        while self._chat_layout.count() > 1:  # Keep the stretch
            item = self._chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._controller:
            self._controller.clear_conversation()

        # Re-add welcome message
        self._add_message("assistant", "Chat cleared. How can I help you?")
