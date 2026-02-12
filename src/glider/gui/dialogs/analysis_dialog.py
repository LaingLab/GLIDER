"""
Analysis Dialog - AI-powered CSV data analysis interface.

Provides a dialog for analyzing tracking CSV data using natural language queries.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from glider.agent.analysis import AnalysisController

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
        self._content_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self._content_label.setMinimumWidth(100)
        layout.addWidget(self._content_label)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Styling based on role
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


class AnalysisDialog(QDialog):
    """
    Dialog for AI-powered CSV data analysis.

    Allows users to load tracking CSV files and ask natural language
    questions about the data.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._controller: Optional[AnalysisController] = None
        self._current_response_bubble: Optional[MessageBubble] = None
        self._loaded_file_ids: dict[str, str] = {}  # file_path -> file_id
        self._setup_ui()
        self._initialize_controller()

    def _setup_ui(self) -> None:
        """Setup the dialog UI."""
        self.setWindowTitle("CSV Data Analysis")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        # Files section
        files_widget = self._create_files_section()
        splitter.addWidget(files_widget)

        # Chat section
        chat_widget = self._create_chat_section()
        splitter.addWidget(chat_widget)

        # Set splitter sizes (files smaller, chat bigger)
        splitter.setSizes([150, 450])

        layout.addWidget(splitter, 1)

        # Input section
        input_frame = self._create_input_section()
        layout.addWidget(input_frame)

    def _create_header(self) -> QFrame:
        """Create the header section."""
        header = QFrame()
        header.setStyleSheet("background-color: #2d2d2d; border-bottom: 1px solid #444;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("CSV Data Analysis")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Status indicator
        self._status_indicator = QLabel("●")
        self._status_indicator.setStyleSheet("color: #888; font-size: 14px;")
        header_layout.addWidget(self._status_indicator)

        self._status_label = QLabel("Initializing...")
        self._status_label.setStyleSheet("color: #888; font-size: 12px;")
        header_layout.addWidget(self._status_label)

        header_layout.addSpacing(16)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)

        return header

    def _create_files_section(self) -> QWidget:
        """Create the files section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 8, 12, 8)

        # Files group
        files_group = QGroupBox("Loaded Files")
        files_layout = QVBoxLayout(files_group)

        # File list
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(100)
        self._file_list.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QPushButton {
                padding: 2px 4px;
            }
            QListWidget::item:selected {
                background-color: #3c3c3c;
            }
        """)
        files_layout.addWidget(self._file_list)

        # File buttons
        btn_layout = QHBoxLayout()

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_file)
        btn_layout.addWidget(browse_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_file)
        btn_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_files)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()
        files_layout.addLayout(btn_layout)

        layout.addWidget(files_group)

        return widget

    def _create_chat_section(self) -> QWidget:
        """Create the chat section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 0, 12, 8)

        # Chat group
        chat_group = QGroupBox("Chat")
        chat_layout = QVBoxLayout(chat_group)
        chat_layout.setContentsMargins(0, 8, 0, 0)

        # Scrollable chat area
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
        chat_layout.addWidget(scroll_area)

        # Quick prompts
        quick_frame = QFrame()
        quick_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
        quick_layout = QHBoxLayout(quick_frame)
        quick_layout.setContentsMargins(8, 4, 8, 4)

        prompts = [
            ("Summarize", "Summarize this tracking data"),
            ("Zones", "How much time was spent in each zone?"),
            ("Distance", "What was the total distance traveled?"),
            ("Compare", "Compare the first half to the second half of the session"),
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
                QPushButton:disabled {
                    color: #666;
                }
            """)
            btn.clicked.connect(lambda checked, p=prompt: self._send_message(p))
            quick_layout.addWidget(btn)

        quick_layout.addStretch()
        chat_layout.addWidget(quick_frame)

        layout.addWidget(chat_group)

        return widget

    def _create_input_section(self) -> QFrame:
        """Create the input section."""
        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #2d2d2d; border-top: 1px solid #444;")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(12, 8, 12, 8)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask a question about your data...")
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
            QLineEdit:disabled {
                background-color: #2a2a2a;
            }
        """)
        self._input.returnPressed.connect(lambda: self._send_message(self._input.text()))
        input_layout.addWidget(self._input)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(70)
        self._send_btn.setStyleSheet("""
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
        self._send_btn.clicked.connect(lambda: self._send_message(self._input.text()))
        input_layout.addWidget(self._send_btn)

        return input_frame

    def _initialize_controller(self) -> None:
        """Initialize the analysis controller."""
        self._controller = AnalysisController()

        # Check connection
        asyncio.create_task(self._check_connection())

    async def _check_connection(self) -> None:
        """Check connection to LLM backend."""
        if not self._controller:
            return

        success, message = await self._controller.check_connection()

        if success:
            self._set_status("Ready", connected=True)
            self._add_message(
                "assistant",
                "Hello! I can help you analyze tracking data from your experiments.\n\n"
                "To get started:\n"
                "1. Click 'Browse...' to load a tracking CSV file\n"
                "2. Ask questions about your data\n\n"
                "Example questions:\n"
                "• What was the total distance traveled?\n"
                "• How much time was spent in each zone?\n"
                "• Compare the first 5 minutes to the last 5 minutes",
            )
        else:
            self._set_status("Disconnected", connected=False)
            self._add_message(
                "assistant",
                f"Could not connect to the LLM backend.\n\n"
                f"Error: {message}\n\n"
                "Make sure Ollama is running:\n"
                "  ollama serve\n\n"
                "And that you have a model installed:\n"
                "  ollama pull llama3.2",
            )

    def _set_status(self, status: str, connected: bool = True) -> None:
        """Update status display."""
        self._status_label.setText(status)
        if connected:
            self._status_indicator.setStyleSheet("color: #4caf50; font-size: 14px;")
        else:
            self._status_indicator.setStyleSheet("color: #f44336; font-size: 14px;")

    def _on_browse_file(self) -> None:
        """Handle browse button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Tracking CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )

        if file_path:
            asyncio.create_task(self._load_file(file_path))

    async def _load_file(self, file_path: str) -> None:
        """Load a CSV file."""
        if not self._controller:
            return

        self._set_status("Loading file...")

        result = await self._controller.load_file(file_path)

        if "error" in result:
            QMessageBox.warning(
                self,
                "Load Error",
                f"Could not load file:\n{result['error']}",
            )
            self._set_status("Ready", connected=True)
            return

        # Store file ID mapping
        self._loaded_file_ids[file_path] = result.get("file_id", "")

        # Add to list
        file_name = result.get("file_name", Path(file_path).name)
        row_count = result.get("row_count", 0)
        duration = result.get("duration_seconds", 0)
        subject = result.get("subject_id", "Unknown")

        item_text = f"{file_name}\n  {row_count} rows, {duration:.1f}s"
        if subject and subject != "Unknown":
            item_text += f", Subject: {subject}"

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        self._file_list.addItem(item)

        self._set_status("Ready", connected=True)

        # Notify in chat
        self._add_message(
            "assistant",
            f"Loaded **{file_name}**\n"
            f"• {row_count} data rows\n"
            f"• Duration: {duration:.1f} seconds\n"
            f"• Subject: {subject}\n\n"
            "You can now ask questions about this data.",
        )

    def _on_remove_file(self) -> None:
        """Handle remove button click."""
        current_item = self._file_list.currentItem()
        if not current_item:
            return

        file_path = current_item.data(Qt.ItemDataRole.UserRole)
        if file_path in self._loaded_file_ids:
            del self._loaded_file_ids[file_path]

        self._file_list.takeItem(self._file_list.row(current_item))

    def _on_clear_files(self) -> None:
        """Handle clear all button click."""
        if not self._controller:
            return

        self._file_list.clear()
        self._loaded_file_ids.clear()
        self._controller.clear_data()
        self._add_message("assistant", "All files cleared.")

    def _send_message(self, text: str) -> None:
        """Send a message to the assistant."""
        text = text.strip()
        if not text:
            return

        if not self._controller:
            return

        # Check if we have files loaded
        if not self._loaded_file_ids:
            self._add_message(
                "assistant",
                "Please load a CSV file first using the 'Browse...' button.",
            )
            return

        self._input.clear()
        self._add_message("user", text)

        # Process message
        self._set_status("Thinking...")
        self._input.setEnabled(False)
        self._send_btn.setEnabled(False)

        # Create response bubble for streaming
        self._current_response_bubble = self._add_message("assistant", "...")

        asyncio.create_task(self._process_message(text))

    async def _process_message(self, text: str) -> None:
        """Process a message through the controller."""
        if not self._controller:
            return

        try:
            accumulated = ""

            async for response in self._controller.process_message(text):
                if response.content:
                    accumulated += response.content
                    if self._current_response_bubble:
                        self._current_response_bubble.update_content(accumulated)
                        self._scroll_to_bottom()

                if response.tool_used:
                    # Could show tool usage indicator
                    pass

                if response.error:
                    if self._current_response_bubble:
                        self._current_response_bubble.update_content(f"Error: {response.error}")

                if response.is_complete:
                    break

        except Exception as e:
            logger.exception("Error processing message")
            if self._current_response_bubble:
                self._current_response_bubble.update_content(f"Sorry, an error occurred: {str(e)}")

        finally:
            self._set_status("Ready", connected=True)
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

    def closeEvent(self, event) -> None:
        """Handle dialog close."""
        if self._controller:
            asyncio.create_task(self._controller.shutdown())
        super().closeEvent(event)
