"""
Agent Settings Dialog

Configure AI agent settings including LLM provider and behavior.
"""

import asyncio
import logging

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from glider.agent.config import DEFAULT_MODELS, AgentConfig, LLMProvider

logger = logging.getLogger(__name__)


class AgentSettingsDialog(QDialog):
    """Dialog for configuring AI agent settings."""

    def __init__(self, config: AgentConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        self._available_models: list[str] = []
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        """Setup the dialog UI."""
        self.setWindowTitle("AI Agent Settings")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()

        # Provider tab
        provider_tab = self._create_provider_tab()
        tabs.addTab(provider_tab, "Provider")

        # Behavior tab
        behavior_tab = self._create_behavior_tab()
        tabs.addTab(behavior_tab, "Behavior")

        # Context tab
        context_tab = self._create_context_tab()
        tabs.addTab(context_tab, "Context")

        layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self._test_connection)
        button_layout.addWidget(test_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save_and_accept)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

    def _create_provider_tab(self) -> QWidget:
        """Create the provider settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Provider selection
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QFormLayout(provider_group)

        self._provider_combo = QComboBox()
        self._provider_combo.addItems([p.value.title() for p in LLMProvider])
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_layout.addRow("Provider:", self._provider_combo)

        self._base_url_edit = QLineEdit()
        self._base_url_edit.setPlaceholderText("http://localhost:11434")
        provider_layout.addRow("Base URL:", self._base_url_edit)

        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("Not required for Ollama")
        provider_layout.addRow("API Key:", self._api_key_edit)

        layout.addWidget(provider_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        model_row = QHBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setMinimumWidth(200)
        model_row.addWidget(self._model_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(refresh_btn)

        model_layout.addRow("Model:", model_row)

        layout.addWidget(model_group)

        # Generation settings
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QFormLayout(gen_group)

        self._temperature_spin = QDoubleSpinBox()
        self._temperature_spin.setRange(0.0, 2.0)
        self._temperature_spin.setSingleStep(0.1)
        self._temperature_spin.setDecimals(1)
        gen_layout.addRow("Temperature:", self._temperature_spin)

        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(256, 16384)
        self._max_tokens_spin.setSingleStep(256)
        gen_layout.addRow("Max Tokens:", self._max_tokens_spin)

        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(10, 300)
        self._timeout_spin.setSuffix(" seconds")
        gen_layout.addRow("Timeout:", self._timeout_spin)

        layout.addWidget(gen_group)

        layout.addStretch()

        return widget

    def _create_behavior_tab(self) -> QWidget:
        """Create the behavior settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Confirmation settings
        confirm_group = QGroupBox("Action Confirmation")
        confirm_layout = QVBoxLayout(confirm_group)

        self._require_confirm_check = QCheckBox("Require confirmation for actions")
        self._require_confirm_check.setToolTip(
            "When enabled, the agent will ask for confirmation before modifying the experiment"
        )
        confirm_layout.addWidget(self._require_confirm_check)

        self._auto_safe_check = QCheckBox("Auto-execute safe actions")
        self._auto_safe_check.setToolTip(
            "Automatically execute read-only actions like explanations"
        )
        confirm_layout.addWidget(self._auto_safe_check)

        self._max_actions_spin = QSpinBox()
        self._max_actions_spin.setRange(1, 50)
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max actions per message:"))
        max_row.addWidget(self._max_actions_spin)
        max_row.addStretch()
        confirm_layout.addLayout(max_row)

        layout.addWidget(confirm_group)

        # Response settings
        response_group = QGroupBox("Responses")
        response_layout = QVBoxLayout(response_group)

        self._stream_check = QCheckBox("Stream responses")
        self._stream_check.setToolTip("Show responses as they are generated")
        response_layout.addWidget(self._stream_check)

        self._show_thinking_check = QCheckBox("Show thinking process")
        self._show_thinking_check.setToolTip("Display the agent's reasoning (if available)")
        response_layout.addWidget(self._show_thinking_check)

        layout.addWidget(response_group)

        layout.addStretch()

        return widget

    def _create_context_tab(self) -> QWidget:
        """Create the context settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Context inclusion
        context_group = QGroupBox("Include in Context")
        context_layout = QVBoxLayout(context_group)

        self._include_flow_check = QCheckBox("Current flow state (nodes and connections)")
        context_layout.addWidget(self._include_flow_check)

        self._include_hardware_check = QCheckBox("Hardware configuration (boards and devices)")
        context_layout.addWidget(self._include_hardware_check)

        self._include_errors_check = QCheckBox("Recent errors")
        context_layout.addWidget(self._include_errors_check)

        self._max_nodes_spin = QSpinBox()
        self._max_nodes_spin.setRange(10, 200)
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max nodes to include:"))
        max_row.addWidget(self._max_nodes_spin)
        max_row.addStretch()
        context_layout.addLayout(max_row)

        layout.addWidget(context_group)

        # Custom instructions
        custom_group = QGroupBox("Custom Instructions")
        custom_layout = QVBoxLayout(custom_group)

        custom_label = QLabel("Additional instructions for the agent:")
        custom_layout.addWidget(custom_label)

        self._custom_instructions = QTextEdit()
        self._custom_instructions.setPlaceholderText(
            "E.g., 'Always use pin 13 for LEDs' or 'Prefer simple loops over complex logic'"
        )
        self._custom_instructions.setMaximumHeight(100)
        custom_layout.addWidget(self._custom_instructions)

        layout.addWidget(custom_group)

        layout.addStretch()

        return widget

    def _load_settings(self) -> None:
        """Load settings from config."""
        # Provider
        provider_index = list(LLMProvider).index(self._config.provider)
        self._provider_combo.setCurrentIndex(provider_index)
        self._base_url_edit.setText(self._config.base_url)
        if self._config.api_key:
            self._api_key_edit.setText(self._config.api_key)

        # Model
        self._model_combo.setCurrentText(self._config.model)

        # Generation
        self._temperature_spin.setValue(self._config.temperature)
        self._max_tokens_spin.setValue(self._config.max_tokens)
        self._timeout_spin.setValue(int(self._config.timeout))

        # Behavior
        self._require_confirm_check.setChecked(self._config.require_confirmation)
        self._auto_safe_check.setChecked(self._config.auto_execute_safe)
        self._max_actions_spin.setValue(self._config.max_actions_per_message)
        self._stream_check.setChecked(self._config.stream_responses)
        self._show_thinking_check.setChecked(self._config.show_thinking)

        # Context
        self._include_flow_check.setChecked(self._config.include_flow_state)
        self._include_hardware_check.setChecked(self._config.include_hardware_state)
        self._include_errors_check.setChecked(self._config.include_recent_errors)
        self._max_nodes_spin.setValue(self._config.max_context_nodes)
        self._custom_instructions.setPlainText(self._config.custom_instructions)

        self._on_provider_changed()

    def _on_provider_changed(self) -> None:
        """Handle provider selection change."""
        provider = list(LLMProvider)[self._provider_combo.currentIndex()]

        # Update UI based on provider
        if provider == LLMProvider.OLLAMA:
            self._base_url_edit.setEnabled(True)
            self._api_key_edit.setEnabled(False)
            self._api_key_edit.setPlaceholderText("Not required for Ollama")
            if not self._base_url_edit.text():
                self._base_url_edit.setText("http://localhost:11434")
        else:
            self._base_url_edit.setEnabled(provider == LLMProvider.OPENAI)
            self._api_key_edit.setEnabled(True)
            self._api_key_edit.setPlaceholderText("Enter API key")

        # Set default model
        if self._model_combo.currentText() in DEFAULT_MODELS.values():
            self._model_combo.setCurrentText(DEFAULT_MODELS.get(provider, ""))

    def _refresh_models(self) -> None:
        """Refresh available models from provider."""
        try:
            asyncio.create_task(self._async_refresh_models())
        except RuntimeError:
            logger.debug("No event loop available for async task")

    async def _async_refresh_models(self) -> None:
        """Async model refresh."""
        from glider.agent.llm_backend import LLMBackend

        config = self._get_current_config()
        backend = LLMBackend(config)

        try:
            models = await backend.list_models()

            if models:
                current = self._model_combo.currentText()
                self._model_combo.clear()
                self._model_combo.addItems(models)
                if current in models:
                    self._model_combo.setCurrentText(current)

                QMessageBox.information(self, "Models Refreshed", f"Found {len(models)} model(s).")
            else:
                QMessageBox.warning(self, "No Models", "No models found. Is the server running?")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch models: {str(e)}")
        finally:
            await backend.close()

    def _test_connection(self) -> None:
        """Test connection to the LLM provider."""
        try:
            asyncio.create_task(self._async_test_connection())
        except RuntimeError:
            logger.debug("No event loop available for async task")

    async def _async_test_connection(self) -> None:
        """Async connection test."""
        from glider.agent.llm_backend import LLMBackend

        config = self._get_current_config()
        backend = LLMBackend(config)

        try:
            connected = await backend.check_connection()

            if connected:
                QMessageBox.information(
                    self,
                    "Connection Successful",
                    f"Successfully connected to {config.provider.value.title()}!",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Connection Failed",
                    "Could not connect to the server. Check your settings.",
                )

        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Error: {str(e)}")
        finally:
            await backend.close()

    def _get_current_config(self) -> AgentConfig:
        """Build config from current UI state."""
        provider = list(LLMProvider)[self._provider_combo.currentIndex()]

        return AgentConfig(
            provider=provider,
            model=self._model_combo.currentText(),
            api_key=self._api_key_edit.text() or None,
            base_url=self._base_url_edit.text(),
            temperature=self._temperature_spin.value(),
            max_tokens=self._max_tokens_spin.value(),
            timeout=float(self._timeout_spin.value()),
            require_confirmation=self._require_confirm_check.isChecked(),
            auto_execute_safe=self._auto_safe_check.isChecked(),
            max_actions_per_message=self._max_actions_spin.value(),
            stream_responses=self._stream_check.isChecked(),
            show_thinking=self._show_thinking_check.isChecked(),
            include_flow_state=self._include_flow_check.isChecked(),
            include_hardware_state=self._include_hardware_check.isChecked(),
            include_recent_errors=self._include_errors_check.isChecked(),
            max_context_nodes=self._max_nodes_spin.value(),
            custom_instructions=self._custom_instructions.toPlainText(),
        )

    def _save_and_accept(self) -> None:
        """Save settings and close dialog."""
        self._config = self._get_current_config()
        self._config.save()
        self.accept()

    def get_config(self) -> AgentConfig:
        """Get the configured settings."""
        return self._config
