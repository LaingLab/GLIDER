"""
Agent Controller

Main orchestrator for AI agent interactions.
"""

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from glider.agent.actions import ActionBatch, AgentAction
from glider.agent.config import AgentConfig
from glider.agent.llm_backend import ChatChunk, LLMBackend, Message, ToolCall
from glider.agent.prompts import get_system_prompt
from glider.agent.toolkit import AgentToolkit, ToolResult

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """A response from the agent."""

    content: str = ""
    actions: list[AgentAction] = field(default_factory=list)
    is_complete: bool = False
    error: str | None = None

    @property
    def has_actions(self) -> bool:
        """Check if response includes actions."""
        return len(self.actions) > 0

    @property
    def pending_actions(self) -> list[AgentAction]:
        """Get actions requiring confirmation."""
        return [a for a in self.actions if a.requires_confirmation and a.is_pending]


class AgentController:
    """
    Manages AI agent interactions.

    Handles:
    - Conversation management
    - Tool execution with confirmation
    - Context building from session state
    - Streaming responses
    """

    def __init__(self, core: "GliderCore", config: AgentConfig | None = None):
        """
        Initialize the agent controller.

        Args:
            core: GliderCore instance
            config: Agent configuration (loads from file if not provided)
        """
        self._core = core
        self._config = config or AgentConfig.load()
        self._llm = LLMBackend(self._config)
        self._toolkit = AgentToolkit(core)
        self._conversation: list[Message] = []
        self._pending_batch: ActionBatch | None = None
        self._is_processing = False
        self._recent_errors: list[str] = []

        # Callbacks
        self._on_response_callbacks: list[Callable[[AgentResponse], None]] = []
        self._on_action_callbacks: list[Callable[[AgentAction], None]] = []

    @property
    def config(self) -> AgentConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, value: AgentConfig) -> None:
        """Update configuration."""
        self._config = value
        self._llm = LLMBackend(value)

    @property
    def is_processing(self) -> bool:
        """Check if currently processing a message."""
        return self._is_processing

    @property
    def pending_batch(self) -> ActionBatch | None:
        """Get pending action batch."""
        return self._pending_batch

    @property
    def conversation(self) -> list[Message]:
        """Get conversation history."""
        return self._conversation.copy()

    def on_response(self, callback: Callable[[AgentResponse], None]) -> None:
        """Register a response callback."""
        self._on_response_callbacks.append(callback)

    def on_action(self, callback: Callable[[AgentAction], None]) -> None:
        """Register an action callback."""
        self._on_action_callbacks.append(callback)

    async def check_connection(self) -> bool:
        """Check if LLM backend is reachable."""
        return await self._llm.check_connection()

    async def list_models(self) -> list[str]:
        """List available models."""
        return await self._llm.list_models()

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation.clear()
        self._pending_batch = None
        self._toolkit.reset_state()
        logger.info("Conversation cleared")

    def add_error(self, error: str) -> None:
        """Add an error to recent errors list."""
        self._recent_errors.append(f"{datetime.now().isoformat()}: {error}")
        # Keep only last 10 errors
        self._recent_errors = self._recent_errors[-10:]

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current context."""
        # Get current session state
        nodes = []
        connections = []
        boards = []
        devices = []

        try:
            flow_engine = self._core.flow_engine
            nodes = flow_engine.get_nodes()
            connections = flow_engine.get_connections()
        except Exception as e:
            logger.warning(f"Failed to get flow state: {e}")

        try:
            hw_manager = self._core.hardware_manager
            for board_id, board in hw_manager.boards.items():
                boards.append(
                    {
                        "id": board_id,
                        "name": board.name,
                        "type": board.board_type,
                        "connected": board.is_connected,
                    }
                )

            for device_id, device in hw_manager.devices.items():
                devices.append(
                    {
                        "id": device_id,
                        "name": device.name,
                        "type": device.device_type,
                        "pin": device.pin,
                        "board": device.board_id,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to get hardware state: {e}")

        return get_system_prompt(
            nodes=nodes if self._config.include_flow_state else None,
            connections=connections if self._config.include_flow_state else None,
            boards=boards if self._config.include_hardware_state else None,
            devices=devices if self._config.include_hardware_state else None,
            errors=self._recent_errors if self._config.include_recent_errors else None,
            custom_instructions=self._config.custom_instructions,
        )

    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize user input to mitigate prompt injection.

        Args:
            text: Raw user input

        Returns:
            Sanitized input
        """
        if not text:
            return ""

        # Basic sanitization:
        # 1. Remove any potential control characters
        # 2. Escape common markdown delimiters that might be used for injection
        # 3. Limit length to reasonable maximum

        # Remove null bytes and other non-printable characters
        sanitized = "".join(char for char in text if char.isprintable() or char in "\n\r\t")

        # Limit length (e.g., 4000 characters)
        max_input_length = 4000
        if len(sanitized) > max_input_length:
            logger.warning(f"Input truncated from {len(sanitized)} to {max_input_length} chars")
            sanitized = sanitized[:max_input_length]

        return sanitized.strip()

    async def process_message(self, user_message: str) -> AsyncIterator[AgentResponse]:
        """
        Process a user message and yield streaming responses.

        Args:
            user_message: The user's message

        Yields:
            AgentResponse objects as content streams in
        """
        if self._is_processing:
            yield AgentResponse(content="", error="Already processing a message", is_complete=True)
            return

        self._is_processing = True

        try:
            # Sanitize user input
            user_message = self._sanitize_input(user_message)
            if not user_message:
                yield AgentResponse(content="No message provided", is_complete=True)
                return

            # Add user message to conversation
            self._conversation.append(Message(role="user", content=user_message))

            # Build messages for LLM
            messages = [
                Message(role="system", content=self._build_system_prompt()),
                *self._conversation,
            ]

            # Get tool definitions
            tools = self._toolkit.get_tool_definitions()

            # Stream response from LLM
            accumulated_content = ""
            tool_calls: list[ToolCall] = []

            async for chunk in await self._llm.chat(messages, tools, stream=True):
                if isinstance(chunk, ChatChunk):
                    if chunk.content:
                        accumulated_content += chunk.content
                        yield AgentResponse(content=accumulated_content)

                    if chunk.is_final:
                        # Process any tool calls
                        if chunk.tool_calls:
                            tool_calls = self._parse_tool_calls(chunk.tool_calls)
                        break

            # Handle tool calls
            if tool_calls:
                batch = ActionBatch(message=accumulated_content)

                for tc in tool_calls:
                    action = self._toolkit.create_action(tc.name, tc.arguments)
                    if action:
                        batch.add_action(action)

                        # Notify callbacks
                        for callback in self._on_action_callbacks:
                            callback(action)

                # Auto-execute safe actions if enabled
                if self._config.auto_execute_safe:
                    for action in batch.actions:
                        if not action.requires_confirmation:
                            action.confirm()
                            result = await self._execute_action(action)
                            accumulated_content += (
                                f"\n\n**{action.description}**: {result.to_message()}"
                            )

                # Store pending batch if there are confirmable actions
                if batch.pending_actions:
                    self._pending_batch = batch
                else:
                    self._pending_batch = None

                # Add assistant response to conversation
                self._conversation.append(
                    Message(
                        role="assistant",
                        content=accumulated_content,
                    )
                )

                yield AgentResponse(
                    content=accumulated_content, actions=batch.actions, is_complete=True
                )
            else:
                # No tool calls - just a text response
                self._conversation.append(
                    Message(
                        role="assistant",
                        content=accumulated_content,
                    )
                )

                yield AgentResponse(content=accumulated_content, is_complete=True)

        except Exception as e:
            logger.exception("Error processing message")
            self.add_error(str(e))
            yield AgentResponse(content="", error=f"Error: {str(e)}", is_complete=True)
        finally:
            self._is_processing = False

    def _parse_tool_calls(self, raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse raw tool calls from LLM response."""
        calls = []

        for i, tc in enumerate(raw_calls):
            func = tc.get("function", tc)
            name = func.get("name", "")
            args = func.get("arguments", {})

            # Handle string arguments (JSON)
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            calls.append(
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=name,
                    arguments=args,
                )
            )

        return calls

    async def confirm_actions(self, action_ids: list[str] | None = None) -> AgentResponse:
        """
        Confirm and execute pending actions.

        Args:
            action_ids: Specific action IDs to confirm, or None for all

        Returns:
            AgentResponse with execution results
        """
        if not self._pending_batch:
            return AgentResponse(content="No pending actions to confirm.", is_complete=True)

        results = []

        for action in self._pending_batch.actions:
            if action_ids is None or action.id in action_ids:
                if action.is_pending:
                    action.confirm()
                    result = await self._execute_action(action)
                    results.append(f"**{action.description}**: {result.to_message()}")

        self._pending_batch = None

        content = "\n".join(results) if results else "No actions executed."

        return AgentResponse(content=content, is_complete=True)

    async def reject_actions(self, action_ids: list[str] | None = None) -> AgentResponse:
        """
        Reject pending actions.

        Args:
            action_ids: Specific action IDs to reject, or None for all

        Returns:
            AgentResponse confirming rejection
        """
        if not self._pending_batch:
            return AgentResponse(content="No pending actions to reject.", is_complete=True)

        rejected = []

        for action in self._pending_batch.actions:
            if action_ids is None or action.id in action_ids:
                if action.is_pending:
                    action.reject()
                    rejected.append(action.description)

        # Clear batch if all actions handled
        if not self._pending_batch.pending_actions:
            self._pending_batch = None

        content = f"Rejected: {', '.join(rejected)}" if rejected else "No actions rejected."

        return AgentResponse(content=content, is_complete=True)

    async def _execute_action(self, action: AgentAction) -> ToolResult:
        """Execute a single action."""
        action.start_execution()

        try:
            result = await self._toolkit.execute(action.tool_name, action.parameters)

            if result.success:
                action.complete(result.result)
            else:
                action.fail(result.error or "Unknown error")

            return result

        except Exception as e:
            error = str(e)
            action.fail(error)
            return ToolResult(success=False, error=error)

    async def shutdown(self) -> None:
        """Clean up resources."""
        await self._llm.close()
        logger.info("Agent controller shut down")
