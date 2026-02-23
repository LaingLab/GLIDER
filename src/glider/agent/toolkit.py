"""
Agent Toolkit

Aggregates all tools available to the AI agent.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from glider.agent.actions import AgentAction
from glider.agent.llm_backend import ToolDefinition
from glider.agent.tools.experiment_tools import EXPERIMENT_TOOLS, ExperimentToolExecutor
from glider.agent.tools.hardware_tools import HARDWARE_TOOLS, HardwareToolExecutor
from glider.agent.tools.knowledge_tools import KNOWLEDGE_TOOLS, KnowledgeToolExecutor

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    result: Any | None = None
    error: str | None = None

    def to_message(self) -> str:
        """Convert to a message string for the LLM."""
        if self.success:
            if isinstance(self.result, dict):
                import json

                return json.dumps(self.result, indent=2)
            return str(self.result)
        else:
            return f"Error: {self.error}"


class AgentToolkit:
    """
    Aggregates all tools and provides unified execution.

    Tools are grouped by category:
    - Experiment: Flow graph manipulation
    - Hardware: Board and device configuration
    - Knowledge: Explanations and suggestions
    """

    def __init__(self, core: "GliderCore"):
        """Initialize the toolkit."""
        self._core = core

        # Initialize executors
        self._experiment_executor = ExperimentToolExecutor(core)
        self._hardware_executor = HardwareToolExecutor(core)
        self._knowledge_executor = KnowledgeToolExecutor(core)

        # Build tool registry
        self._tools: dict[str, ToolDefinition] = {}
        self._tool_executors: dict[str, Any] = {}

        for tool in EXPERIMENT_TOOLS:
            self._tools[tool.name] = tool
            self._tool_executors[tool.name] = self._experiment_executor

        for tool in HARDWARE_TOOLS:
            self._tools[tool.name] = tool
            self._tool_executors[tool.name] = self._hardware_executor

        for tool in KNOWLEDGE_TOOLS:
            self._tools[tool.name] = tool
            self._tool_executors[tool.name] = self._knowledge_executor

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions for the LLM."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a specific tool definition."""
        return self._tools.get(name)

    def create_action(self, tool_name: str, args: dict[str, Any]) -> AgentAction | None:
        """Create an action for a tool call."""
        executor = self._tool_executors.get(tool_name)
        if executor is None:
            return None

        return executor.create_action(tool_name, args)

    async def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments

        Returns:
            ToolResult with success/failure and result/error
        """
        executor = self._tool_executors.get(tool_name)

        if executor is None:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        try:
            result = await executor.execute(tool_name, args)

            if result.get("success", False):
                return ToolResult(success=True, result=result.get("result"))
            else:
                return ToolResult(success=False, error=result.get("error", "Unknown error"))

        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return ToolResult(success=False, error=str(e))

    def reset_state(self) -> None:
        """Reset any stateful components (like auto-layout)."""
        self._experiment_executor.reset_layout()

    @property
    def tool_names(self) -> list[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        """Get total number of tools."""
        return len(self._tools)
