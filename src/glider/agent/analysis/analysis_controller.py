"""
Analysis Controller - Manages AI-powered CSV analysis conversations.

Provides a simplified controller for data analysis, using the LLM backend
to process natural language queries about tracking data.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from glider.agent.analysis.analysis_prompts import get_analysis_system_prompt
from glider.agent.analysis.analysis_tools import ANALYSIS_TOOLS, AnalysisToolExecutor
from glider.agent.config import AgentConfig
from glider.agent.llm_backend import LLMBackend, Message

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResponse:
    """Response from the analysis controller."""

    content: str = ""
    is_complete: bool = False
    error: str | None = None
    tool_used: str | None = None


class AnalysisController:
    """
    Manages AI-powered CSV analysis conversations.

    A simplified controller focused on data analysis without
    hardware control or action confirmation.
    """

    MAX_TOOL_ITERATIONS = 10  # Prevent infinite loops

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the analysis controller.

        Args:
            config: Optional agent configuration. Loads default if not provided.
        """
        self._config = config or AgentConfig.load()
        self._llm = LLMBackend(self._config)
        self._toolkit = AnalysisToolExecutor()
        self._conversation: list[Message] = []
        self._is_processing = False

    @property
    def is_processing(self) -> bool:
        """Whether the controller is currently processing a message."""
        return self._is_processing

    async def check_connection(self) -> tuple[bool, str]:
        """
        Check if the LLM backend is reachable.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Try to get available models as a connection test
            client = await self._llm._get_client()
            response = await client.get(f"{self._config.base_url}/api/tags")

            if response.status_code == 200:
                return True, "Connected to Ollama"
            else:
                return False, f"Connection failed: HTTP {response.status_code}"

        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            return False, f"Connection failed: {str(e)}"

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._conversation.clear()
        logger.debug("Conversation cleared")

    def clear_data(self) -> None:
        """Clear all loaded CSV data."""
        self._toolkit.clear()
        logger.debug("Analysis data cleared")

    def get_loaded_files(self) -> list[dict[str, Any]]:
        """
        Get list of currently loaded files.

        Returns:
            List of file info dictionaries
        """
        return self._toolkit.get_loaded_files_summary()

    async def load_file(self, file_path: str) -> dict[str, Any]:
        """
        Load a CSV file for analysis.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary with file info or error
        """
        result = self._toolkit.load_file(file_path)
        if "error" not in result:
            logger.info(f"Loaded file: {result.get('file_name')}")
        return result

    async def process_message(self, user_message: str) -> AsyncIterator[AnalysisResponse]:
        """
        Process a user message and yield streaming responses.

        Args:
            user_message: The user's message/query

        Yields:
            AnalysisResponse chunks with content and status
        """
        if self._is_processing:
            yield AnalysisResponse(error="Already processing a message", is_complete=True)
            return

        self._is_processing = True

        try:
            # Add user message to conversation
            self._conversation.append(Message(role="user", content=user_message))

            # Build messages with system prompt
            loaded_files = self._toolkit.get_loaded_files_summary()
            system_prompt = get_analysis_system_prompt(loaded_files)

            messages = [Message(role="system", content=system_prompt)]
            messages.extend(self._conversation)

            # Process with tool loop
            iteration = 0
            while iteration < self.MAX_TOOL_ITERATIONS:
                iteration += 1

                # Get LLM response with tools
                accumulated_content = ""
                tool_calls = []

                # Await chat to get the async iterator, then iterate
                stream = await self._llm.chat(
                    messages=messages,
                    tools=ANALYSIS_TOOLS,
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.content:
                        accumulated_content += chunk.content
                        yield AnalysisResponse(content=chunk.content)

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls

                    if chunk.is_final:
                        break

                # If there are tool calls, execute them
                if tool_calls:
                    # Add assistant message with tool calls to conversation
                    messages.append(
                        Message(
                            role="assistant",
                            content=accumulated_content,
                            tool_calls=tool_calls,
                        )
                    )

                    # Execute each tool and add results
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("function", {}).get("name", "")
                        tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                        tool_id = tool_call.get("id", "")

                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            tool_args = {}

                        logger.debug(f"Executing tool: {tool_name}({tool_args})")

                        # Execute the tool
                        result = self._toolkit.execute_tool(tool_name, tool_args)

                        yield AnalysisResponse(tool_used=tool_name)

                        # Add tool result to messages
                        messages.append(
                            Message(
                                role="tool",
                                content=json.dumps(result),
                                tool_call_id=tool_id,
                                name=tool_name,
                            )
                        )

                    # Continue loop to let LLM process tool results
                    continue

                else:
                    # No tool calls - we're done
                    if accumulated_content:
                        self._conversation.append(
                            Message(role="assistant", content=accumulated_content)
                        )

                    yield AnalysisResponse(is_complete=True)
                    break

            else:
                # Hit max iterations
                logger.warning("Hit max tool iterations")
                yield AnalysisResponse(
                    content="\n\n(Reached maximum tool iterations)",
                    is_complete=True,
                )

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            yield AnalysisResponse(error=str(e), is_complete=True)

        finally:
            self._is_processing = False

    async def shutdown(self) -> None:
        """Clean up resources."""
        try:
            await self._llm.close()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
