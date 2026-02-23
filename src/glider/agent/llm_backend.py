"""
LLM Backend

Provides abstraction over different LLM providers with focus on O3.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional, Union

import httpx

from glider.agent.config import AgentConfig, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # Tool name for tool responses

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        msg = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class ToolDefinition:
    """Definition of a tool the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def to_ollama_format(self) -> dict[str, Any]:
        """Convert to Ollama tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Response from the LLM."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str  # "stop", "tool_calls", "length"
    usage: Optional[dict[str, int]] = None


@dataclass
class ChatChunk:
    """A streaming chunk from the LLM."""

    content: str = ""
    tool_calls: Optional[list[dict[str, Any]]] = None
    is_final: bool = False
    finish_reason: Optional[str] = None


class LLMBackend:
    """
    Unified interface for LLM providers.

    Supports:
    - Ollama (local, free)
    - OpenAI (cloud, paid)
    - Anthropic (cloud, paid)
    """

    def __init__(self, config: AgentConfig):
        """Initialize the LLM backend."""
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._config.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        stream: bool = True,
    ) -> Union[ChatResponse, AsyncIterator[ChatChunk]]:
        """
        Send a chat request to the LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools the LLM can call
            stream: Whether to stream the response

        Returns:
            ChatResponse if not streaming, AsyncIterator[ChatChunk] if streaming
        """
        if self._config.provider == LLMProvider.OLLAMA:
            if stream:
                return self._ollama_chat_stream(messages, tools)
            else:
                return await self._ollama_chat(messages, tools)
        elif self._config.provider == LLMProvider.OPENAI:
            if stream:
                return self._openai_chat_stream(messages, tools)
            else:
                return await self._openai_chat(messages, tools)
        elif self._config.provider == LLMProvider.ANTHROPIC:
            raise NotImplementedError(
                "Anthropic provider is not yet implemented. "
                "Use Ollama (local) or OpenAI instead, or set provider in agent settings."
            )
        else:
            raise ValueError(f"Unsupported provider: {self._config.provider}")

    # =========================================================================
    # Ollama Implementation
    # =========================================================================

    async def _ollama_chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
    ) -> ChatResponse:
        """Non-streaming Ollama chat."""
        client = await self._get_client()

        url = f"{self._config.base_url}/api/chat"

        payload = {
            "model": self._config.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }

        if tools:
            payload["tools"] = [t.to_ollama_format() for t in tools]

        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            return self._parse_ollama_response(data)

        except httpx.HTTPError as e:
            logger.error(f"Ollama request failed: {e}")
            raise

    async def _ollama_chat_stream(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
    ) -> AsyncIterator[ChatChunk]:
        """Streaming Ollama chat."""
        client = await self._get_client()

        url = f"{self._config.base_url}/api/chat"

        payload = {
            "model": self._config.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }

        if tools:
            payload["tools"] = [t.to_ollama_format() for t in tools]

        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code == 404:
                    # Model likely not found
                    raise ValueError(
                        f"Model '{self._config.model}' not found. "
                        f"Run 'ollama pull {self._config.model}' or change model in settings."
                    )
                response.raise_for_status()

                accumulated_tool_calls = []

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Check if this is the final message
                    if data.get("done", False):
                        # Parse final response for tool calls
                        message = data.get("message", {})
                        tool_calls = message.get("tool_calls", [])

                        if tool_calls:
                            accumulated_tool_calls = tool_calls

                        yield ChatChunk(
                            content="",
                            tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                            is_final=True,
                            finish_reason="tool_calls" if accumulated_tool_calls else "stop",
                        )
                    else:
                        # Streaming content
                        message = data.get("message", {})
                        content = message.get("content", "")

                        if content:
                            yield ChatChunk(content=content)

        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming request failed: {e}")
            raise

    def _parse_ollama_response(self, data: dict[str, Any]) -> ChatResponse:
        """Parse Ollama response into ChatResponse."""
        message = data.get("message", {})
        content = message.get("content", "")
        tool_calls_data = message.get("tool_calls", [])

        tool_calls = []
        for i, tc in enumerate(tool_calls_data):
            func = tc.get("function", {})
            tool_calls.append(
                ToolCall(
                    id=f"call_{i}",
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                )
            )

        finish_reason = "tool_calls" if tool_calls else "stop"

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    # =========================================================================
    # OpenAI Implementation (for future use)
    # =========================================================================

    async def _openai_chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
    ) -> ChatResponse:
        """Non-streaming OpenAI chat."""
        client = await self._get_client()

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._config.get_api_key()}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }

        if tools:
            payload["tools"] = [t.to_openai_format() for t in tools]

        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            return self._parse_openai_response(data)

        except httpx.HTTPError as e:
            logger.error(f"OpenAI request failed: {e}")
            raise

    async def _openai_chat_stream(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
    ) -> AsyncIterator[ChatChunk]:
        """Streaming OpenAI chat."""
        client = await self._get_client()

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._config.get_api_key()}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": True,
        }

        if tools:
            payload["tools"] = [t.to_openai_format() for t in tools]

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        yield ChatChunk(is_final=True, finish_reason="stop")
                        break

                    try:
                        data = json.loads(data_str)
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})

                        content = delta.get("content", "")
                        finish_reason = choice.get("finish_reason")

                        yield ChatChunk(
                            content=content,
                            is_final=finish_reason is not None,
                            finish_reason=finish_reason,
                        )
                    except json.JSONDecodeError:
                        continue

        except httpx.HTTPError as e:
            logger.error(f"OpenAI streaming request failed: {e}")
            raise

    def _parse_openai_response(self, data: dict[str, Any]) -> ChatResponse:
        """Parse OpenAI response into ChatResponse."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content", "") or ""
        tool_calls_data = message.get("tool_calls", [])

        tool_calls = []
        for tc in tool_calls_data:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args,
                )
            )

        usage = data.get("usage", {})

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def check_connection(self) -> bool:
        """Check if the LLM backend is reachable."""
        try:
            if self._config.provider == LLMProvider.OLLAMA:
                client = await self._get_client()
                response = await client.get(f"{self._config.base_url}/api/tags")
                return response.status_code == 200
            elif self._config.provider == LLMProvider.OPENAI:
                client = await self._get_client()
                headers = {"Authorization": f"Bearer {self._config.get_api_key()}"}
                response = await client.get("https://api.openai.com/v1/models", headers=headers)
                return response.status_code == 200
            return False
        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            return False

    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            if self._config.provider == LLMProvider.OLLAMA:
                client = await self._get_client()
                response = await client.get(f"{self._config.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
            return []
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []
