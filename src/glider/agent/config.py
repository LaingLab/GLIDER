"""
Agent Configuration

Defines configuration options for the AI agent.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class AgentConfig:
    """Configuration for the AI agent."""

    # LLM Provider settings
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.2:latest"  # Change to your installed model
    api_key: str | None = None
    base_url: str = "http://localhost:11434"

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0  # seconds

    # Behavior settings
    require_confirmation: bool = True
    auto_execute_safe: bool = True
    max_actions_per_message: int = 10

    # Context settings
    include_flow_state: bool = True
    include_hardware_state: bool = True
    include_recent_errors: bool = True
    max_context_nodes: int = 50

    # Custom instructions
    custom_instructions: str = ""

    # UI settings
    show_thinking: bool = False
    stream_responses: bool = True

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the path to the config file."""
        config_dir = Path.home() / ".glider"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "agent_config.json"

    @classmethod
    def load(cls) -> "AgentConfig":
        """Load configuration from file."""
        config_path = cls.get_config_path()

        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)

                # Convert provider string to enum
                if "provider" in data:
                    data["provider"] = LLMProvider(data["provider"])

                # Handle API key from environment
                if data.get("api_key_env"):
                    data["api_key"] = os.environ.get(data.pop("api_key_env"))

                return cls(**data)

            except Exception as e:
                logger.warning(f"Failed to load agent config: {e}")

        return cls()

    def save(self) -> None:
        """Save configuration to file."""
        config_path = self.get_config_path()

        data = asdict(self)
        data["provider"] = self.provider.value

        # Don't save API key directly - use environment variable reference
        if self.api_key:
            data["api_key"] = None
            data["api_key_env"] = f"{self.provider.name}_API_KEY"

        try:
            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved agent config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save agent config: {e}")

    def get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key

        # Try environment variables
        env_vars = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }

        if self.provider in env_vars:
            return os.environ.get(env_vars[self.provider])

        return None


# Default models for each provider
DEFAULT_MODELS = {
    LLMProvider.OLLAMA: "llama3.2",
    LLMProvider.OPENAI: "gpt-4-turbo",
    LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
}
