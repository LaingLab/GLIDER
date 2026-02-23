"""
Centralized Configuration for GLIDER.

This module provides a single source of truth for configuration values
that were previously hardcoded throughout the codebase.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TimingConfig:
    """Timing-related configuration values."""

    # UI refresh intervals (milliseconds)
    device_refresh_interval_ms: int = 250
    elapsed_timer_interval_ms: int = 1000

    # Hardware polling defaults (seconds)
    default_poll_interval: float = 0.1
    analog_read_interval: float = 0.1
    digital_read_interval: float = 0.1

    # Timeouts (seconds)
    board_ready_timeout: float = 10.0
    board_operation_timeout: float = 5.0
    thread_join_timeout: float = 2.0
    function_execution_timeout: float = 60.0
    camera_warmup_time: float = 0.5

    # Reconnection settings
    reconnect_interval: float = 5.0

    # Brief pulse duration for stepper/motor control
    pulse_duration: float = 0.05


@dataclass
class UIConfig:
    """UI-related configuration values."""

    # Window dimensions
    min_window_width: int = 1024
    min_window_height: int = 768
    default_window_width: int = 1400
    default_window_height: int = 900

    # Runner view dimensions (Pi touchscreen)
    runner_header_height: int = 50
    runner_controls_height: int = 160
    runner_device_card_height: int = 80

    # Control panel dimensions
    control_panel_min_width: int = 200
    control_panel_max_width: int = 400

    # Undo/Redo stack size
    max_undo_stack_size: int = 100

    # Graph view
    graph_view_min_width: int = 400
    graph_view_min_height: int = 300


@dataclass
class HardwareConfig:
    """Hardware-related configuration values."""

    # ADC settings (assuming 10-bit ADC, 5V reference)
    adc_resolution: int = 1023
    adc_reference_voltage: float = 5.0

    # PWM range
    pwm_min: int = 0
    pwm_max: int = 255

    # Servo range
    servo_min_angle: int = 0
    servo_max_angle: int = 180
    servo_default_angle: int = 90

    # Default poll interval for input devices (milliseconds)
    input_poll_interval_ms: int = 100
    min_poll_interval_ms: int = 50
    max_poll_interval_ms: int = 5000


@dataclass
class PathConfig:
    """Path-related configuration values."""

    # User configuration directory
    user_config_dir: Path = field(default_factory=lambda: Path.home() / ".glider")

    # Library paths
    library_dir: Path = field(default_factory=lambda: Path.home() / ".glider" / "library")
    devices_dir: Path = field(
        default_factory=lambda: Path.home() / ".glider" / "library" / "devices"
    )
    functions_dir: Path = field(
        default_factory=lambda: Path.home() / ".glider" / "library" / "functions"
    )

    # File extensions
    experiment_extension: str = ".glider"
    device_extension: str = ".gdevice"
    flow_extension: str = ".gflow"
    library_extension: str = ".glibrary"


@dataclass
class GliderConfig:
    """Main configuration container for GLIDER."""

    timing: TimingConfig = field(default_factory=TimingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "timing": {
                "device_refresh_interval_ms": self.timing.device_refresh_interval_ms,
                "elapsed_timer_interval_ms": self.timing.elapsed_timer_interval_ms,
                "default_poll_interval": self.timing.default_poll_interval,
                "board_ready_timeout": self.timing.board_ready_timeout,
                "board_operation_timeout": self.timing.board_operation_timeout,
                "function_execution_timeout": self.timing.function_execution_timeout,
            },
            "ui": {
                "min_window_width": self.ui.min_window_width,
                "min_window_height": self.ui.min_window_height,
                "default_window_width": self.ui.default_window_width,
                "default_window_height": self.ui.default_window_height,
                "max_undo_stack_size": self.ui.max_undo_stack_size,
            },
            "hardware": {
                "adc_resolution": self.hardware.adc_resolution,
                "adc_reference_voltage": self.hardware.adc_reference_voltage,
                "pwm_max": self.hardware.pwm_max,
                "input_poll_interval_ms": self.hardware.input_poll_interval_ms,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GliderConfig":
        """Create configuration from dictionary."""
        config = cls()

        if "timing" in data:
            timing_data = data["timing"]
            for key, value in timing_data.items():
                if hasattr(config.timing, key):
                    setattr(config.timing, key, value)

        if "ui" in data:
            ui_data = data["ui"]
            for key, value in ui_data.items():
                if hasattr(config.ui, key):
                    setattr(config.ui, key, value)

        if "hardware" in data:
            hw_data = data["hardware"]
            for key, value in hw_data.items():
                if hasattr(config.hardware, key):
                    setattr(config.hardware, key, value)

        return config

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = self.paths.user_config_dir / "config.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: Path | None = None) -> "GliderConfig":
        """Load configuration from file, using defaults if not found."""
        config = cls()

        if path is None:
            path = config.paths.user_config_dir / "config.json"

        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                config = cls.from_dict(data)
                logger.info(f"Configuration loaded from {path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {path}: {e}")
                logger.info("Using default configuration")

        return config


# Global configuration instance - lazy loaded
_config: GliderConfig | None = None


def get_config() -> GliderConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = GliderConfig.load()
    return _config


def set_config(config: GliderConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
