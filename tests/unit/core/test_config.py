"""
Tests for glider.core.config module.

Tests the configuration classes and persistence.
"""

import json
from pathlib import Path

from glider.core.config import (
    GliderConfig,
    HardwareConfig,
    PathConfig,
    TimingConfig,
    UIConfig,
    get_config,
    set_config,
)


class TestTimingConfig:
    """Tests for TimingConfig dataclass."""

    def test_default_values(self):
        """Test TimingConfig default values."""
        config = TimingConfig()

        assert config.device_refresh_interval_ms == 250
        # Lowered from 250 ms to 50 ms so the runner-mode elapsed timer can
        # render centiseconds (MM:SS.cc) smoothly. See RunControlPanel._update_elapsed_time.
        assert config.elapsed_timer_interval_ms == 50
        assert config.default_poll_interval == 0.1
        assert config.board_ready_timeout == 10.0
        assert config.board_operation_timeout == 5.0
        assert config.function_execution_timeout == 60.0

    def test_custom_values(self):
        """Test TimingConfig with custom values."""
        config = TimingConfig(device_refresh_interval_ms=500, board_ready_timeout=30.0)

        assert config.device_refresh_interval_ms == 500
        assert config.board_ready_timeout == 30.0
        # Other values remain default
        assert config.default_poll_interval == 0.1


class TestUIConfig:
    """Tests for UIConfig dataclass."""

    def test_default_values(self):
        """Test UIConfig default values."""
        config = UIConfig()

        assert config.min_window_width == 1024
        assert config.min_window_height == 768
        assert config.default_window_width == 1400
        assert config.default_window_height == 900
        assert config.max_undo_stack_size == 100

    def test_custom_values(self):
        """Test UIConfig with custom values."""
        config = UIConfig(min_window_width=800, min_window_height=600)

        assert config.min_window_width == 800
        assert config.min_window_height == 600


class TestHardwareConfig:
    """Tests for HardwareConfig dataclass."""

    def test_default_values(self):
        """Test HardwareConfig default values."""
        config = HardwareConfig()

        assert config.adc_resolution == 1023
        assert config.adc_reference_voltage == 5.0
        assert config.pwm_min == 0
        assert config.pwm_max == 255
        assert config.servo_min_angle == 0
        assert config.servo_max_angle == 180
        assert config.input_poll_interval_ms == 100

    def test_custom_values(self):
        """Test HardwareConfig with custom values."""
        config = HardwareConfig(adc_resolution=4095, adc_reference_voltage=3.3)

        assert config.adc_resolution == 4095
        assert config.adc_reference_voltage == 3.3


class TestPathConfig:
    """Tests for PathConfig dataclass."""

    def test_default_values(self):
        """Test PathConfig default values."""
        config = PathConfig()

        assert isinstance(config.user_config_dir, Path)
        assert isinstance(config.library_dir, Path)
        assert config.experiment_extension == ".glider"
        assert config.device_extension == ".gdevice"

    def test_paths_are_under_home(self):
        """Test that default paths are under home directory."""
        config = PathConfig()

        assert ".glider" in str(config.user_config_dir)
        assert ".glider" in str(config.library_dir)


class TestGliderConfig:
    """Tests for GliderConfig dataclass."""

    def test_default_values(self):
        """Test GliderConfig default values."""
        config = GliderConfig()

        assert isinstance(config.timing, TimingConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.hardware, HardwareConfig)
        assert isinstance(config.paths, PathConfig)

    def test_to_dict(self):
        """Test GliderConfig serialization."""
        config = GliderConfig()

        data = config.to_dict()

        assert "timing" in data
        assert "ui" in data
        assert "hardware" in data
        # to_dict only serializes subset of fields
        assert "device_refresh_interval_ms" in data["timing"]
        assert "min_window_width" in data["ui"]
        assert "adc_resolution" in data["hardware"]

    def test_from_dict(self):
        """Test GliderConfig deserialization."""
        data = {
            "timing": {"device_refresh_interval_ms": 500, "board_ready_timeout": 20.0},
            "ui": {"min_window_width": 800},
            "hardware": {"adc_resolution": 4095},
        }

        config = GliderConfig.from_dict(data)

        assert config.timing.device_refresh_interval_ms == 500
        assert config.timing.board_ready_timeout == 20.0
        assert config.ui.min_window_width == 800
        assert config.hardware.adc_resolution == 4095

    def test_from_dict_with_missing_sections(self):
        """Test that from_dict handles missing sections."""
        data = {}  # Empty dict

        config = GliderConfig.from_dict(data)

        # Should use defaults
        assert config.timing.device_refresh_interval_ms == 250
        assert config.ui.min_window_width == 1024

    def test_save_and_load(self, temp_dir):
        """Test saving and loading configuration."""
        config = GliderConfig()
        config.timing.device_refresh_interval_ms = 999
        file_path = temp_dir / "test_config.json"

        config.save(file_path)
        loaded = GliderConfig.load(file_path)

        assert loaded.timing.device_refresh_interval_ms == 999

    def test_load_nonexistent_returns_defaults(self, temp_dir):
        """Test that loading nonexistent file returns defaults."""
        nonexistent = temp_dir / "nonexistent_config.json"

        config = GliderConfig.load(nonexistent)

        # Should return default config
        assert config.timing.device_refresh_interval_ms == 250

    def test_save_creates_parent_directories(self, temp_dir):
        """Test that save creates parent directories."""
        config = GliderConfig()
        nested_path = temp_dir / "nested" / "dirs" / "config.json"

        config.save(nested_path)

        assert nested_path.exists()

    def test_roundtrip_preserves_values(self, temp_dir):
        """Test that save/load preserves configured values."""
        original = GliderConfig()
        original.timing.device_refresh_interval_ms = 123
        original.ui.max_undo_stack_size = 50

        file_path = temp_dir / "roundtrip.json"
        original.save(file_path)
        loaded = GliderConfig.load(file_path)

        # Note: only values included in to_dict() are preserved
        assert loaded.timing.device_refresh_interval_ms == 123
        assert loaded.ui.max_undo_stack_size == 50

    def test_json_format(self, temp_dir):
        """Test that saved config is valid JSON."""
        config = GliderConfig()
        file_path = temp_dir / "config.json"
        config.save(file_path)

        # Should be valid JSON
        content = file_path.read_text()
        parsed = json.loads(content)

        assert isinstance(parsed, dict)
        assert "timing" in parsed


def test_behavior_projects_dir_default():
    from glider.core.config import GliderConfig

    cfg = GliderConfig()
    assert cfg.paths.behavior_projects_dir == Path.home() / ".glider" / "behavior_projects"


class TestGlobalConfig:
    """Tests for global configuration functions."""

    def test_get_config_returns_instance(self):
        """Test that get_config returns a GliderConfig instance."""
        config = get_config()

        assert isinstance(config, GliderConfig)

    def test_set_config_changes_global(self):
        """Test that set_config changes the global instance."""
        custom = GliderConfig()
        custom.timing.device_refresh_interval_ms = 12345

        set_config(custom)
        retrieved = get_config()

        assert retrieved.timing.device_refresh_interval_ms == 12345
