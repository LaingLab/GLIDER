"""
Plugin Manager - Discovers and loads GLIDER plugins.

Plugins are Python packages that can provide:
- Hardware drivers (board implementations)
- Device types
- Node types
- UI components

Plugins are discovered from:
1. Entry points (glider.driver, glider.device, glider.node)
2. Plugin directory (~/.glider/plugins)
"""

import asyncio
import importlib
import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a plugin."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    entry_point: str = ""  # e.g., "my_plugin:setup"
    plugin_type: str = "generic"  # "driver", "device", "node", "generic"
    requirements: list[str] = field(default_factory=list)
    enabled: bool = True
    loaded: bool = False
    module: Any | None = None
    error: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "entry_point": self.entry_point,
            "plugin_type": self.plugin_type,
            "requirements": self.requirements,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "error": self.error,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginInfo":
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            entry_point=data.get("entry_point", ""),
            plugin_type=data.get("plugin_type", "generic"),
            requirements=data.get("requirements", []),
            enabled=data.get("enabled", True),
            path=data.get("path"),
        )


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Plugins can be discovered from:
    - Python entry points (installed packages)
    - Plugin directory (local development/distribution)
    """

    # Default plugin directory
    DEFAULT_PLUGIN_DIR = Path.home() / ".glider" / "plugins"

    # Regex for validating pip requirement strings (PEP 508 subset).
    # Allows package names with optional extras, version specifiers, and environment markers.
    _REQUIREMENT_PATTERN = re.compile(
        r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?(\[[A-Za-z0-9,._-]+\])?"
        r"(\s*(~=|==|!=|<=?|>=?|===)\s*[A-Za-z0-9.*+!_-]+)*$"
    )

    def __init__(
        self,
        plugin_dirs: list[Path] | None = None,
        enable_directory_plugins: bool = False,
    ):
        """
        Initialize the plugin manager.

        Args:
            plugin_dirs: Additional directories to search for plugins
            enable_directory_plugins: Whether to load plugins from the directory.
                Disabled by default because directory plugins execute arbitrary code.
        """
        self._plugins: dict[str, PluginInfo] = {}
        self._directory_plugins_enabled = enable_directory_plugins
        self._plugin_dirs = [self.DEFAULT_PLUGIN_DIR]
        if plugin_dirs:
            self._plugin_dirs.extend(plugin_dirs)

        # Ensure plugin directories exist
        for plugin_dir in self._plugin_dirs:
            plugin_dir.mkdir(parents=True, exist_ok=True)

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        """Dictionary of discovered plugins."""
        return self._plugins.copy()

    @property
    def loaded_plugins(self) -> dict[str, PluginInfo]:
        """Dictionary of loaded plugins."""
        return {k: v for k, v in self._plugins.items() if v.loaded}

    def get_plugin(self, name: str) -> PluginInfo | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    async def discover_plugins(self) -> list[PluginInfo]:
        """
        Discover all available plugins.

        Searches entry points and plugin directories.

        Returns:
            List of discovered plugins
        """
        discovered = []

        # Discover from entry points
        discovered.extend(await self._discover_from_entry_points())

        # Discover from plugin directories (disabled by default — executes arbitrary code)
        if self._directory_plugins_enabled:
            logger.warning(
                "Directory plugin loading is enabled. Plugins in %s can execute "
                "arbitrary code. Only enable this if you trust the plugin source.",
                [str(d) for d in self._plugin_dirs],
            )
            for plugin_dir in self._plugin_dirs:
                discovered.extend(await self._discover_from_directory(plugin_dir))
        else:
            logger.debug("Directory plugin loading is disabled")

        # Update registry
        for plugin in discovered:
            if plugin.name not in self._plugins:
                self._plugins[plugin.name] = plugin

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    async def _discover_from_entry_points(self) -> list[PluginInfo]:
        """Discover plugins from Python entry points."""
        discovered = []

        try:
            # Try importlib.metadata (Python 3.10+)
            try:
                from importlib.metadata import entry_points
            except ImportError:
                from importlib_metadata import entry_points

            # Look for GLIDER entry point groups
            for group in ["glider.driver", "glider.device", "glider.node"]:
                try:
                    # Handle different Python versions
                    eps = entry_points()
                    if hasattr(eps, "select"):
                        # Python 3.10+
                        group_eps = eps.select(group=group)
                    elif hasattr(eps, "get"):
                        # Python 3.9
                        group_eps = eps.get(group, [])
                    else:
                        # Older versions
                        group_eps = eps.get(group, [])

                    for ep in group_eps:
                        plugin_type = group.split(".")[-1]
                        info = PluginInfo(
                            name=ep.name,
                            entry_point=f"{ep.value}",
                            plugin_type=plugin_type,
                        )
                        discovered.append(info)
                        logger.debug(f"Discovered entry point plugin: {ep.name}")

                except Exception as e:
                    logger.warning(f"Error discovering entry points for {group}: {e}")

        except Exception as e:
            logger.warning(f"Entry point discovery failed: {e}")

        return discovered

    async def _discover_from_directory(self, plugin_dir: Path) -> list[PluginInfo]:
        """Discover plugins from a directory."""
        discovered = []

        if not plugin_dir.exists():
            return discovered

        try:
            dir_items = list(plugin_dir.iterdir())
        except PermissionError:
            logger.warning(f"Permission denied accessing plugin directory: {plugin_dir}")
            return discovered
        except OSError as e:
            logger.warning(f"Error accessing plugin directory {plugin_dir}: {e}")
            return discovered

        for item in dir_items:
            if item.is_dir():
                # Look for manifest.json
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, encoding="utf-8") as f:
                            manifest = json.load(f)

                        info = PluginInfo.from_dict(manifest)
                        info.path = str(item)
                        discovered.append(info)
                        logger.debug(f"Discovered directory plugin: {info.name}")

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON in manifest {manifest_path}: "
                            f"line {e.lineno}, column {e.colno}: {e.msg}"
                        )
                    except KeyError as e:
                        logger.warning(f"Missing required field in manifest {manifest_path}: {e}")
                    except PermissionError:
                        logger.warning(f"Permission denied reading manifest: {manifest_path}")
                    except UnicodeDecodeError as e:
                        logger.warning(f"Encoding error in manifest {manifest_path}: {e}")
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error loading manifest from {item}: {type(e).__name__}: {e}"
                        )

                # Also check for __init__.py with glider_plugin info
                elif (item / "__init__.py").exists():
                    try:
                        info = PluginInfo(
                            name=item.name,
                            entry_point=f"{item.name}:setup",
                            path=str(item),
                        )
                        discovered.append(info)
                        logger.debug(f"Discovered package plugin: {info.name}")

                    except Exception as e:
                        logger.warning(f"Error discovering plugin from {item}: {e}")

        return discovered

    async def load_plugins(self) -> dict[str, bool]:
        """
        Load all enabled plugins.

        Returns:
            Dictionary of plugin_name -> success
        """
        results = {}

        for name, info in self._plugins.items():
            if info.enabled and not info.loaded:
                results[name] = await self.load_plugin(name)
            else:
                results[name] = info.loaded

        return results

    async def load_plugin(self, name: str) -> bool:
        """
        Load a specific plugin.

        Args:
            name: Plugin name

        Returns:
            True if loaded successfully
        """
        info = self._plugins.get(name)
        if info is None:
            logger.error(f"Plugin not found: {name}")
            return False

        if info.loaded:
            return True

        logger.info(f"Loading plugin: {name}")

        try:
            # Add plugin path to sys.path if needed
            if info.path:
                plugin_path = Path(info.path)
                if plugin_path.parent not in [Path(p) for p in sys.path]:
                    sys.path.insert(0, str(plugin_path.parent))

            # Parse entry point
            if info.entry_point:
                module_name, _, attr_name = info.entry_point.partition(":")
                attr_name = attr_name or "setup"

                # Import module
                if info.path:
                    # Load from file path
                    spec = importlib.util.spec_from_file_location(
                        module_name, Path(info.path) / "__init__.py"
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                    else:
                        raise ImportError(f"Could not load module from {info.path}")
                else:
                    # Load from installed package
                    module = importlib.import_module(module_name)

                info.module = module

                # Call setup function if it exists
                if hasattr(module, attr_name):
                    setup_func = getattr(module, attr_name)
                    if asyncio.iscoroutinefunction(setup_func):
                        await setup_func()
                    else:
                        setup_func()

                # Register plugin components
                await self._register_plugin_components(info, module)

            info.loaded = True
            info.error = None
            logger.info(f"Successfully loaded plugin: {name}")
            return True

        except ModuleNotFoundError as e:
            info.error = f"Module not found: {e.name}"
            logger.error(f"Plugin {name} failed to load - module not found: {e.name}")
            return False

        except ImportError as e:
            info.error = f"Import error: {e}"
            logger.error(f"Plugin {name} failed to load - import error: {e}")
            return False

        except SyntaxError as e:
            info.error = f"Syntax error in {e.filename}:{e.lineno}: {e.msg}"
            logger.error(f"Plugin {name} has syntax error at {e.filename}:{e.lineno}: {e.msg}")
            return False

        except AttributeError as e:
            info.error = f"Missing attribute: {e}"
            logger.error(f"Plugin {name} failed - missing attribute: {e}")
            return False

        except TypeError as e:
            info.error = f"Type error during setup: {e}"
            logger.error(f"Plugin {name} setup function failed with type error: {e}")
            return False

        except FileNotFoundError as e:
            info.error = f"File not found: {e.filename}"
            logger.error(f"Plugin {name} failed - file not found: {e.filename}")
            return False

        except PermissionError as e:
            info.error = f"Permission denied: {e.filename}"
            logger.error(f"Plugin {name} failed - permission denied: {e.filename}")
            return False

        except Exception as e:
            # Catch-all for unexpected errors
            info.error = f"Unexpected error: {type(e).__name__}: {e}"
            logger.error(f"Plugin {name} failed with unexpected error: {type(e).__name__}: {e}")
            return False

    async def _register_plugin_components(self, info: PluginInfo, module: Any) -> None:
        """Register components provided by a plugin."""
        # Register board drivers
        if hasattr(module, "BOARD_DRIVERS"):
            from glider.core.hardware_manager import HardwareManager

            for driver_name, driver_class in module.BOARD_DRIVERS.items():
                HardwareManager.register_driver(driver_name, driver_class)
                logger.debug(f"Registered driver from plugin {info.name}: {driver_name}")

        # Register device types
        if hasattr(module, "DEVICE_TYPES"):
            from glider.hal.base_device import DEVICE_REGISTRY

            for device_name, device_class in module.DEVICE_TYPES.items():
                DEVICE_REGISTRY[device_name] = device_class
                logger.debug(f"Registered device from plugin {info.name}: {device_name}")

        # Register node types
        if hasattr(module, "NODE_TYPES"):
            from glider.core.flow_engine import FlowEngine

            for node_name, node_class in module.NODE_TYPES.items():
                FlowEngine.register_node(node_name, node_class)
                logger.debug(f"Registered node from plugin {info.name}: {node_name}")

    async def unload_plugin(self, name: str) -> bool:
        """
        Unload a specific plugin.

        Args:
            name: Plugin name

        Returns:
            True if unloaded successfully
        """
        info = self._plugins.get(name)
        if info is None or not info.loaded:
            return True

        logger.info(f"Unloading plugin: {name}")

        try:
            # Call teardown function if it exists
            if info.module and hasattr(info.module, "teardown"):
                teardown_func = info.module.teardown
                if asyncio.iscoroutinefunction(teardown_func):
                    await teardown_func()
                else:
                    teardown_func()

            info.loaded = False
            info.module = None
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {name}: {e}")
            return False

    async def unload_all(self) -> None:
        """Unload all loaded plugins."""
        for name in list(self._plugins.keys()):
            await self.unload_plugin(name)

    async def reload_plugin(self, name: str) -> bool:
        """
        Reload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if reloaded successfully
        """
        await self.unload_plugin(name)
        return await self.load_plugin(name)

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin."""
        info = self._plugins.get(name)
        if info:
            info.enabled = True
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin."""
        info = self._plugins.get(name)
        if info:
            info.enabled = False
            return True
        return False

    async def install_requirements(self, name: str) -> bool:
        """
        Install requirements for a plugin.

        Args:
            name: Plugin name

        Returns:
            True if installation successful
        """
        info = self._plugins.get(name)
        if info is None:
            return False

        if not info.requirements:
            return True

        # Validate each requirement string to prevent command injection
        for req in info.requirements:
            if not self._REQUIREMENT_PATTERN.match(req):
                logger.error(f"Invalid requirement string for plugin {name}: {req!r}")
                return False

        logger.info(f"Installing requirements for plugin {name}: {info.requirements}")

        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + info.requirements,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Failed to install requirements: {result.stderr}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error installing requirements: {e}")
            return False

    def get_plugin_info_list(self) -> list[dict[str, Any]]:
        """Get list of all plugins as dictionaries."""
        return [info.to_dict() for info in self._plugins.values()]
