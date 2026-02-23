"""
Device and Flow Function Library - Import/Export functionality.

Provides the ability to save custom devices and flow functions to
standalone files for sharing and reuse across projects.
"""

import json
import logging
from pathlib import Path
from typing import Any

from glider.core.custom_device import CustomDeviceDefinition
from glider.core.flow_function import FlowFunctionDefinition

logger = logging.getLogger(__name__)


# File extensions
DEVICE_EXTENSION = ".gdevice"
FLOW_FUNCTION_EXTENSION = ".gflow"
LIBRARY_EXTENSION = ".glibrary"


class DeviceLibrary:
    """
    Manages import/export of custom devices and flow functions.

    Supports:
    - Exporting individual definitions to files
    - Importing definitions from files
    - Managing a library of definitions in a directory
    """

    def __init__(self, library_path: Path | None = None):
        """
        Initialize the device library.

        Args:
            library_path: Path to the library directory (default: user's home/.glider/library)
        """
        if library_path is None:
            library_path = Path.home() / ".glider" / "library"
        self._library_path = library_path
        self._ensure_library_exists()

    def _ensure_library_exists(self) -> None:
        """Ensure the library directory exists."""
        self._library_path.mkdir(parents=True, exist_ok=True)
        (self._library_path / "devices").mkdir(exist_ok=True)
        (self._library_path / "functions").mkdir(exist_ok=True)

    @property
    def library_path(self) -> Path:
        """Path to the library directory."""
        return self._library_path

    # =========================================================================
    # Custom Device Import/Export
    # =========================================================================

    def export_device(self, definition: CustomDeviceDefinition, path: Path | None = None) -> Path:
        """
        Export a custom device definition to a file.

        Args:
            definition: The device definition to export
            path: Target file path (default: library/devices/{name}.gdevice)

        Returns:
            Path to the exported file
        """
        if path is None:
            safe_name = definition.name.lower().replace(" ", "_")
            path = self._library_path / "devices" / f"{safe_name}{DEVICE_EXTENSION}"

        data = {
            "type": "custom_device",
            "version": "1.0",
            "definition": definition.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported custom device '{definition.name}' to {path}")
        return path

    def import_device(self, path: Path) -> CustomDeviceDefinition:
        """
        Import a custom device definition from a file.

        Args:
            path: Path to the device file

        Returns:
            The imported device definition

        Raises:
            ValueError: If the file is not a valid device definition
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "custom_device":
            raise ValueError(f"Not a valid device file: {path}")

        definition = CustomDeviceDefinition.from_dict(data["definition"])
        logger.info(f"Imported custom device '{definition.name}' from {path}")
        return definition

    def list_library_devices(self) -> list[dict[str, Any]]:
        """
        List all devices in the library.

        Returns:
            List of device info dictionaries with 'name', 'id', 'path'
        """
        devices = []
        devices_dir = self._library_path / "devices"

        for file_path in devices_dir.glob(f"*{DEVICE_EXTENSION}"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("type") == "custom_device":
                    definition = data.get("definition", {})
                    devices.append(
                        {
                            "name": definition.get("name", "Unknown"),
                            "id": definition.get("id", ""),
                            "description": definition.get("description", ""),
                            "path": str(file_path),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to read device file {file_path}: {e}")

        return devices

    # =========================================================================
    # Flow Function Import/Export
    # =========================================================================

    def export_flow_function(
        self, definition: FlowFunctionDefinition, path: Path | None = None
    ) -> Path:
        """
        Export a flow function definition to a file.

        Args:
            definition: The flow function definition to export
            path: Target file path (default: library/functions/{name}.gflow)

        Returns:
            Path to the exported file
        """
        if path is None:
            safe_name = definition.name.lower().replace(" ", "_")
            path = self._library_path / "functions" / f"{safe_name}{FLOW_FUNCTION_EXTENSION}"

        data = {
            "type": "flow_function",
            "version": "1.0",
            "definition": definition.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported flow function '{definition.name}' to {path}")
        return path

    def import_flow_function(self, path: Path) -> FlowFunctionDefinition:
        """
        Import a flow function definition from a file.

        Args:
            path: Path to the flow function file

        Returns:
            The imported flow function definition

        Raises:
            ValueError: If the file is not a valid flow function definition
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "flow_function":
            raise ValueError(f"Not a valid flow function file: {path}")

        definition = FlowFunctionDefinition.from_dict(data["definition"])
        logger.info(f"Imported flow function '{definition.name}' from {path}")
        return definition

    def list_library_functions(self) -> list[dict[str, Any]]:
        """
        List all flow functions in the library.

        Returns:
            List of function info dictionaries with 'name', 'id', 'path'
        """
        functions = []
        functions_dir = self._library_path / "functions"

        for file_path in functions_dir.glob(f"*{FLOW_FUNCTION_EXTENSION}"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("type") == "flow_function":
                    definition = data.get("definition", {})
                    functions.append(
                        {
                            "name": definition.get("name", "Unknown"),
                            "id": definition.get("id", ""),
                            "description": definition.get("description", ""),
                            "path": str(file_path),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to read function file {file_path}: {e}")

        return functions

    # =========================================================================
    # Combined Library Export/Import
    # =========================================================================

    def export_library(
        self,
        devices: list[CustomDeviceDefinition],
        functions: list[FlowFunctionDefinition],
        path: Path,
    ) -> Path:
        """
        Export multiple devices and functions to a single library file.

        Args:
            devices: List of device definitions
            functions: List of flow function definitions
            path: Target file path

        Returns:
            Path to the exported file
        """
        data = {
            "type": "glider_library",
            "version": "1.0",
            "devices": [d.to_dict() for d in devices],
            "functions": [f.to_dict() for f in functions],
        }

        # Ensure correct extension
        if not str(path).endswith(LIBRARY_EXTENSION):
            path = Path(str(path) + LIBRARY_EXTENSION)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Exported library with {len(devices)} devices and {len(functions)} functions to {path}"
        )
        return path

    def import_library(
        self, path: Path
    ) -> tuple[list[CustomDeviceDefinition], list[FlowFunctionDefinition]]:
        """
        Import devices and functions from a library file.

        Args:
            path: Path to the library file

        Returns:
            Tuple of (devices, functions)

        Raises:
            ValueError: If the file is not a valid library file
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "glider_library":
            raise ValueError(f"Not a valid library file: {path}")

        devices = [CustomDeviceDefinition.from_dict(d) for d in data.get("devices", [])]
        functions = [FlowFunctionDefinition.from_dict(f) for f in data.get("functions", [])]

        logger.info(
            f"Imported library with {len(devices)} devices and {len(functions)} functions from {path}"
        )
        return devices, functions

    # =========================================================================
    # Session Integration
    # =========================================================================

    def export_session_definitions(self, session, path: Path) -> Path:
        """
        Export all custom definitions from a session to a library file.

        Args:
            session: ExperimentSession to export from
            path: Target file path

        Returns:
            Path to the exported file
        """
        devices = [
            CustomDeviceDefinition.from_dict(d) for d in session.custom_device_definitions.values()
        ]
        functions = [
            FlowFunctionDefinition.from_dict(f) for f in session.flow_function_definitions.values()
        ]

        return self.export_library(devices, functions, path)

    def import_to_session(self, session, path: Path, overwrite: bool = False) -> tuple[int, int]:
        """
        Import definitions from a file into a session.

        Args:
            session: ExperimentSession to import into
            path: Path to the file (device, function, or library)
            overwrite: Whether to overwrite existing definitions with same ID

        Returns:
            Tuple of (devices_imported, functions_imported)
        """
        file_ext = Path(path).suffix.lower()

        devices_imported = 0
        functions_imported = 0

        if file_ext == DEVICE_EXTENSION:
            device = self.import_device(path)
            if overwrite or device.id not in session.custom_device_definitions:
                session.add_custom_device_definition(device.to_dict())
                devices_imported = 1

        elif file_ext == FLOW_FUNCTION_EXTENSION:
            func = self.import_flow_function(path)
            if overwrite or func.id not in session.flow_function_definitions:
                session.add_flow_function_definition(func.to_dict())
                functions_imported = 1

        elif file_ext == LIBRARY_EXTENSION:
            devices, functions = self.import_library(path)
            for device in devices:
                if overwrite or device.id not in session.custom_device_definitions:
                    session.add_custom_device_definition(device.to_dict())
                    devices_imported += 1
            for func in functions:
                if overwrite or func.id not in session.flow_function_definitions:
                    session.add_flow_function_definition(func.to_dict())
                    functions_imported += 1

        else:
            raise ValueError(f"Unknown file type: {file_ext}")

        return devices_imported, functions_imported


# Global library instance
_default_library: DeviceLibrary | None = None


def get_default_library() -> DeviceLibrary:
    """Get the default device library instance."""
    global _default_library
    if _default_library is None:
        _default_library = DeviceLibrary()
    return _default_library
