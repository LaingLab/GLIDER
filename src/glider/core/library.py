"""
Device Library - Import/Export functionality.

Provides the ability to save custom devices to standalone files for
sharing and reuse across projects.
"""

import json
import logging
from pathlib import Path
from typing import Any

from glider.core.custom_device import CustomDeviceDefinition

logger = logging.getLogger(__name__)


# File extensions
DEVICE_EXTENSION = ".gdevice"
LIBRARY_EXTENSION = ".glibrary"


class DeviceLibrary:
    """
    Manages import/export of custom devices.

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
    # Combined Library Export/Import
    # =========================================================================

    def export_library(self, devices: list[CustomDeviceDefinition], path: Path) -> Path:
        """
        Export multiple devices to a single library file.

        Args:
            devices: List of device definitions
            path: Target file path

        Returns:
            Path to the exported file
        """
        data = {
            "type": "glider_library",
            "version": "1.0",
            "devices": [d.to_dict() for d in devices],
        }

        # Ensure correct extension
        if not str(path).endswith(LIBRARY_EXTENSION):
            path = Path(str(path) + LIBRARY_EXTENSION)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported library with {len(devices)} devices to {path}")
        return path

    def import_library(self, path: Path) -> list[CustomDeviceDefinition]:
        """
        Import devices from a library file.

        Args:
            path: Path to the library file

        Returns:
            List of imported device definitions

        Raises:
            ValueError: If the file is not a valid library file
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "glider_library":
            raise ValueError(f"Not a valid library file: {path}")

        devices = [CustomDeviceDefinition.from_dict(d) for d in data.get("devices", [])]

        logger.info(f"Imported library with {len(devices)} devices from {path}")
        return devices

    # =========================================================================
    # Session Integration
    # =========================================================================

    def export_session_definitions(self, session, path: Path) -> Path:
        """
        Export all custom devices from a session to a library file.

        Args:
            session: ExperimentSession to export from
            path: Target file path

        Returns:
            Path to the exported file
        """
        devices = [
            CustomDeviceDefinition.from_dict(d) for d in session.custom_device_definitions.values()
        ]

        return self.export_library(devices, path)

    def import_to_session(self, session, path: Path, overwrite: bool = False) -> int:
        """
        Import custom devices from a file into a session.

        Args:
            session: ExperimentSession to import into
            path: Path to the file (device or library)
            overwrite: Whether to overwrite existing definitions with same ID

        Returns:
            Number of devices imported
        """
        file_ext = Path(path).suffix.lower()

        devices_imported = 0

        if file_ext == DEVICE_EXTENSION:
            device = self.import_device(path)
            if overwrite or device.id not in session.custom_device_definitions:
                session.add_custom_device_definition(device.to_dict())
                devices_imported = 1

        elif file_ext == LIBRARY_EXTENSION:
            for device in self.import_library(path):
                if overwrite or device.id not in session.custom_device_definitions:
                    session.add_custom_device_definition(device.to_dict())
                    devices_imported += 1

        else:
            raise ValueError(f"Unknown file type: {file_ext}")

        return devices_imported


# Global library instance
_default_library: DeviceLibrary | None = None


def get_default_library() -> DeviceLibrary:
    """Get the default device library instance."""
    global _default_library
    if _default_library is None:
        _default_library = DeviceLibrary()
    return _default_library
