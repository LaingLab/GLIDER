"""
Device library: load, save, and register declarative ``.gdevice`` definitions.

Declarative custom devices live as ``.gdevice`` JSON files in the user's device
library (``~/.glider/library/devices`` by default). Each one is turned into a
``BaseDevice`` subclass and registered into ``DEVICE_REGISTRY`` so it behaves
exactly like a built-in device (appears in Add Device, works with DeviceRead,
the recorder, etc.).

Loading is safe by default because definitions are pure data, not code.
"""

import json
import logging
import re
from pathlib import Path

from glider.hal.base_device import DEVICE_REGISTRY
from glider.hal.declarative_device import build_device_class, validate_definition

logger = logging.getLogger(__name__)

DEVICE_EXTENSION = ".gdevice"


def _slug(name: str) -> str:
    """Filesystem-safe slug for a device name."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return s.strip("_") or "device"


def definition_path(devices_dir: Path, name: str) -> Path:
    return Path(devices_dir) / f"{_slug(name)}{DEVICE_EXTENSION}"


def save_definition(definition: dict, devices_dir: Path) -> Path:
    """Validate and write a definition to the device library. Returns the path."""
    errors = validate_definition(definition)
    if errors:
        raise ValueError("; ".join(errors))
    devices_dir = Path(devices_dir)
    devices_dir.mkdir(parents=True, exist_ok=True)
    path = definition_path(devices_dir, definition["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(definition, f, indent=2)
    logger.info("Saved device definition '%s' to %s", definition["name"], path)
    return path


def load_definitions(devices_dir: Path) -> list[dict]:
    """Read all valid ``.gdevice`` definitions from a directory."""
    devices_dir = Path(devices_dir)
    if not devices_dir.exists():
        return []
    definitions = []
    for path in sorted(devices_dir.glob(f"*{DEVICE_EXTENSION}")):
        try:
            with open(path, encoding="utf-8") as f:
                defn = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.warning("Skipping invalid device definition %s: %s", path, e)
            continue
        errors = validate_definition(defn)
        if errors:
            logger.warning("Skipping device definition %s: %s", path, "; ".join(errors))
            continue
        definitions.append(defn)
    return definitions


def register_definition(definition: dict) -> str:
    """Build and register a device class from a definition. Returns the type name."""
    errors = validate_definition(definition)
    if errors:
        raise ValueError("; ".join(errors))
    name = definition["name"]
    DEVICE_REGISTRY[name] = build_device_class(definition)
    logger.debug("Registered custom device type: %s", name)
    return name


def load_and_register_all(devices_dir: Path) -> list[str]:
    """Load every definition in the library and register it. Returns the names."""
    names = []
    for defn in load_definitions(devices_dir):
        try:
            names.append(register_definition(defn))
        except ValueError as e:
            logger.warning("Could not register device '%s': %s", defn.get("name"), e)
    if names:
        logger.info("Registered %d custom device type(s): %s", len(names), ", ".join(names))
    return names
