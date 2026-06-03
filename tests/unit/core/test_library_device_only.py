"""DeviceLibrary is device-only after flow-function definitions were removed."""

import inspect

from glider.core.custom_device import CustomDeviceDefinition
from glider.core.library import DeviceLibrary


def test_library_has_no_flow_function_api():
    for attr in ("export_flow_function", "import_flow_function", "list_library_functions"):
        assert not hasattr(DeviceLibrary, attr), f"{attr} should be removed"


def test_export_library_signature_is_device_only():
    sig = inspect.signature(DeviceLibrary.export_library)
    assert "functions" not in sig.parameters


def test_device_export_import_round_trip(tmp_path):
    lib = DeviceLibrary(library_path=tmp_path)
    definition = CustomDeviceDefinition(name="Widget", description="test device")

    path = lib.export_device(definition)
    restored = lib.import_device(path)

    assert restored.name == "Widget"
