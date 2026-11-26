import sys
from pathlib import Path

# The arch tests reuse test_lift3d's synthetic mouse and stand-in models.
sys.path.insert(0, str(Path(__file__).parent))

# On Windows, torch's DLLs fail to load once PyQt6 is imported (pytest-qt loads
# it first). GLIDER preloads torch in __main__ for the same reason.
try:
    import torch  # noqa: F401
except Exception:  # noqa: BLE001 - torch is optional
    pass
