"""Load torch's native libraries before anything else can poison the order.

On Windows, importing PyQt6 before torch exhausts the process's static TLS
slots, and the next ``import torch`` dies with

    [WinError 1114] A dynamic link library (DLL) initialization routine
    failed. Error loading ...\\torch\\lib\\c10.dll

``glider/__main__.py`` already guards against this by importing torch above its
PyQt6 imports, which is enough for a source install. It is not enough for a
frozen build: PyInstaller's bootstrap runs before the entry script, so by the
time ``__main__`` executes the order can already be lost -- and because that
guard swallows every exception, the failure is silent and resurfaces much later
as "tracking fell back to contour tracking" with nothing in the log.

A PyInstaller runtime hook runs earlier than the entry script, which is the
only place left to get in front of it.

Best-effort by design: torch is optional (absent from Raspberry Pi and minimal
builds), so a failure here must never stop the app launching -- the vision
layer degrades exactly as it does without torch installed. Use --check-deps to
see whether it actually loaded.
"""

import sys

if sys.platform == "win32":
    try:
        import torch  # noqa: F401  (imported for its DLL-load side effect)
    except Exception:
        pass
