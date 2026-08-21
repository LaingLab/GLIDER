"""
Centralized color palette for the GLIDER "Deep Navy" theme.

All colors across the GUI reference these constants.
String constants for QSS/f-strings, QColor objects for QPainter.
"""

from PyQt6.QtGui import QColor


def with_alpha(hex_color: str, alpha: float) -> str:
    """Convert a hex color to rgba() string for use in QSS stylesheets.

    Args:
        hex_color: Color in "#RRGGBB" format.
        alpha: Alpha value from 0.0 (transparent) to 1.0 (opaque).

    Returns:
        String like "rgba(R, G, B, A)" suitable for QSS.
        NOTE: This is for QSS only. For QPainter, use qcolor_with_alpha() instead.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def qcolor_with_alpha(color: QColor, alpha: float) -> QColor:
    """Return a copy of a QColor with the given alpha. For QPainter usage.

    Args:
        color: Source QColor.
        alpha: Alpha from 0.0 to 1.0.

    Returns:
        New QColor with alpha applied.
    """
    c = QColor(color)
    c.setAlphaF(alpha)
    return c


# === Surfaces (layered depth, blue-black tones) ===
CANVAS = "#0a0e13"
CHROME = "#0b0f14"
BASE = "#0f1419"
SURFACE_1 = "#111820"
SURFACE_2 = "#151c25"
BORDER = "#1e2530"

# === Text ===
TEXT_PRIMARY = "#e2e8f0"
TEXT_SECONDARY = "#c0c8d4"
TEXT_TERTIARY = "#94a3b8"
TEXT_MUTED = "#718096"
TEXT_DISABLED = "#475569"

# === Accent (cyan) ===
ACCENT = "#38bdf8"
ACCENT_HOVER = "#0ea5e9"
ACCENT_PRESSED = "#0284c7"

# === Status (semantic) ===
SUCCESS = "#34d399"
WARNING = "#fbbf24"
ERROR = "#f87171"
INFO = "#60a5fa"

# === Semantic state (kept deliberately apart from ACCENT) ===
#
# Used by state chips that say what a thing *is* -- a plugin row's Enabled /
# Not compatible / Install failed pill, for example. Held separate from the
# cyan ACCENT on purpose: the accent marks the primary *action*, so if
# "needs attention" were painted in it too, the two would compete and neither
# would read at a glance. Slightly brighter than the SUCCESS/WARNING/ERROR
# above, which are tuned for thin strokes and small text rather than for
# coloured text sitting on a tinted chip.
STATE_OK = "#4ade80"
STATE_WARN = "#fbbf24"
STATE_ERR = "#f87171"

# === Node Categories ===
CAT_HARDWARE = "#1a4a2e"
CAT_LOGIC = "#1e3a5f"
CAT_INTERFACE = "#4a3a1a"
CAT_SCRIPT = "#3d1a5f"
CAT_DEFAULT = "#2a2a35"

# Category gradients: (from, to) for QLinearGradient
CAT_HARDWARE_GRADIENT = ("#1a4a2e", "#163d26")
CAT_LOGIC_GRADIENT = ("#1e3a5f", "#1a2d47")
CAT_INTERFACE_GRADIENT = ("#4a3a1a", "#3d3018")
CAT_SCRIPT_GRADIENT = ("#3d1a5f", "#301447")
CAT_DEFAULT_GRADIENT = ("#2a2a35", "#22222c")

# === Node Graph ===
NODE_BODY = "#151c25"
NODE_HEADER_TEXT = "#e2e8f0"
NODE_PORT_LABEL = "#c0c8d4"
NODE_SELECTED = "#38bdf8"

# === Ports & Connections ===
PORT_DATA = "#38bdf8"
PORT_EXEC = "#e2e8f0"
CONN_ACTIVE = "#34d399"
CONN_SELECTED = "#38bdf8"

# === Special ===
RECORDING = "#f87171"
LED_ON = "#34d399"
LED_OFF = "#718096"
INPUT_VALUE = "#34d399"

# === Node Library Category Colors ===
LIB_FLOW = "#1e3a5f"
LIB_FUNCTIONS = "#1a4a4a"
LIB_CONTROL = "#4a3a1a"
LIB_IO = "#1a4a2e"
LIB_AUDIO = "#3d1a5f"
LIB_VIDEO = "#1e3a5f"
LIB_ZONES = "#4a3a1a"
LIB_PLUGINS = "#1a3a4a"
LIB_BEHAVIOR = "#5f1a3a"

# === Behavior State Colors (CV-specific, not theme colors) ===
BEHAVIOR_FREEZE = "#0000FF"
BEHAVIOR_IMMOBILE = "#FFFF00"
BEHAVIOR_MOVING = "#00FF00"
BEHAVIOR_DARTING = "#FF0000"

# === QColor versions for QPainter usage ===
Q_CANVAS = QColor(CANVAS)
Q_CHROME = QColor(CHROME)
Q_BASE = QColor(BASE)
Q_SURFACE_1 = QColor(SURFACE_1)
Q_SURFACE_2 = QColor(SURFACE_2)
Q_BORDER = QColor(BORDER)
Q_TEXT_PRIMARY = QColor(TEXT_PRIMARY)
Q_TEXT_SECONDARY = QColor(TEXT_SECONDARY)
Q_TEXT_TERTIARY = QColor(TEXT_TERTIARY)
Q_TEXT_MUTED = QColor(TEXT_MUTED)
Q_TEXT_DISABLED = QColor(TEXT_DISABLED)
Q_ACCENT = QColor(ACCENT)
Q_ACCENT_HOVER = QColor(ACCENT_HOVER)
Q_ACCENT_PRESSED = QColor(ACCENT_PRESSED)
Q_SUCCESS = QColor(SUCCESS)
Q_WARNING = QColor(WARNING)
Q_ERROR = QColor(ERROR)
Q_INFO = QColor(INFO)
Q_CAT_HARDWARE = QColor(CAT_HARDWARE)
Q_CAT_LOGIC = QColor(CAT_LOGIC)
Q_CAT_INTERFACE = QColor(CAT_INTERFACE)
Q_CAT_SCRIPT = QColor(CAT_SCRIPT)
Q_CAT_DEFAULT = QColor(CAT_DEFAULT)
Q_NODE_BODY = QColor(NODE_BODY)
Q_NODE_HEADER_TEXT = QColor(NODE_HEADER_TEXT)
Q_NODE_PORT_LABEL = QColor(NODE_PORT_LABEL)
Q_NODE_SELECTED = QColor(NODE_SELECTED)
Q_PORT_DATA = QColor(PORT_DATA)
Q_PORT_EXEC = QColor(PORT_EXEC)
Q_CONN_ACTIVE = QColor(CONN_ACTIVE)
Q_CONN_SELECTED = QColor(CONN_SELECTED)
Q_RECORDING = QColor(RECORDING)
Q_LED_ON = QColor(LED_ON)
Q_LED_OFF = QColor(LED_OFF)
Q_INPUT_VALUE = QColor(INPUT_VALUE)
