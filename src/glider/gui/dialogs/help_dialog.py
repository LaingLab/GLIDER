"""
GLIDER Help Dialog

Comprehensive in-app help with tabbed content covering all GLIDER features.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

STYLE = """
QDialog {
    background-color: #1a1a2e;
}
QTabWidget::pane {
    border: 1px solid #3498db;
    background-color: #1a1a2e;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #16213e;
    color: #cccccc;
    padding: 8px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #1a1a2e;
    color: white;
    border-bottom: 2px solid #3498db;
}
QTabBar::tab:hover {
    background-color: #1f2b47;
}
QScrollArea {
    border: none;
    background-color: #1a1a2e;
}
QLabel {
    color: #e0e0e0;
}
QPushButton {
    background-color: #16213e;
    color: white;
    border: 1px solid #3498db;
    padding: 6px 20px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #1f2b47;
}
"""

CONTENT_STYLE = """
body {
    color: #e0e0e0;
    font-family: sans-serif;
    font-size: 14px;
    line-height: 1.5;
}
h2 {
    color: #3498db;
    border-bottom: 1px solid #2a2a4e;
    padding-bottom: 4px;
}
h3 { color: #5dade2; }
code {
    background-color: #3c3c3c;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: monospace;
    color: #f0c040;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
}
th {
    background-color: #16213e;
    color: #3498db;
    padding: 8px;
    text-align: left;
    border: 1px solid #2a2a4e;
}
td {
    padding: 6px 8px;
    border: 1px solid #2a2a4e;
}
tr:nth-child(even) { background-color: #16213e; }
ul, ol { padding-left: 20px; }
li { margin-bottom: 4px; }
"""


def _wrap_html(body: str) -> str:
    return f"<html><head><style>{CONTENT_STYLE}</style></head><body>{body}</body></html>"


def _make_scroll_tab(html: str) -> QWidget:
    """Create a scrollable tab with HTML content."""
    label = QLabel()
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignTop)
    label.setContentsMargins(16, 16, 16, 16)
    label.setText(_wrap_html(html))
    label.setOpenExternalLinks(True)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(label)

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(scroll)
    return container


# ── Tab content ──────────────────────────────────────────────────────────


_GETTING_STARTED = """
<h2>Welcome to GLIDER</h2>
<p>
<b>GLIDER</b> (General Laboratory Interface for Design, Experimentation, and
Recording) is a modular experimental orchestration platform for laboratory
hardware control through visual flow-based programming. It supports Arduino
(via Telemetrix), Raspberry Pi (via gpiozero), computer vision for
tracking and behavioral analysis, and an AI assistant.
</p>

<h3>Two Modes</h3>
<table>
<tr><th>Mode</th><th>Description</th><th>Launch</th></tr>
<tr>
  <td><b>Builder</b></td>
  <td>Full desktop IDE with node graph editor, property panels, and dock
      widgets for designing experiments.</td>
  <td><code>glider --builder</code></td>
</tr>
<tr>
  <td><b>Runner</b></td>
  <td>Touch-optimized dashboard showing only the controls and displays you
      expose. Ideal for running experiments on small screens or tablets.</td>
  <td><code>glider --runner</code></td>
</tr>
</table>
<p>With no flag the app auto-detects: small/touch screens default to Runner,
desktops default to Builder.</p>

<h3>Basic Workflow</h3>
<ol>
<li><b>Add a board</b> &mdash; Hardware menu &rarr; Add Board (select board type
    and serial port).</li>
<li><b>Add devices</b> &mdash; Hardware menu &rarr; Add Device (pick board, pin,
    and device type such as Digital Output or Analog Input).</li>
<li><b>Build a flow</b> &mdash; Drag nodes from the Node Library onto the graph,
    connect them, and bind hardware devices to hardware nodes.</li>
<li><b>Run the experiment</b> &mdash; Press <b>F5</b> (or the Start button) to
    execute the flow.</li>
</ol>

<h3>Command-Line Options</h3>
<table>
<tr><th>Command</th><th>Description</th></tr>
<tr><td><code>glider</code></td><td>Launch with auto-detected mode</td></tr>
<tr><td><code>glider --builder</code></td><td>Force Builder (desktop IDE) mode</td></tr>
<tr><td><code>glider --runner</code></td><td>Force Runner (touch dashboard) mode</td></tr>
<tr><td><code>glider --debug</code></td><td>Enable debug logging</td></tr>
<tr><td><code>glider --file &lt;path&gt;</code></td>
    <td>Open an experiment file on launch</td></tr>
</table>
"""

_HARDWARE_SETUP = """
<h2>Hardware Setup</h2>

<h3>Supported Boards</h3>
<table>
<tr><th>Board Type</th><th>Backend</th><th>Notes</th></tr>
<tr>
  <td><b>Arduino</b></td>
  <td>Telemetrix</td>
  <td>Requires Telemetrix4Arduino firmware on the board.
      Connect via USB serial port.</td>
</tr>
<tr>
  <td><b>Raspberry Pi</b></td>
  <td>gpiozero</td>
  <td>Uses GPIO pins directly on the Pi running GLIDER.</td>
</tr>
<tr>
  <td><b>Mock Board</b></td>
  <td>Software</td>
  <td>Simulated board for testing &mdash; no real hardware needed.</td>
</tr>
</table>

<h3>Adding a Board</h3>
<ol>
<li>Go to <b>Hardware</b> menu &rarr; <b>Add Board</b>.</li>
<li>Select the board type (Arduino, Raspberry Pi, or Mock).</li>
<li>For Arduino: choose the serial port (e.g. <code>COM3</code> on Windows,
    <code>/dev/ttyUSB0</code> on Linux).</li>
<li>Click <b>Connect</b> to establish the connection.</li>
</ol>

<h3>Device Types</h3>
<table>
<tr><th>Device</th><th>Description</th></tr>
<tr><td><b>Digital Output</b></td><td>On/off control (LEDs, relays, solenoids)</td></tr>
<tr><td><b>Digital Input</b></td><td>Read high/low state (buttons, switches, sensors)</td></tr>
<tr><td><b>Analog Input</b></td><td>Read voltage (0&ndash;1023 on Arduino)</td></tr>
<tr><td><b>PWM Output</b></td><td>Variable duty-cycle output (motor speed, LED brightness)</td></tr>
<tr><td><b>Servo</b></td><td>Angular position control (0&ndash;180&deg;)</td></tr>
<tr><td><b>I2C (ADS1115)</b></td><td>High-precision analog input via I2C ADC</td></tr>
</table>

<h3>Adding a Device</h3>
<ol>
<li>Go to <b>Hardware</b> menu &rarr; <b>Add Device</b>.</li>
<li>Select the parent board.</li>
<li>Choose the device type and pin number.</li>
<li>Give the device a descriptive name (e.g. &ldquo;LED_13&rdquo;).</li>
</ol>

<h3>Pin Conflicts</h3>
<p>GLIDER tracks pin usage and will warn you if you try to assign two devices
to the same pin. Resolve conflicts before connecting.</p>
"""

_NODE_GRAPH = """
<h2>Node Graph</h2>
<p>The node graph is GLIDER's visual flow programming environment. You build
experiments by placing nodes on a canvas, connecting their ports, and binding
hardware devices to hardware nodes.</p>

<h3>Port Types</h3>
<table>
<tr><th>Type</th><th>Color</th><th>Behavior</th></tr>
<tr>
  <td><b>DATA</b></td><td>Blue</td>
  <td>Passes values between nodes reactively &mdash; downstream nodes update
      whenever an upstream value changes.</td>
</tr>
<tr>
  <td><b>EXEC</b></td><td>White</td>
  <td>Controls execution order &mdash; nodes fire sequentially along EXEC
      connections (imperative flow).</td>
</tr>
</table>

<h3>Node Categories</h3>

<h4>Experiment</h4>
<p>Core flow control: <b>Start</b>, <b>End</b>, <b>Delay</b>, <b>Output</b>,
<b>Input</b>, <b>Loop</b>, <b>WaitForInput</b>.</p>

<h4>Hardware</h4>
<p>Interact with physical devices: <b>Digital Write</b>, <b>Digital Read</b>,
<b>Analog Read</b>, <b>PWM Write</b>, <b>Device Action</b>,
<b>Device Read</b>.</p>

<h4>Logic &mdash; Math</h4>
<p><b>Add</b>, <b>Subtract</b>, <b>Multiply</b>, <b>Divide</b>,
<b>MapRange</b>, <b>Clamp</b>.</p>

<h4>Logic &mdash; Comparison</h4>
<p><b>Threshold</b> (above/below), <b>InRange</b> (min/max).</p>

<h4>Logic &mdash; Control</h4>
<p><b>PID Controller</b>, <b>Toggle</b>, <b>Sequence</b>, <b>Timer</b>.</p>

<h4>Interface &mdash; Input</h4>
<p>Dashboard controls exposed at run time: <b>Button</b>,
<b>Toggle Switch</b>, <b>Slider</b>, <b>Numeric Input</b>.</p>

<h4>Interface &mdash; Display</h4>
<p>Dashboard readouts: <b>Label</b>, <b>Gauge</b>, <b>Chart</b>,
<b>LED Indicator</b>.</p>

<h4>Vision</h4>
<p>Camera zone events: <b>Zone Occupied</b>, <b>Object Count</b>,
<b>On Enter</b>, <b>On Exit</b>.</p>

<h4>Flow Functions</h4>
<p>Reusable sub-graphs: <b>StartFunction</b>, <b>EndFunction</b>,
<b>FunctionCall</b>.</p>

<h3>Connecting Nodes</h3>
<ol>
<li>Click and drag from an <b>output port</b> on one node.</li>
<li>Drop onto a compatible <b>input port</b> on another node.</li>
<li>DATA ports connect to DATA ports; EXEC ports connect to EXEC ports.</li>
</ol>

<h3>Binding Devices</h3>
<p>Select a hardware node, then use the <b>Properties</b> panel on the right
to pick which device it controls. Only devices of the correct type are
shown.</p>
"""

_RUNNING_EXPERIMENTS = """
<h2>Running Experiments</h2>

<h3>Experiment States</h3>
<table>
<tr><th>State</th><th>Description</th></tr>
<tr><td>IDLE</td><td>No experiment loaded or ready.</td></tr>
<tr><td>INITIALIZING</td><td>Hardware and flow are being prepared.</td></tr>
<tr><td>READY</td><td>Experiment is loaded and ready to start.</td></tr>
<tr><td>RUNNING</td><td>Flow is actively executing.</td></tr>
<tr><td>PAUSED</td><td>Execution is paused (can resume).</td></tr>
<tr><td>STOPPING</td><td>Experiment is shutting down.</td></tr>
<tr><td>ERROR</td><td>An error occurred during execution.</td></tr>
</table>

<h3>Controls</h3>
<table>
<tr><th>Action</th><th>Shortcut</th><th>Description</th></tr>
<tr><td><b>Start</b></td><td><code>F5</code></td>
    <td>Begin executing the experiment flow.</td></tr>
<tr><td><b>Stop</b></td><td><code>Shift+F5</code></td>
    <td>Stop the running experiment gracefully.</td></tr>
<tr><td><b>Emergency Stop</b></td><td><code>Ctrl+Shift+Escape</code></td>
    <td>Immediately halt all hardware and stop execution.</td></tr>
</table>

<h3>Data Recording</h3>
<p>GLIDER records hardware measurements to <b>CSV files</b> automatically
during an experiment run. Use the <b>Output</b> node to log specific
values. Recorded data can be analyzed with the built-in Analysis dialog
or the AI Agent.</p>

<h3>Saving &amp; Loading</h3>
<p>Experiments are saved as <code>.glider</code> files (JSON format) containing
hardware configuration, the flow graph, dashboard layout, camera settings,
zones, and subject information.</p>
<ul>
<li><b>Ctrl+S</b> &mdash; Save</li>
<li><b>Ctrl+Shift+S</b> &mdash; Save As</li>
<li><b>Ctrl+O</b> &mdash; Open</li>
<li><b>Ctrl+N</b> &mdash; New experiment</li>
</ul>

<h3>Runner Mode Dashboard</h3>
<p>In Runner mode, only the controls and displays you expose via Interface
nodes appear. The dashboard is touch-friendly with large buttons and
readouts, ideal for running pre-built experiments on small screens.</p>
"""

_CAMERA_VISION = """
<h2>Camera &amp; Vision</h2>

<h3>Connecting Cameras</h3>
<p>Open the <b>Camera</b> panel (View menu or dock widget) and click
<b>Add Camera</b>. GLIDER supports USB webcams, IP cameras, and
video files. Multiple cameras can be active simultaneously.</p>

<h3>Detection Backends</h3>
<table>
<tr><th>Backend</th><th>Description</th></tr>
<tr><td><b>Background Subtraction</b></td>
    <td>Fast, lightweight detection using frame differencing. Good for
        controlled environments.</td></tr>
<tr><td><b>YOLOv8</b></td>
    <td>Deep-learning object detection via Ultralytics. Accurate but
        requires more processing power.</td></tr>
<tr><td><b>ByteTrack</b></td>
    <td>Multi-object tracker that maintains persistent IDs across frames.</td></tr>
<tr><td><b>Motion Only</b></td>
    <td>Detects movement without identifying individual objects.</td></tr>
</table>

<h3>Object Tracking</h3>
<p>When using a detection backend, GLIDER assigns persistent IDs to tracked
objects so you can follow individuals across frames. Tracking data
(position, speed) is available to vision nodes.</p>

<h3>Zone System</h3>
<p>Draw rectangular or polygonal zones on the camera view. Each zone
can trigger events:</p>
<ul>
<li><b>Entry</b> &mdash; fires when an object enters the zone</li>
<li><b>Exit</b> &mdash; fires when an object leaves the zone</li>
<li><b>Dwell</b> &mdash; fires after an object stays in the zone for a
    configured duration</li>
<li><b>Occupied</b> / <b>Object Count</b> &mdash; continuous data about
    zone contents</li>
</ul>
<p>Connect zone events to Vision nodes in the flow graph.</p>

<h3>Behavioral Analysis</h3>
<p>GLIDER can classify tracked-object behavior into categories:</p>
<ul>
<li><b>Freeze</b> &mdash; no movement detected</li>
<li><b>Immobile</b> &mdash; very little movement</li>
<li><b>Locomotion</b> &mdash; normal movement</li>
<li><b>Darting</b> &mdash; rapid movement bursts</li>
</ul>

<h3>Video Recording</h3>
<p>Record camera feeds with optional overlays (zones, tracking IDs, timestamps).
Recordings are saved alongside experiment data.</p>

<h3>Calibration</h3>
<p>Use the Calibration dialog to map pixel distances to physical units
(e.g. centimeters). This enables real-world measurements in zone and
tracking data.</p>
"""

_AI_AGENT = """
<h2>AI Agent</h2>
<p>GLIDER includes an AI assistant that can help you build and run
experiments using natural language.</p>

<h3>Supported Providers</h3>
<table>
<tr><th>Provider</th><th>Details</th></tr>
<tr><td><b>Ollama</b></td>
    <td>Run models locally for free. Requires Ollama installed and running
        on your machine (<code>http://localhost:11434</code>).</td></tr>
<tr><td><b>OpenAI</b></td>
    <td>Cloud-hosted models. Requires an API key.</td></tr>
</table>

<h3>Configuration</h3>
<p>Open the Agent panel and click the <b>Settings</b> (gear) icon, or go to
the Agent Settings dialog. Choose your provider, model, and generation
parameters.</p>

<h3>What the Agent Can Do</h3>
<ul>
<li><b>Add boards and devices</b> &mdash; &ldquo;Add an Arduino on COM3 with
    an LED on pin 13.&rdquo;</li>
<li><b>Configure hardware</b> &mdash; &ldquo;Set the PWM frequency to
    1000 Hz.&rdquo;</li>
<li><b>Control experiments</b> &mdash; &ldquo;Start the experiment&rdquo;,
    &ldquo;Stop recording.&rdquo;</li>
<li><b>Explain the flow</b> &mdash; &ldquo;What does this experiment
    do?&rdquo;</li>
</ul>

<h3>Data Analysis</h3>
<p>Load a CSV file from the Analysis dialog and ask natural language
questions about your data, e.g. &ldquo;What was the average temperature
during the first 5 minutes?&rdquo;</p>
"""

_KEYBOARD_SHORTCUTS = """
<h2>Keyboard Shortcuts</h2>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><code>Ctrl+N</code></td><td>New experiment</td></tr>
<tr><td><code>Ctrl+O</code></td><td>Open experiment file</td></tr>
<tr><td><code>Ctrl+S</code></td><td>Save experiment</td></tr>
<tr><td><code>Ctrl+Shift+S</code></td><td>Save As</td></tr>
<tr><td><code>Ctrl+Q</code></td><td>Quit GLIDER</td></tr>
<tr><td colspan="2" style="border:none;">&nbsp;</td></tr>
<tr><td><code>F5</code></td><td>Start experiment</td></tr>
<tr><td><code>Shift+F5</code></td><td>Stop experiment</td></tr>
<tr><td><code>Ctrl+Shift+Escape</code></td><td>Emergency stop</td></tr>
<tr><td colspan="2" style="border:none;">&nbsp;</td></tr>
<tr><td><code>Ctrl+Z</code></td><td>Undo</td></tr>
<tr><td><code>Ctrl+Y</code></td><td>Redo</td></tr>
<tr><td><code>Delete</code></td><td>Delete selected node or connection</td></tr>
<tr><td colspan="2" style="border:none;">&nbsp;</td></tr>
<tr><td><code>F1</code></td><td>Open this Help dialog</td></tr>
<tr><td><code>F11</code></td><td>Toggle fullscreen</td></tr>
</table>
"""

_TROUBLESHOOTING = """
<h2>Troubleshooting</h2>

<h3>Board Won't Connect</h3>
<ul>
<li>Check that the correct serial port is selected (e.g.
    <code>COM3</code>, <code>/dev/ttyUSB0</code>).</li>
<li>Ensure <b>Telemetrix4Arduino</b> firmware is flashed on your Arduino.</li>
<li>Try unplugging and reconnecting the USB cable.</li>
<li>On Linux, make sure your user is in the <code>dialout</code> group.</li>
</ul>

<h3>Camera Not Detected</h3>
<ul>
<li>Verify the camera is plugged in and recognized by the OS.</li>
<li>Try a different camera index (0, 1, 2&hellip;).</li>
<li>Install vision extras: <code>pip install "glider[vision]"</code></li>
</ul>

<h3>Nodes Not Executing</h3>
<ul>
<li>Make sure your flow has a <b>Start</b> node.</li>
<li>Check that <b>EXEC</b> (white) ports are connected in the correct order.</li>
<li>Verify hardware nodes have a device assigned in the Properties panel.</li>
</ul>

<h3>Missing Optional Dependencies</h3>
<table>
<tr><th>Feature</th><th>Install Command</th></tr>
<tr><td>Computer vision</td><td><code>pip install "glider[vision]"</code></td></tr>
<tr><td>I2C devices (ADS1115)</td><td><code>pip install "glider[i2c]"</code></td></tr>
<tr><td>All PC extras</td><td><code>pip install "glider[pc]"</code></td></tr>
<tr><td>Development tools</td><td><code>pip install "glider[dev]"</code></td></tr>
</table>

<h3>Touch / Runner Mode Issues</h3>
<ul>
<li>Force runner mode with <code>glider --runner</code>.</li>
<li>Make sure your display resolution is detected correctly.</li>
<li>Use the on-screen menu button (top-left) for navigation.</li>
</ul>
"""


class HelpDialog(QDialog):
    """Comprehensive help dialog for GLIDER."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("GLIDER Help")
        self.setMinimumSize(800, 600)
        self.resize(800, 600)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(_make_scroll_tab(_GETTING_STARTED), "Getting Started")
        tabs.addTab(_make_scroll_tab(_HARDWARE_SETUP), "Hardware Setup")
        tabs.addTab(_make_scroll_tab(_NODE_GRAPH), "Node Graph")
        tabs.addTab(_make_scroll_tab(_RUNNING_EXPERIMENTS), "Running Experiments")
        tabs.addTab(_make_scroll_tab(_CAMERA_VISION), "Camera & Vision")
        tabs.addTab(_make_scroll_tab(_AI_AGENT), "AI Agent")
        tabs.addTab(_make_scroll_tab(_KEYBOARD_SHORTCUTS), "Shortcuts")
        tabs.addTab(_make_scroll_tab(_TROUBLESHOOTING), "Troubleshooting")
        layout.addWidget(tabs)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
