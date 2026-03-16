# GLIDER: Comprehensive Design Specification for Modular Experimental Orchestration

## 1. Introduction and System Philosophy

The contemporary laboratory environment is increasingly defined by the integration of heterogeneous hardware systems, ranging from commodity microcontrollers like Arduino and Raspberry Pi to specialized industrial actuators and sensors. While Python has established itself as the lingua franca of scientific computing due to its rich ecosystem of data analysis libraries, the domain of experimental orchestration---specifically the real-time control of hardware coupled with complex logic flows---remains fragmented. Researchers often resort to monolithic, brittle scripts that are difficult to modify, impossible to scale, and challenging to debug. The GLIDER (General Laboratory Interface for Design, Experimentation, and Recording) project addresses this systemic inefficiency by providing a robust, open-source software architecture that prioritizes modularity, scalability, and user accessibility.

This design document articulates the implemented architecture for GLIDER, a system-requirements-compliant application capable of operating across a spectrum of computing environments, from high-performance desktop workstations to resource-constrained embedded systems like the Raspberry Pi. The core philosophy driving GLIDER's design is the strict separation of concerns between the logical definition of an experiment, the physical actuation of hardware, and the graphical presentation of state. This separation allows the software to serve two distinct but interrelated roles: a sophisticated Integrated Development Environment (IDE) for constructing experimental flows (Builder mode), and a streamlined, touch-optimized runtime interface for execution in the field (Runner mode).

GLIDER adopts a split-core architecture utilizing **PyQt6** for the graphical shell, **ryvencore** for the underlying flow-based logic engine, and **asyncio** (integrated via **qasync**) for managing the high-concurrency demands of hardware I/O without blocking the user interface. Beyond its original design scope, the implemented system now includes a comprehensive **computer vision pipeline** for real-time object tracking and behavioral analysis, a **multi-camera management** system, an **LLM-powered AI agent** for programmatic experiment construction, a **custom device definition** system, reusable **flow functions**, and an **audio/video playback** framework. This document details the technical specifications, design patterns, and implementation strategies that realize this vision.

### 1.1 The Role of Python in Embedded Orchestration

Python's dominance in the embedded sector has grown significantly, transitioning from a scripting language to a primary driver for hardware control. The availability of libraries such as RPi.GPIO and gpiozero has lowered the barrier to entry for physical computing. However, the direct utilization of these libraries within a graphical application often leads to "spaghetti code" where UI logic is tightly coupled with hardware timing, resulting in interfaces that freeze during sensor reads or motor movements.

GLIDER addresses this by treating Python not just as the glue code, but as the hosting environment for a sophisticated event loop that arbitrates between the user's intent and the hardware's capabilities. By leveraging the advanced features of Python 3.8+, specifically Abstract Base Classes (ABCs) and asynchronous coroutines, GLIDER ensures that hardware drivers are swappable plugins rather than hardcoded dependencies. This allows a researcher to prototype an experiment using an Arduino Uno and seamlessly migrate to a robust industrial controller by simply changing the driver selection in the software, preserving the high-level logic flow.

### 1.2 Addressing the Dual-Interface Requirement

A unique constraint of the GLIDER specification is the necessity to support both a standard desktop environment and a 480x800 vertical touchscreen interface on a Raspberry Pi. This requirement fundamentally shapes the UI/UX architecture. Desktop interfaces benefit from high information density, multi-window workflows, and precise mouse interaction. In contrast, a vertical touch interface on a Raspberry Pi (often used in "kiosk mode") demands large hit targets, simplified navigation hierarchies, and high-contrast visuals for readability in lab environments.

The implemented design utilizes a responsive **ViewManager** strategy. The core application logic remains invariant, but the presentation layer creates distinct view hierarchies based on the detected runtime environment. The ViewManager auto-detects display properties at startup, using threshold values of 480x800 pixels to distinguish Runner mode from Desktop mode. This avoids the maintenance burden of developing two separate applications while ensuring that the Runner experience on the Pi is not merely a shrunken version of the Desktop Builder, but a purpose-built interface for experimental execution.

---

## 2. Architectural Overview and Design Patterns

The architecture of GLIDER is predicated on the Model-View-Controller (MVC) pattern, adapted for asynchronous event-driven programming. This structure is essential to decouple the high-frequency events of hardware sensors from the rendering cycle of the Graphical User Interface (GUI).

### 2.1 The Split-Core Architecture

To ensure scalability and testability, GLIDER comprises two primary subsystems: the **GLIDER Core** and the **GLIDER Shell**.

The **GLIDER Core** is a headless, pure-Python library responsible for the lifecycle management of experiments. It loads hardware drivers, maintains the state of the "Flow" (the logical graph of the experiment), manages data logging, coordinates computer vision processing, and executes the event loop. Crucially, the Core has no dependencies on PyQt6 or any display libraries. This design decision ensures that the Core can be run as a system service (daemon) on the Raspberry Pi, executing experiments autonomously even if the GUI process terminates or if the device is running in a headless configuration.

The **GLIDER Shell** is the PyQt6 application that provides the visual interface. It acts as the "View" and "Controller" in the MVC paradigm. The Shell communicates with the Core via a strict API facade. When a user drags a node onto the canvas in the Shell, a command is sent to the Core to instantiate the corresponding logic object. When the Core receives new sensor data, it emits a signal that the Shell consumes to update a label or plot. This decoupling is vital for stability; an unhandled exception in the GUI rendering logic should not crash the ongoing physical experiment managed by the Core.

### 2.2 Concurrency Model: The Event Loop

Traditional hardware control software often relies on multi-threading to handle simultaneous tasks (e.g., reading a sensor while spinning a motor). However, Python's Global Interpreter Lock (GIL) limits true parallelism, and managing thread safety---avoiding race conditions and deadlocks---adds significant complexity. Furthermore, hardware operations are typically I/O-bound (waiting for a serial response) rather than CPU-bound, making them ideal candidates for cooperative multitasking.

GLIDER employs Python's **asyncio** library as its primary concurrency model. By defining hardware interactions as non-blocking coroutines (`async def`), the system can handle hundreds of simultaneous device inputs on a single thread without blocking. To integrate this with the PyQt6 GUI, which has its own event loop, GLIDER utilizes the **qasync** library. qasync is an implementation of the asyncio event loop that runs on top of the Qt event loop. This allows Qt signals/slots and asyncio futures/tasks to coexist seamlessly. A button click in PyQt can trigger an async function that communicates with an Arduino, waits for a response, and updates the UI, all without freezing the interface.

GLIDER also employs dedicated background threads where necessary. The **TelemetrixThread** runs the Telemetrix-AIO library in its own event loop, isolating Arduino serial communication from the Qt/asyncio main loop. Cross-thread calls are marshalled via `asyncio.run_coroutine_threadsafe()`, and pin-change callbacks are dispatched back to the main loop via `call_soon_threadsafe()`. Similarly, the **FrameWriterThread** decouples video disk I/O from camera capture using a queue-based asynchronous pattern.

### 2.3 Modularity and Plugin System

To satisfy the requirement for custom boards and devices, GLIDER implements a strict plugin architecture. The Core does not contain hardcoded support for specific hardware. Instead, it defines Abstract Base Classes (ABCs) that specify the contract a driver must fulfill (e.g., `write_digital`, `read_analog`).

Plugins are Python packages managed via `importlib` and `importlib.metadata`. At startup, the GLIDER Plugin Manager scans for installed packages advertising the `glider.driver` entry point in `pyproject.toml`. Built-in drivers (Arduino Telemetrix, Raspberry Pi GPIO) are also registered at import time via `HardwareManager.register_driver()`. This allows researchers to distribute drivers for their custom hardware as standalone Python packages that GLIDER automatically detects and integrates into the UI.

### 2.4 Data Flow vs. Execution Flow

Inspired by visual programming environments like Unreal Engine Blueprints and the underlying mechanics of Ryven, GLIDER distinguishes between two types of signal propagation within the experiment graph:

**Data Flow:** This represents the continuous propagation of state. When a sensor node updates its value, that data flows downstream to any connected nodes (e.g., a "Threshold" check node). This is reactive programming; the graph re-evaluates automatically when inputs change. In the implementation, `DataNode` subclasses implement a `process()` method that computes outputs from inputs, triggered whenever an input changes via `update_event()`.

**Execution Flow:** This represents the imperative sequence of actions. Nodes can have "Exec" inputs and outputs (visualized as distinct white triangular ports). An "Action" node (e.g., "Output") performs its task only when it receives an Execution signal, regardless of the state of its data inputs. In the implementation, `ExecNode` subclasses implement an async `execute()` method. The `_fire_exec_output()` method triggers downstream execution by invoking registered callbacks, which the FlowEngine manages via `_propagate_execution()` using `asyncio.create_task()`. This distinction is critical for defining experimental protocols (e.g., "Wait 10 seconds, THEN turn on the pump") which cannot be modeled purely with reactive data flow.

### 2.5 Custom Device System

Beyond the built-in device types, GLIDER implements a **Custom Device Definition** system that allows users to create new device abstractions without writing Python code. A `CustomDeviceDefinition` specifies named pins with types (DIGITAL_OUTPUT, DIGITAL_INPUT, ANALOG_INPUT, PWM, etc.), and a `CustomDeviceRunner` executes pin-level operations generically at runtime. Custom devices are serialized as JSON and can be exported/imported via the **Device Library** system (`.gdevice` files). This Strategy pattern enables device behavior to be defined via configuration rather than code.

### 2.6 Flow Functions (Reusable Subgraphs)

GLIDER supports **Flow Functions**---reusable node subgraphs that encapsulate common experimental patterns. A `FlowFunctionDefinition` contains parameters, outputs, and an internal graph of nodes and connections. At runtime, a `FunctionCallNode` invokes its subgraph via a `FlowFunctionRunner`, which creates internal nodes, wires them, dispatches execution from a `StartFunctionNode` to an `EndFunctionNode`, and signals completion via `asyncio.Event`. Flow functions are exported as `.gflow` files and can be shared across experiments via the library system. Multiple definitions can be bundled into `.glibrary` archives.

---

## 3. Graphical User Interface (GUI) Strategy

The selection of the GUI framework is a pivotal design decision. The requirements entail a sophisticated node-graph editor for the creation phase and a touch-optimized dashboard for the execution phase.

### 3.1 Framework Selection: PyQt6

While frameworks like Kivy are often recommended for touch interfaces due to their mobile-first design, they lack the mature desktop widget ecosystem required for a complex IDE-like interface. Tkinter is insufficient for modern, high-performance graphics and lacks built-in support for complex custom widgets like node graphs.

**PyQt6** (and the underlying Qt 6 framework) is the optimal choice for GLIDER for several reasons:

1. **The Graphics View Framework:** Qt provides `QGraphicsScene` and `QGraphicsView`, a highly optimized 2D rendering engine capable of handling thousands of interactive items (nodes, wires, ports). GLIDER implements a fully custom node graph editor using these primitives, with `NodeItem` (QGraphicsRectItem), `PortItem` (QGraphicsEllipseItem), and `ConnectionItem` (QGraphicsPathItem) classes providing professional-grade visual scripting.
2. **Hardware Acceleration:** Qt 6 leverages the GPU for rendering (via RHI---Rendering Hardware Interface). On the Raspberry Pi 4 and 5, this ensures that the UI renders at a smooth 60 FPS, which is critical for touch responsiveness.
3. **Stylesheets (QSS):** Qt Style Sheets allow the application's appearance to be radically transformed using a CSS-like syntax. GLIDER maintains two distinct stylesheets: `desktop.qss` (compact, mouse-oriented Builder theme) and `touch.qss` (large buttons, high contrast Runner theme). A "Deep Navy" dark color palette (`colors.py`) provides reduced eye strain with layered surface colors from `#0a0e13` (canvas) to `#151c25` (elevated surfaces).
4. **Cross-Platform Consistency:** PyQt6 ensures identical behavior on Windows, macOS, Linux, and Raspberry Pi OS, fulfilling the requirement for portability.

### 3.2 Responsive Interface Design

To accommodate the 480x800 vertical resolution of the Raspberry Pi screen, GLIDER employs a **ViewManager** that detects the display properties at startup.

**Table 1: UI Adaptation Strategy**

| Feature | Desktop Mode (Builder) | Runner Mode (Raspberry Pi 480x800) |
|---------|----------------------|-----------------------------------|
| **Window Layout** | QMainWindow with dockable QDockWidgets for Library, Properties, Hardware, Camera, and Device Control panels. Horizontal QSplitter for graph view. | Single-window with header, scrollable device cards, and fixed control buttons. No floating docks. |
| **Navigation** | Menu bars (File, Edit, Experiment, View, Hardware, Run, Help), detailed toolbars, keyboard shortcuts (Ctrl+Z undo, F fit-to-view, Home reset). | Header with experiment name, timer, and hamburger menu. Large START/STOP/EMERGENCY STOP buttons. |
| **Node Graph** | Full interactive editing: pan (middle mouse), zoom (0.1x-3.0x), drag-and-drop, rubber band selection, context menus, bezier curve connections. | Hidden. Focus is on device status cards and dashboard controls. |
| **Input Widgets** | Standard QSpinBox, QLineEdit, QComboBox, small checkboxes. Font sizes 10-14px. | Custom Touch Widgets: TouchButton (80px min-height), TouchToggle, TouchSlider (40px handle), TouchGauge, TouchNumericInput. Font sizes 16-24px. |
| **Scrolling** | Mouse wheel, 12px scrollbars. | Kinetic scrolling (QScroller), 40px wide touch-friendly scrollbars. |
| **Button Sizing** | 30x24px minimum. 8px padding. | 80x60px minimum. 20px padding. |

The application detects the screen resolution via `QScreen.size()`. If the width is detected as <= 480 pixels (or matches the 480x800 / 800x480 portrait/landscape signatures), the application automatically launches in Runner Mode, loading the `touch.qss` stylesheet that overrides widget padding, font sizes, and touch target dimensions.

### 3.3 The Node Graph Editor

The visual scripting interface is built using a fully custom implementation on top of Qt's Graphics View Framework. Unlike the original design specification which proposed using ryvencore-qt for frontend rendering, the implemented system uses custom `QGraphicsItem` subclasses for maximum control over the visual experience.

#### Core Graphics Components

- **NodeGraphView** (`QGraphicsView`): The main canvas supporting pan, zoom, drag-and-drop creation, manual connection drawing, rubber band selection, and keyboard navigation (Delete, Home, F for fit-all).
- **NodeGraphScene** (`QGraphicsScene`): Scene with 20px/100px grid background on the `#0a0e13` canvas color.
- **NodeItem** (`QGraphicsRectItem`): Visual node with category-colored gradient header (green for Hardware, blue for Logic, orange for Interface, purple for Script), input/output ports, and optional embedded widgets via `QGraphicsProxyWidget`. Minimum width 150px, 30px header, 24px per port row, 8px corner radius.
- **PortItem** (`QGraphicsEllipseItem`): Circular data ports (cyan `#38bdf8`) and triangular execution ports (light gray `#e2e8f0`) with hover effects and cursor changes.
- **ConnectionItem** (`QGraphicsPathItem`): Bezier curves with arrow heads on execution connections. Data connections render as thin cyan lines; execution connections as thick white lines. Active connections highlight green (`#34d399`).
- **TempConnectionItem**: Transient connection drawn during drag-connect operations.

#### Editor Features

- **Drag-and-Drop Library:** A `NodeLibraryPanel` side panel listing all available nodes organized by category (Flow, Functions, Control, I/O, Audio, Video, Custom Devices, Flow Functions, Zones). Nodes are represented by `DraggableNodeButton` widgets that emit MIME data on drag.
- **Inline Widgets:** Nodes can contain embedded Qt widgets (e.g., a value spinner inside a "Delay" node, a dropdown inside a "MotorGovernor" node) using `QGraphicsProxyWidget`, allowing users to adjust parameters directly on the graph.
- **Properties Panel:** Selecting a node dynamically populates a right-dock properties panel with type-specific controls (duration spinners, PWM value sliders, threshold inputs, mode selectors) managed by `NodeEditorController`.
- **Undo/Redo System:** All graph operations (create node, delete node, move node, create connection, delete connection, change property) are encapsulated as `Command` objects pushed onto an `UndoStack`, supporting full Ctrl+Z / Ctrl+Y operation.

### 3.4 The Builder Mode Layout

The Builder mode provides a full IDE experience:

```
MainWindow (QMainWindow)
+-- MenuBar (File, Edit, Experiment, View, Hardware, Run, Help)
+-- ToolBar (New, Open, Save, Connect, Start, Stop)
+-- Central: QStackedWidget
|   +-- Index 0: QSplitter -> NodeGraphView
|   +-- Index 1: RunnerPanel
+-- Left Dock (Tabified)
|   +-- NodeLibraryPanel (draggable node palette)
|   +-- HardwarePanel (board/device tree with add/edit/delete)
|   +-- DeviceControlPanel (manual device testing)
+-- Right Dock (Tabified)
|   +-- PropertiesPanel (dynamic node property editor)
|   +-- CameraPanel (live preview, CV settings, zone overlay)
+-- StatusBar
    +-- Connection indicator (colored dot + label)
    +-- Session state label
    +-- Node/connection count
```

### 3.5 The Runner Dashboard

The Runner Dashboard serves two purposes: it is the primary interface in Runner mode (Raspberry Pi), and it is accessible as a panel in Builder mode for testing.

#### RunnerPanel Layout
```
+-------------------------------+
| [Experiment Name] [Timer] [=] |  <- Header (50px, hamburger menu)
+-------------------------------+
| * REC                         |  <- Recording indicator (when active)
+-------------------------------+
| +---------------------------+ |
| | Device: Motor1            | |  <- Scrollable device cards
| | Type: PWM  Status: 128   | |     with real-time values
| +---------------------------+ |
| +---------------------------+ |
| | Device: Sensor1           | |
| | Type: Analog  V: 2.45    | |     Analog shows voltage
| +---------------------------+ |
+-------------------------------+
| [> START]         [# STOP]   |  <- Control buttons (60px)
| [!! EMERGENCY STOP]          |  <- Full-width emergency
+-------------------------------+
```

#### Dynamic Dashboard Widgets
The `RunnerDashboard` (in `runner/dashboard.py`) dynamically creates touch widgets for nodes marked `visible_in_runner=True`. A `WidgetFactory` maps node types to widget classes:

| Node Type | Widget | Description |
|-----------|--------|-------------|
| LabelNode | TouchLabel | 24pt display text |
| ButtonNode | TouchButton | 80px-tall press button |
| ToggleSwitchNode | TouchToggle | ON/OFF with green/gray coloring |
| SliderNode | TouchSlider | Range slider with 40px handle |
| GaugeNode | TouchGauge | Circular analog gauge |
| ChartNode | TouchChart | Time-series scrolling chart |
| LEDIndicatorNode | TouchLED | Red/green LED indicator |
| NumericInputNode | TouchNumericInput | Numeric entry with keypad |

Layout modes supported: `"vertical"`, `"horizontal"`, and `"grid"` (configurable columns). Kinetic scrolling is enabled via `QScroller` on the viewport.

---

## 4. The Flow Engine and Logic Core

The heart of GLIDER is the logic engine that executes the experiment. This engine must be robust, deterministic, and capable of handling complex dependencies.

### 4.1 Ryvencore Integration

GLIDER utilizes **ryvencore**, the backend library of Ryven, to manage the graph state. Ryven's architecture is uniquely suited for this project because it natively supports the concept of distinct "Data" and "Execution" flows, a feature often missing in simpler dataflow libraries.

The `FlowEngine` wraps ryvencore with a fallback to standalone mode if ryvencore is unavailable. This hybrid architecture provides resilience---the system functions without the ryvencore dependency. The engine maintains a class-level `_node_registry` (shared across all instances) for global node type availability, and a `FlowState` enum (`STOPPED`, `RUNNING`, `PAUSED`, `ERROR`) for state management.

In ryvencore, a Node is a Python class. The graph execution is managed by event propagation:

- **Data Updates:** When a `DataNode` calls `set_output(name, value)`, the engine pushes this value to connected inputs and triggers an `update_event()` on the downstream node, which invokes the node's `process()` method. This recursive propagation handles the reactive logic (e.g., Sensor -> Math -> Plot).
- **Execution Signals:** When an `ExecNode` calls `_fire_exec_output(output_name)`, the FlowEngine's registered callback creates an `asyncio.Task` via `_propagate_execution()`, which awaits the target node's `execute()` method. This allows for sequential imperative protocols (e.g., Start -> Delay -> Output -> End).
- **EndExperiment Detection:** When `_propagate_execution()` reaches a node whose definition name is `"EndExperiment"`, it calls `_notify_complete()` to signal experiment completion, transitioning the session back to READY state.

### 4.2 Custom Node Architecture

GLIDER extends node architecture through a class hierarchy rooted in `GliderNode(ABC)`:

```
GliderNode (ABC)
|-- DataNode         (reactive: inputs change -> process() -> outputs update)
|   +-- LogicNode    (math, comparison, control logic; blue #2d4a5a)
|
|-- ExecNode         (imperative: execute() called on exec signal)
|   +-- HardwareNode (device-bound operations; green #2d5a2d)
|
+-- InterfaceNode    (dashboard UI widgets; orange #5a4a2d; visible_in_runner=True)
```

Each node class declares a `NodeDefinition` containing:
- `name`: Display name
- `category`: `NodeCategory` enum (HARDWARE, LOGIC, INTERFACE)
- `description`: Documentation string
- `inputs`/`outputs`: Lists of `PortDefinition` objects specifying `name`, `port_type` (DATA or EXEC), `data_type`, `default_value`, and `description`
- `color`: Hex color for visual differentiation

**Key Node Categories and Implementations:**

#### Experiment Control Nodes
| Node | Type | Description |
|------|------|-------------|
| StartExperiment | GliderNode | Entry point; fires `next` exec output on experiment start |
| EndExperiment | GliderNode | Terminal node; signals experiment completion |
| Output | GliderNode | Writes a value to a bound device (digital HIGH/LOW or PWM 0-255) |
| Input | GliderNode | Reads a value from a bound device; outputs data and fires `next` |
| MotorGovernor | GliderNode | Controls motor governor device (up/down/stop with position feedback) |
| CustomDevice | GliderNode | Interfaces with user-defined custom devices via CustomDeviceRunner |

#### Logic and Control Nodes
| Node | Type | Description |
|------|------|-------------|
| DelayNode | ExecNode | Async delay via `await asyncio.sleep(seconds)` |
| LoopNode | ExecNode | Iterative execution with configurable count and inter-iteration delay |
| WaitForInputNode | ExecNode | Polls a device for edge detection (digital) or threshold crossing (analog) |
| SequenceNode | ExecNode | Fires up to 4 exec outputs in sequential order |
| TimerNode | ExecNode | Periodic timer with pause/resume and count output |
| PIDNode | DataNode | Full PID controller (Kp, Ki, Kd) with anti-windup |
| ToggleNode | DataNode | Binary state toggle with set_on/set_off control |
| ThresholdNode | DataNode | Hysteresis-enabled threshold comparison |
| InRangeNode | DataNode | Range check (min <= value <= max) |
| AddNode, SubtractNode, MultiplyNode, DivideNode | DataNode | Arithmetic operations |
| MapRangeNode | DataNode | Linear range mapping (in_min..in_max -> out_min..out_max) |
| ClampNode | DataNode | Value clamping between min and max |

#### Hardware I/O Nodes
| Node | Type | Description |
|------|------|-------------|
| DigitalWriteNode | HardwareNode | Write boolean to digital output pin |
| DigitalReadNode | HardwareNode | Read digital input with optional continuous polling |
| AnalogReadNode | HardwareNode | Read ADC value (raw, voltage, threshold), configurable resolution |
| PWMWriteNode | HardwareNode | Write PWM value (0-255) to output pin |
| DeviceActionNode | HardwareNode | Execute any named device action with arguments |
| DeviceReadNode | HardwareNode | Generic device read operation |

#### Interface (Dashboard) Nodes
| Node | Type | Description |
|------|------|-------------|
| LabelNode | InterfaceNode | Formatted text display with configurable format string |
| GaugeNode | InterfaceNode | Meter display with min/max/unit |
| ChartNode | InterfaceNode | Real-time scrolling chart with deque-based history |
| LEDIndicatorNode | InterfaceNode | On/off LED with configurable on/off colors |
| ButtonNode | ExecNode + InterfaceNode | Touch button that fires `Pressed` exec output (dual inheritance) |
| ToggleSwitchNode | InterfaceNode | Toggle switch with boolean state output |
| SliderNode | InterfaceNode | Range slider (min/max/step) |
| NumericInputNode | InterfaceNode | Numeric entry with optional keypad |

#### Audio and Video Nodes
| Node | Type | Description |
|------|------|-------------|
| AudioPlaybackNode | GliderNode | Plays WAV/MP3 files; uses `sounddevice`+`soundfile` or `pydub`; non-blocking; fires `next` after start |
| VideoPlaybackNode | GliderNode | Plays MP4/AVI/MOV fullscreen via `cv2.VideoCapture` in a frameless `VideoPlayerWindow`; fires `next` after completion |

#### Vision Nodes
| Node | Type | Description |
|------|------|-------------|
| ZoneInputNode | InterfaceNode | Monitors zone occupancy from CV processor; outputs Occupied (bool), Object Count (int); fires On Enter/On Exit exec signals |

#### Flow Function Nodes
| Node | Type | Description |
|------|------|-------------|
| StartFunctionNode | GliderNode | Entry point for user-defined flow functions |
| EndFunctionNode | GliderNode | Exit point that signals function completion |
| FunctionCallNode | GliderNode | Invokes a flow function subgraph via FlowFunctionRunner |

### 4.3 Serialization and Experiment Files

Experiments are saved as `.glider` files in JSON format. The choice of JSON over binary formats (like Pickle) ensures human readability and version control compatibility (Git). GLIDER employs a typed schema validation system (`schema.py`) with recursive `from_dict()` validation and detailed path-context error messages via `SchemaValidationError`.

The `ExperimentSerializer` performs atomic writes (temp file + `os.replace()`) to prevent corruption and supports version migration via `_validate_and_migrate()`.

**Schema Structure (v1.0.0):**

```json
{
  "schema_version": "1.0.0",
  "metadata": {
    "name": "Experiment Name",
    "description": "...",
    "author": "Researcher",
    "created": "2026-03-16T12:00:00",
    "modified": "2026-03-16T12:30:00",
    "tags": ["behavioral", "tracking"]
  },
  "hardware": {
    "boards": [
      {"id": "uuid", "type": "telemetrix", "port": "COM3", "settings": {}}
    ],
    "devices": [
      {"id": "uuid", "type": "DigitalOutput", "board_id": "uuid",
       "pin": 13, "name": "LED", "settings": {}}
    ]
  },
  "flow": {
    "nodes": [
      {"id": "uuid", "type": "StartExperiment",
       "position": {"x": 100, "y": 100},
       "properties": {}, "inputs": [], "outputs": []}
    ],
    "connections": [
      {"id": "uuid", "from_node": "uuid", "from_port": "next",
       "to_node": "uuid", "to_port": "exec",
       "connection_type": "exec"}
    ]
  },
  "dashboard": {
    "layout_mode": "vertical",
    "widgets": [
      {"node_id": "uuid", "position": {}, "size": {}, "visible": true}
    ]
  }
}
```

The `ExperimentSession` model extends beyond this schema with additional runtime state:
- **`CameraConfig`**: Resolution, FPS, camera index, CV backend, tracking settings
- **`ZoneConfig`**: Zone definitions with normalized coordinates
- **`Subject`**: Animal/subject metadata (ID, age, sex, weight, group, dose)
- **`custom_device_definitions`**: User-defined device templates
- **`flow_function_definitions`**: Reusable subgraph definitions

The separation of hardware and flow allows the user to replace a board definition (e.g., swap Arduino for Pi) without breaking the flow logic, provided the new device instance maintains the same UUID.

---

## 5. Hardware Abstraction Layer (HAL)

The HAL is the interface between the high-level Flow Engine and the low-level physical world. Its primary goal is to provide a uniform API for diverse hardware, enabling the software to treat a "Digital Output" on a Raspberry Pi exactly the same as one on an Arduino or a dedicated DAQ card.

### 5.1 Abstract Base Classes (ABCs)

To enforce consistency, GLIDER defines a set of Abstract Base Classes using Python's `abc` module.

#### The Board Interface (`BaseBoard`)

```python
class BaseBoard(ABC):
    @abstractmethod
    async def connect(self) -> bool: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def set_pin_mode(self, pin: int, mode: PinMode,
                           pin_type: PinType = PinType.DIGITAL) -> None: ...
    @abstractmethod
    async def write_digital(self, pin: int, value: bool) -> None: ...
    @abstractmethod
    async def read_digital(self, pin: int) -> bool: ...
    @abstractmethod
    async def write_analog(self, pin: int, value: int) -> None: ...
    @abstractmethod
    async def read_analog(self, pin: int) -> int: ...
```

Any hardware plugin must implement this interface. This polymorphism allows the Core to iterate over a list of `BaseBoard` objects and perform operations without knowing the specific hardware implementation details.

**Additional BaseBoard infrastructure:**
- **BoardConnectionState** enum: `DISCONNECTED`, `CONNECTING`, `CONNECTED`, `ERROR`, `RECONNECTING`
- **PinType** enum: `DIGITAL`, `ANALOG`, `PWM`, `I2C`, `SPI`, `SERVO`
- **PinMode** enum: `INPUT`, `OUTPUT`, `INPUT_PULLUP`, `INPUT_PULLDOWN`
- **PinCapability** dataclass: Per-pin supported types, max values, descriptions
- **BoardCapabilities** dataclass: Complete board profile (pins, analog/PWM resolution, I2C/SPI buses)
- **Auto-reconnection**: Background `_attempt_reconnect()` task with configurable interval (default 5s)
- **Emergency stop**: `emergency_stop()` sets all outputs to safe state
- **I2C locking**: Shared `asyncio.Lock` for I2C bus arbitration across devices
- **Callback system**: Pin value change, error, and connection state callbacks

#### The Device Interface (`BaseDevice`)

Devices represent higher-level components attached to the board (e.g., "Stepper Motor", "Temperature Sensor"). They wrap the `BaseBoard` methods into semantic actions.

```python
class BaseDevice(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...
    @abstractmethod
    async def shutdown(self) -> None: ...
    @property
    @abstractmethod
    def actions(self) -> dict[str, Callable]: ...
```

The `actions` property returns a dictionary mapping command strings to async methods (e.g., `{'activate': self.turn_on}`). This structure allows the Flow Engine to trigger device actions generically using `device.execute_action(name)` without needing to know the specific class type of the device.

### 5.2 Board Implementations

#### Arduino (Telemetrix-AIO)

For Arduino communication, GLIDER utilizes the **Telemetrix-AIO** library. Unlike the older pyfirmata, Telemetrix is actively maintained and supports asynchronous operation natively. It communicates with the Arduino over serial (USB) using a custom protocol that allows for callback-based reporting. This means the software doesn't need to constantly poll the Arduino; the Arduino pushes data when pins change, which is far more efficient for the event loop.

**Implementation Details (`TelemetrixBoard`):**
- **Threading Model**: Runs Telemetrix in a dedicated `TelemetrixThread` with its own asyncio event loop, completely isolating Arduino serial I/O from the Qt main loop.
- **Cross-thread RPC**: `TelemetrixThread.call_method()` uses `asyncio.run_coroutine_threadsafe()` with 5s call timeout and 10s outer timeout.
- **Board Types**: Supports Arduino Uno (14 digital + 6 analog + 6 PWM pins) and Arduino Mega (54 digital + 16 analog + 15 PWM pins) with pre-defined capability maps.
- **PWM Zero Handling**: Special three-step sequence when writing PWM 0: (1) zero the PWM register, (2) switch pin to digital output mode (disconnects PWM timer), (3) drive LOW with digital driver. This overcomes back-fed voltage issues on certain Arduino boards. Pins are tracked in `_pwm_pins_forced_low` and re-enabled on next non-zero write.
- **Callback Marshalling**: Pin change callbacks run in the Telemetrix thread and are dispatched to the main event loop via `call_soon_threadsafe()`.
- **Write Retries**: `write_analog()` retries up to 3 times on failure.

#### Raspberry Pi (GPIO)

For direct GPIO control on the Pi, GLIDER uses the **gpiozero** library with **lgpio** as the backend.

**Implementation Details (`PiGPIOBoard`):**
- **Pin Capabilities**: GPIO 2-27 general purpose; GPIO 12, 13, 18, 19 are hardware PWM; GPIO 2-3 are I2C (SDA/SCL); GPIO 7-11 are SPI.
- **Device Mapping**: Each pin gets a gpiozero device object---`DigitalOutputDevice` for outputs, `DigitalInputDevice` for inputs, `PWMOutputDevice` for PWM, `Servo` for servo control.
- **Async Wrapping**: Since gpiozero uses blocking calls and its own threading model, all operations are wrapped with `asyncio.to_thread()` to ensure they do not block the main event loop.
- **Input Callbacks**: gpiozero's `when_activated`/`when_deactivated` callbacks are marshalled to the main event loop via `call_soon_threadsafe()`.
- **Servo Mapping**: Angles 0-180 are mapped to gpiozero's -1 to +1 range.
- **No ADC**: The Pi has no built-in ADC; `read_analog()` raises `NotImplementedError`. Users should use the ADS1115 I2C device for analog readings.

#### Mock Board (Testing)

The `MockBoard` provides an in-memory simulated board (Arduino Mega-like, 54 pins) for testing without physical hardware. All operations are logged and pin states stored in dictionaries. Connects immediately with no real I/O.

### 5.3 Built-in Device Types

GLIDER includes seven built-in device types registered in the `DEVICE_REGISTRY`:

| Device Type | Required Pins | Actions | Description |
|-------------|--------------|---------|-------------|
| **DigitalOutput** | `output` | on, off, toggle, set | LED, relay control |
| **DigitalInput** | `input` | read | Button, beam break sensor; supports pullup; change callbacks |
| **AnalogInput** | `input` | read, read_voltage | Potentiometer, light sensor; configurable reference voltage |
| **PWMOutput** | `output` | set, set_percent, off | LED brightness, motor speed (0-255) |
| **Servo** | `signal` | set_angle, center | Servo motor (configurable min/max angle, default 0-180) |
| **ADS1115** | *(none---I2C)* | read, read_channel, read_voltage, read_all | 16-bit ADC via I2C for Raspberry Pi; 4 channels; configurable gain (2/3x to 16x) and data rate (8-860 SPS); uses board-level I2C lock |
| **MotorGovernor** | `up`, `down`, `signal` | up, down, stop, read_position | Motorized positioning with analog feedback; pulse-based up/down with 50ms timing |

### 5.4 Device Assignment and Pin Management

A common source of errors in experimental setups is pin conflict (e.g., assigning two devices to Pin 13). The GLIDER HAL includes a **PinManager** that tracks the allocation of resources per board.

**PinManager Features:**
- **Atomic allocation**: `allocate_device_pins()` allocates all pins for a device atomically---if any pin conflicts, none are allocated.
- **Conflict detection**: Raises `PinConflictError(pin, existing_device, new_device)` with clear error messages.
- **Type validation**: `validate_pin_type()` raises `InvalidPinError` if a pin doesn't support the requested operation (e.g., PWM on a digital-only pin).
- **Capability filtering**: `get_available_compatible_pins(pin_type)` returns only unallocated pins supporting the requested operation, used by the GUI to populate dropdown menus.
- **Per-device tracking**: `get_pins_for_device(device_id)` returns all allocations for a device, enabling clean release on device removal.

---

## 6. Computer Vision and Tracking System

A major addition beyond the original design specification, the Vision System provides a complete pipeline for real-time object tracking, behavioral analysis, and zone-based event triggering during experiments.

### 6.1 Camera Management

The **CameraManager** handles camera capture with support for multiple backends:
- **OpenCV** (`cv2.VideoCapture`): Primary backend for most cameras.
- **FFmpeg fallback** (`FFmpegCapture`): Subprocess-based capture for problematic cameras (e.g., ANY-MAZE Y800 grayscale cameras on Windows DirectShow).
- **Configurable settings**: Resolution, FPS, brightness, contrast, camera index, backend selection.
- **Frame callback system**: Registered callbacks receive frames for real-time processing.
- **Current FPS measurement**: Adaptive frame rate tracking.

The **MultiCameraManager** orchestrates multiple simultaneous cameras:
- Designate a primary camera for CV processing.
- Per-camera frame callbacks.
- Primary camera callback follows camera changes.
- Parallel streaming on all cameras.

### 6.2 Computer Vision Processing

The **CVProcessor** is the central real-time CV engine supporting four detection backends:

1. **Background Subtraction**: Detects moving objects against a learned background model.
2. **YOLO V8**: Object detection using the Ultralytics YOLO model.
3. **YOLO + ByteTrack**: YOLO detection with ByteTrack multi-object tracking for persistent IDs across frames.
4. **Motion-only**: Detects areas of motion without object classification.

**Key Data Structures:**
```python
Detection(class_id, class_name, confidence, bbox, centroid, track_id)
TrackedObject(track_id, class_name, bbox, confidence, centroid, age,
              behavioral_state, velocity, keypoints)
MotionResult(motion_detected, motion_area, motion_contours, motion_mask)
CVSettings(100+ configurable parameters)
```

**Processing Pipeline:**
1. Frame received from CameraManager callback
2. Optional frame skipping for performance optimization
3. Detection via selected backend
4. Multi-object tracking with persistent ID assignment
5. Behavioral state analysis per tracked object
6. Zone occupancy computation
7. Overlay rendering (bounding boxes, labels, trails, vision cones, zone fills)
8. Callback dispatch (detection, tracking, motion events)

### 6.3 Behavioral Analysis

The **BehaviorAnalyzer** classifies object movement patterns in real-time:

| State | Threshold | Color | Description |
|-------|-----------|-------|-------------|
| FREEZE | < 1.0 px/frame (sustained 15 frames) | Blue | Complete stillness |
| IMMOBILE | < 5.0 px/frame | Yellow | Minimal movement |
| MOVING | 5.0 - 50.0 px/frame | Green | Normal locomotion |
| DARTING | > 50.0 px/frame | Red | Rapid escape-like movement |

Features:
- Velocity smoothing over configurable window (default 5 frames)
- Sustained freeze detection requiring N consecutive frames below threshold
- Per-object independent state tracking

### 6.4 Zone System

The **Zone** system enables spatial event triggering:
- **Zone shapes**: Rectangle, Circle, Polygon
- **Normalized coordinates** (0-1): Resolution-independent zone definitions
- **ZoneConfiguration**: Container with `save()`/`load()` JSON persistence
- **ZoneTracker**: Tracks per-zone state (occupied, object count, object IDs) with enter/exit event flags per frame
- **ZoneInputNode**: Vision node that monitors zone occupancy and fires `On Enter`/`On Exit` execution signals

### 6.5 Camera Calibration

The **CameraCalibration** system converts pixel measurements to real-world units:
- **CalibrationLine**: Stores pixel coordinates, real-world length, and units (mm, cm, m, inches, feet)
- **Normalization**: All coordinates stored as normalized (0-1) for resolution independence
- **Averaging**: Multiple calibration lines averaged for improved accuracy
- **Conversion methods**: `pixels_to_mm()`, `mm_to_pixels()`, `real_distance()`

### 6.6 Video Recording

**VideoRecorder** (single camera):
- Dual output: raw video + annotated video with CV overlays
- `RecordingState` enum: IDLE, RECORDING, PAUSED, FINALIZING
- FPS drift detection and auto-correction via re-encoding
- Threaded writing via `FrameWriterThread` with configurable buffer (60 frames on Pi, 300 on PC)

**MultiVideoRecorder** (multi-camera):
- Separate video file per camera with naming convention `{experiment}_{timestamp}_cam{N}.mp4`
- Optional annotated video for primary camera only
- Per-camera frame drop tracking

### 6.7 Tracking Data Logger

The **TrackingDataLogger** produces CSV files with per-frame tracking data:

| Column | Description |
|--------|-------------|
| frame | Frame number |
| timestamp | Absolute timestamp |
| elapsed_ms | Milliseconds since experiment start |
| object_id | Persistent track ID |
| class | Object class name |
| bbox | Bounding box (x, y, w, h) |
| confidence | Detection confidence |
| center | Centroid (x, y) |
| distance_px | Frame-to-frame distance in pixels |
| distance_mm | Frame-to-frame distance in mm (if calibrated) |
| cumulative_mm | Total distance traveled |
| zone_ids | List of zones the object occupies |
| behavioral_state | FREEZE / IMMOBILE / MOVING / DARTING |
| velocity | Current velocity in px/frame |

Metadata headers include protocol, experimenter, lab, active subject, and calibration data. Heartbeat logging every 30 seconds when inactive prevents empty file detection issues.

---

## 7. LLM-Powered Agent System

GLIDER includes an AI-powered assistant that can build experiments programmatically using natural language instructions.

### 7.1 Architecture

```
AgentController  <--  AgentConfig (provider, model, temperature, etc.)
       |
   LLMBackend  (Ollama / OpenAI / Anthropic)
       |
   AgentToolkit
   +-- ExperimentToolExecutor  <-- FlowEngine
   +-- HardwareToolExecutor    <-- HardwareManager
   +-- KnowledgeToolExecutor   <-- Documentation
```

### 7.2 LLM Backend

The **LLMBackend** provides a unified interface over multiple LLM providers:
- **Ollama**: Local inference (e.g., `llama3.2:latest`)
- **OpenAI**: Cloud API (e.g., `gpt-4o`)
- **Anthropic**: Cloud API (e.g., `claude-sonnet-4-6`)

Features: Streaming and non-streaming responses, tool format conversion between providers, usage tracking.

### 7.3 Tool Categories

#### Experiment Tools (7 tools)
- `create_node`: Create nodes with auto-layout positioning
- `delete_node`: Remove node and attached connections
- `connect_nodes` / `disconnect_nodes`: Manage connections
- `set_node_property`: Update node configuration
- `get_flow_state`: Retrieve current graph state
- `validate_flow` / `clear_flow`: Validation and reset

#### Hardware Tools (12 tools)
- `list_boards` / `add_board` / `remove_board` / `connect_board` / `disconnect_board`
- `list_devices` / `add_device` / `remove_device` / `configure_device`
- `scan_ports`: Enumerate available serial ports
- `test_device`: Blink LED, read sensor, toggle output
- `get_pin_capabilities`: Board pin information

#### Knowledge Tools (5 tools)
- `explain_node`: Detailed node documentation with tips
- `explain_concept`: GLIDER concept explanations
- `get_examples`: Example experiments by category (basic, sensors, motors, advanced)
- `suggest_flow`: Design recommendations for task descriptions
- `troubleshoot`: Diagnose issues with solutions

### 7.4 Action Safety Model

Actions are classified by safety level:

| Classification | Actions | Behavior |
|---------------|---------|----------|
| **Safe** (auto-execute) | explain, suggest, get_state, get_documentation, validate_flow, scan_ports | No confirmation needed |
| **Normal** (configurable) | create_node, delete_node, connect_nodes, add_device, etc. | Confirmation when `require_confirmation=True` |
| **Dangerous** (always confirm) | clear_flow, remove_board | Always requires user approval |

Actions flow through states: `PENDING` -> `CONFIRMED` -> `EXECUTING` -> `COMPLETED` or `FAILED`.

### 7.5 Analysis Module

A separate `analysis/` subsystem provides post-experiment data analysis capabilities with specialized tools, prompts, and an `AnalysisController` for managing analysis workflows.

---

## 8. Data Recording and Logging

### 8.1 Data Recorder

The **DataRecorder** logs device states to CSV at configurable intervals during experiment execution:
- Default sampling interval: 100ms
- Columns auto-generated from registered devices
- Each sample includes timestamp, all device values, and zone data
- Output directory configurable; files named by experiment and timestamp

### 8.2 Device Library

The **DeviceLibrary** provides import/export functionality for reusable definitions:

| Format | Extension | Contents |
|--------|-----------|----------|
| Custom Device | `.gdevice` | Single CustomDeviceDefinition (JSON) |
| Flow Function | `.gflow` | Single FlowFunctionDefinition (JSON) |
| Library Bundle | `.glibrary` | Multiple devices and functions (JSON archive) |

Methods include `list_library_devices()`, `list_library_functions()`, `export_session_definitions()`, and `import_to_session()`.

---

## 9. Concurrency and Performance

Orchestrating hardware requires precise timing. If the UI thread freezes while rendering a complex graph, a motor might overrun its limit switch. GLIDER's concurrency model addresses this risk.

### 9.1 The Asyncio-Qt Bridge

The integration of asyncio with PyQt6 is achieved through **qasync**. This library provides a custom `QEventLoop` that acts as a drop-in replacement for the standard asyncio loop but pumps Qt events (mouse clicks, redraws) within the same cycle.

**Implementation Strategy:**
1. **Startup:** The `__main__.py` entry point creates a `QApplication` instance and a `qasync.QEventLoop`.
2. **Fallback:** If qasync is unavailable, `run_sync_fallback()` provides degraded operation.
3. **Deferred Loading:** Plugins are loaded asynchronously after core initialization.
4. **Tasks:** Hardware operations are scheduled as `asyncio.Task` objects, tracked in `FlowEngine._running_tasks` with done callbacks for automatic cleanup.
5. **Shutdown:** Waits for loop to close; uses try/except to handle already-closed loops gracefully.

```python
def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    core = GliderCore()
    # Async initialization
    loop.run_until_complete(init_glider(app, args))
    window = create_main_window(app, core, force_mode)
    window.show()

    with loop:
        loop.run_forever()
```

### 9.2 Thread Isolation Strategy

GLIDER uses dedicated threads where asyncio alone is insufficient:

| Thread | Purpose | Communication |
|--------|---------|---------------|
| **Main Thread** | Qt event loop + asyncio tasks | Direct |
| **TelemetrixThread** | Arduino serial I/O via telemetrix-aio | `run_coroutine_threadsafe()` / `call_soon_threadsafe()` |
| **FrameWriterThread** | Video disk I/O (per recorder) | `queue.Queue` with `put_nowait()` |
| **gpiozero callbacks** | Raspberry Pi GPIO interrupts | `call_soon_threadsafe()` to main loop |

This prevents the "QObject: Cannot create children for a parent that is in a different thread" error common in PyQt applications while ensuring no hardware I/O blocks the UI.

### 9.3 Managing Throughput and Backpressure

High-frequency sensors (e.g., reading an accelerometer at 100Hz) can flood the event loop if every data point triggers a UI update (which runs at screen refresh rate, typically 60Hz).

**Implemented Strategies:**

1. **DataRecorder sampling**: Configurable interval (default 100ms) decouples data logging from device polling rate.
2. **Device refresh timer**: RunnerPanel updates device status cards at 200ms intervals, not per-change.
3. **Frame skipping**: CVProcessor can process every Nth frame for performance on resource-constrained devices.
4. **Buffered video writing**: FrameWriterThread queues frames (60-frame buffer on Pi, 300 on PC) with frame drop tracking and periodic warnings.
5. **Callback-based analog reporting**: TelemetrixBoard's analog callbacks fire only on value changes (push-based, not poll-based), with cached values for read operations.

### 9.4 Task Management

The FlowEngine tracks all execution tasks:
- Tasks stored in `_running_tasks` set
- Done callbacks automatically discard completed tasks
- On `stop()`: snapshot tasks, cancel all, await completion with `asyncio.gather(*tasks, return_exceptions=True)`
- `asyncio.Lock` (`_experiment_lock`) in GliderCore prevents concurrent start/stop operations
- `asyncio.Event` used for FlowFunctionRunner completion signaling
- `asyncio.wait_for()` with timeout for long-running operations

---

## 10. Scalability and Modularity

Scalability in GLIDER refers to the ability to manage increasing complexity in both the size of the experiment (number of nodes) and the diversity of hardware.

### 10.1 Plugin Discovery Mechanism

The plugin system uses **entry points** for driver discovery. `pyproject.toml` declares the `glider.driver` entry point group, and at startup, the Plugin Manager scans `importlib.metadata` for installed packages advertising this group. Additionally, built-in drivers auto-register at import time:

```python
# Bottom of hardware_manager.py
HardwareManager.register_driver("arduino", TelemetrixBoard)
HardwareManager.register_driver("raspberry_pi", PiGPIOBoard)
```

Node types register similarly via category-specific `register_*_nodes()` functions called from `GliderCore._register_builtin_nodes()`:
```python
def _register_builtin_nodes(self):
    register_experiment_nodes(self._flow_engine)
    register_hardware_nodes(self._flow_engine)
    register_logic_nodes(self._flow_engine)
    register_interface_nodes(self._flow_engine)
    register_flow_function_nodes(self._flow_engine)
    register_vision_nodes(self._flow_engine)
```

### 10.2 Class-Level Registries

Both `FlowEngine._node_registry` and `HardwareManager._driver_registry` are **class-level dictionaries** shared across all instances. This enables plugins to register nodes and drivers without needing access to a specific engine instance---they simply call the class method:

```python
FlowEngine.register_node("MyCustomNode", MyCustomNodeClass)
HardwareManager.register_driver("my_board", MyBoardClass)
```

### 10.3 Configuration System

GLIDER provides a layered configuration system via `GliderConfig`:

| Config Section | Parameters |
|---------------|------------|
| **TimingConfig** | Device refresh rate, polling intervals, timeouts, pulse durations |
| **UIConfig** | Window sizes, panel dimensions, undo stack size |
| **HardwareConfig** | ADC resolution, PWM range, servo angles, poll intervals |
| **PathConfig** | User config directory, library paths, file extensions |

Global singleton access via `get_config()` with lazy loading. Persistent to `~/.glider/config.json`.

### 10.4 Type System

The `NodeType` and `DeviceType` enums provide string-safe type identification:
- `NodeType`: StartExperiment, EndExperiment, Delay, Output, Input, CustomDevice, Loop, WaitForInput, etc.
- `DeviceType`: DigitalOutput, DigitalInput, AnalogInput, PWMOutput, Servo, Stepper
- Both support `from_string()` with normalization and `from_string_safe()` returning None on mismatch.
- `DeviceType` properties: `is_input`, `is_output`, `is_analog` for filtering.

---

## 11. Deployment and Environment Configuration

### 11.1 Raspberry Pi Kiosk Deployment

To function effectively as an appliance on the Raspberry Pi, GLIDER must take over the user experience.

**Autostart Configuration:**
The recommended deployment uses the LXDE autostart mechanism or a Systemd User Service to launch GLIDER immediately after the X server starts.

**Command:** `glider --runner --fullscreen` (or `glider --builder` for desktop)

**Screen Configuration:**
The 480x800 resolution is non-standard for many desktop environments. The deployment documentation specifies the `/boot/config.txt` settings (e.g., `display_rotate=1` or `dtoverlay=vc4-kms-dsi-7inch`) required to ensure the touchscreen is correctly oriented and calibrated.

### 11.2 Packaging and Dependencies

GLIDER uses `pyproject.toml` with **uv** as the package manager and optional dependency groups:

```toml
[project.optional-dependencies]
pc = ["telemetrix-aio", "pyserial", ...]      # Desktop/Arduino dependencies
rpi = ["gpiozero", "lgpio"]                    # Raspberry Pi GPIO
dev = ["pytest", "pytest-asyncio", "pytest-qt"] # Development/testing
```

**Installation:**
```bash
# Desktop development
uv sync --extra pc --extra dev

# Raspberry Pi (requires system PyQt6: sudo apt install python3-pyqt6)
pip install -e .
```

**Entry Point:**
```toml
[project.scripts]
glider = "glider.__main__:main"

[project.entry-points."glider.driver"]
arduino = "glider.hal.boards.telemetrix_board:TelemetrixBoard"
raspberry_pi = "glider.hal.boards.pi_gpio_board:PiGPIOBoard"
```

**Platform Note:** Raspberry Pi requires system-installed PyQt6 (`sudo apt install python3-pyqt6`) and a venv with `--system-site-packages`. The application does not support Python 3.14+.

### 11.3 Error Handling and Recovery

Hardware in the loop implies instability (wires get pulled, power fails).

- **Reconnection Logic:** The `BaseBoard` includes an `auto_reconnect` flag. If a connection exception occurs, the board starts a background `_attempt_reconnect()` task that retries every 5 seconds, notifying the GUI via state change callbacks.
- **Safe State:** On any unrecoverable error, the Core triggers `emergency_stop()`, which invokes `shutdown()` on all registered devices (ensuring motors stop and heaters turn off) and cancels all running flow tasks.
- **Experiment Lock:** An `asyncio.Lock` prevents concurrent start/stop operations that could leave the system in an inconsistent state.
- **State Callbacks:** Multiple callback lists throughout the system (state changes, errors, connection changes) enable the GUI to reflect system status in real-time without polling.
- **Atomic File Writes:** The serializer uses temp file + `os.replace()` to prevent experiment file corruption on crash.

---

## 12. Testing

GLIDER uses **pytest** with specialized plugins for comprehensive testing:

- **pytest-asyncio**: For testing async hardware operations and flow execution
- **pytest-qt**: For GUI widget testing
- **MockBoard**: In-memory simulated board (54-pin Arduino Mega profile) for hardware-free testing
- **Mock fixtures** (`tests/conftest.py`): `mock_board`, `mock_device`, `mock_hardware_manager`, `mock_core`

**Test Structure:**
```
tests/
+-- unit/
|   +-- core/          # GliderCore, FlowEngine, HardwareManager
|   +-- hal/           # Board drivers, devices, PinManager
|   +-- nodes/         # Node implementations
|   +-- vision/        # Zones, calibration, frame writer
|   +-- serialization/ # Schema validation
+-- integration/
|   +-- test_experiment_workflow.py
+-- latency_test.py
```

---

## 13. Core Class Definitions

**Table 2: Core Class Hierarchy**

| Class | Type | Responsibility | Dependencies |
|-------|------|----------------|-------------|
| **GliderCore** | Controller (Facade) | Central orchestrator. Initializes event loop, loads plugins, manages ExperimentSession, coordinates hardware/flow/recording/vision subsystems. | asyncio, importlib |
| **ExperimentSession** | Model | Represents current state: Hardware Map, Logic Graph, Camera/Zone config, Custom Devices, Flow Functions. Serializable to JSON. Dirty tracking and state callbacks. | json, dataclasses |
| **HardwareManager** | Sub-Controller | Manages board/device lifecycle. Class-level driver registry. Per-board PinManager. Parallel connect/initialize. | BaseBoard, BaseDevice, PinManager |
| **FlowEngine** | Sub-Controller | Wraps ryvencore session (with standalone fallback). Class-level node registry. Execution propagation via asyncio tasks. | ryvencore (optional) |
| **DataRecorder** | Sub-Controller | CSV logging of device states at configurable intervals. | csv, asyncio |
| **CameraManager** | Sub-Controller | Camera capture with OpenCV/FFmpeg backends. Frame callbacks. | cv2, subprocess |
| **MultiCameraManager** | Sub-Controller | Multi-camera orchestration with primary camera designation. | CameraManager |
| **CVProcessor** | Sub-Controller | Real-time CV with 4 detection backends, tracking, behavioral analysis. | ultralytics, cv2 |
| **VideoRecorder** | Sub-Controller | Single-camera video recording with annotated output. | cv2, FrameWriterThread |
| **MultiVideoRecorder** | Sub-Controller | Multi-camera recording with per-camera files. | VideoRecorder |
| **TrackingDataLogger** | Sub-Controller | CSV tracking data output with metadata headers. | csv |
| **AgentController** | Sub-Controller | LLM-powered assistant orchestrator. Lazy initialization. | LLMBackend, AgentToolkit |
| **CustomDeviceRunner** | Executor | Runtime executor for user-defined devices (pin read/write). | BaseBoard |
| **FlowFunctionRunner** | Executor | Subgraph executor for flow functions (entry -> exit). | FlowEngine, asyncio.Event |
| **DeviceLibrary** | Utility | Import/export of custom devices and flow functions. | json |
| **MainWindow** | View | Primary PyQt6 window. Manages layout, menus, docks, status bar, view switching. | PyQt6 |
| **NodeGraphView** | View | Custom QGraphicsView for visual flow editing. Pan, zoom, drag-and-drop. | PyQt6.QGraphicsView |
| **NodeEditorController** | Controller | Brain of Builder mode. Handles node/connection CRUD, properties panel, undo/redo integration. | PyQt6.QObject |
| **RunnerPanel** | View | Touch-optimized runtime interface with device cards, timer, controls. | PyQt6 |
| **RunnerDashboard** | View | Dynamic widget container for InterfaceNode touch widgets. | WidgetFactory |
| **ViewManager** | Utility | Display detection, responsive configuration, stylesheet selection. | QScreen |

---

## 14. Conclusion

The GLIDER architecture provides a comprehensive solution for modern experimental orchestration. By leveraging the specific strengths of **PyQt6** for visualization, **ryvencore** for logic definition, and **asyncio/qasync** for concurrency, it successfully navigates the trade-offs between ease of use and technical capability.

Beyond the original design vision, the implemented system delivers several major capabilities that elevate GLIDER from a hardware control tool to a complete experimental platform:

- **Computer Vision Pipeline**: Real-time object detection (YOLO V8), multi-object tracking (ByteTrack), behavioral analysis (freeze/immobile/moving/darting classification), zone-based event triggering, and calibrated distance measurement---all integrated into the node graph via ZoneInputNode.
- **Multi-Camera Support**: Simultaneous capture from multiple cameras with per-camera video recording, primary camera designation for CV processing, and a unified management interface.
- **AI-Powered Experiment Design**: An LLM agent system supporting Ollama, OpenAI, and Anthropic providers that can build experiments programmatically using 24+ tools across experiment design, hardware configuration, and knowledge domains, with a safety model that gates dangerous operations behind user confirmation.
- **Custom Device Abstractions**: A configuration-driven device definition system that allows users to create new device types without writing Python code, exportable and shareable as library files.
- **Reusable Flow Functions**: Subgraph encapsulation enabling researchers to define common experimental patterns once and reuse them across experiments.
- **Audio/Video Playback**: Integrated media nodes for stimulus presentation during behavioral experiments.
- **Professional IDE Experience**: A fully custom node graph editor with undo/redo, category-colored nodes, bezier curve connections, properties panel, drag-and-drop library, and comprehensive keyboard navigation.

The result is a platform that empowers researchers to construct complex, hardware-integrated experiments with the same ease as drawing a flowchart, scalable from a laboratory workbench to a deployed embedded sensor station. GLIDER bridges the gap between design and execution in the laboratory, providing a single tool that handles hardware control, computer vision, data recording, and experimental logic in a unified, extensible architecture.
