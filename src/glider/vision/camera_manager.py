"""
Camera Manager - Enumerate and manage webcam devices.

Provides thread-safe webcam capture with configurable settings
and frame callbacks for video recording and CV processing.

Includes FFmpeg fallback for cameras that OpenCV cannot handle
(e.g., Y800/grayscale scientific cameras like ANYMAZE).
"""

import os

# Suppress OpenCV warnings before importing cv2
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import logging
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Empty, Queue
from typing import Any, Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FFmpegCapture:
    """
    FFmpeg-based video capture for cameras that OpenCV cannot handle.

    Uses FFmpeg's DirectShow input on Windows to capture frames from
    cameras with problematic formats like Y800 (grayscale).
    """

    def __init__(self, device_name: str, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize FFmpeg capture.

        Args:
            device_name: DirectShow device name (e.g., "ANY-MAZE 3.1")
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self._device_name = device_name
        self._width = width
        self._height = height
        self._fps = fps
        self._process: Optional[subprocess.Popen] = None
        self._frame_size = width * height * 3  # BGR output
        self._is_open = False

    def open(self) -> bool:
        """Start FFmpeg capture process."""
        if self._process is not None:
            self.release()

        # Check if ffmpeg is available
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.warning("FFmpeg not found in PATH")
            return False

        try:
            # FFmpeg command to capture from DirectShow and output raw BGR frames
            cmd = [
                ffmpeg_path,
                "-f",
                "dshow",
                "-video_size",
                f"{self._width}x{self._height}",
                "-framerate",
                str(self._fps),
                "-pixel_format",
                "gray",  # Request grayscale input (Y800)
                "-i",
                f"video={self._device_name}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",  # Output as BGR for OpenCV
                "-an",  # No audio
                "-",  # Output to stdout
            ]

            logger.info(f"Starting FFmpeg capture: {' '.join(cmd)}")

            # Start FFmpeg process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 4,  # Buffer a few frames
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            # Give FFmpeg time to start
            time.sleep(1.0)

            # Check if process is still running
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                logger.error(f"FFmpeg exited immediately: {stderr[:500]}")
                self._process = None
                return False

            self._is_open = True
            logger.info("FFmpeg capture started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            self._process = None
            return False

    def isOpened(self) -> bool:
        """Check if capture is open."""
        return self._is_open and self._process is not None and self._process.poll() is None

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from FFmpeg."""
        if not self.isOpened():
            return False, None

        try:
            # Read raw frame data
            raw_frame = self._process.stdout.read(self._frame_size)

            if len(raw_frame) != self._frame_size:
                return False, None

            # Convert to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((self._height, self._width, 3))

            return True, frame

        except Exception as e:
            logger.debug(f"FFmpeg read error: {e}")
            return False, None

    def grab(self) -> bool:
        """Grab a frame (for compatibility with OpenCV API)."""
        ret, self._last_frame = self.read()
        return ret

    def retrieve(self) -> tuple[bool, Optional[np.ndarray]]:
        """Retrieve the last grabbed frame."""
        if hasattr(self, "_last_frame") and self._last_frame is not None:
            return True, self._last_frame
        return False, None

    def get(self, prop_id: int) -> float:
        """Get property (limited support for compatibility)."""
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        elif prop_id == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*"BGR3"))
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        """Set property (limited support - requires restart)."""
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self._width = int(value)
            return True
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self._height = int(value)
            return True
        elif prop_id == cv2.CAP_PROP_FPS:
            self._fps = int(value)
            return True
        return False

    def release(self) -> None:
        """Release FFmpeg process."""
        self._is_open = False
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            logger.info("FFmpeg capture released")


def _get_camera_backend() -> int:
    """Get the appropriate camera backend for the current platform."""
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    elif sys.platform == "linux":
        return cv2.CAP_V4L2
    else:
        # macOS and others - let OpenCV auto-select
        return cv2.CAP_ANY


def _get_windows_fallback_backends() -> list[int]:
    """Get list of backends to try on Windows in order of preference."""
    return [
        cv2.CAP_DSHOW,  # DirectShow - most compatible with USB cameras
        cv2.CAP_FFMPEG,  # FFmpeg - better format support including Y800
        cv2.CAP_ANY,  # Auto-detect - fallback
    ]


# Common pixel formats to try for problematic cameras
# Y800/GREY first for scientific cameras like ANYMAZE
PIXEL_FORMATS_TO_TRY = [
    "Y800",  # 8-bit grayscale - ANYMAZE and scientific cameras
    "GREY",  # 8-bit grayscale (alternate name)
    "Y8  ",  # 8-bit grayscale (with padding)
    None,  # Let camera choose default
    "MJPG",  # Motion JPEG - widely supported
    "YUY2",  # YUV 4:2:2 - common USB camera format
    "YUYV",  # Same as YUY2, different name
    "NV12",  # YUV 4:2:0 - used by some cameras
    "I420",  # YUV 4:2:0 planar
    "RAW ",  # Raw format
]


def _is_raspberry_pi() -> bool:
    """Check if running on a Raspberry Pi."""
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().lower()
            return "raspberry pi" in model
    except (FileNotFoundError, PermissionError):
        return False


def _get_windows_camera_names() -> list[str]:
    """
    Get camera device names on Windows using DirectShow enumeration.

    Returns:
        List of camera device names in order (index 0, 1, 2, ...)
    """
    camera_names = []

    if sys.platform != "win32":
        return camera_names

    try:
        # Use DirectShow device enumeration via ffmpeg/ffprobe if available
        import subprocess

        # Try ffmpeg device listing (most reliable for DirectShow order)
        try:
            result = subprocess.run(
                ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                ),
            )
            # Parse output - device names appear after "DirectShow video devices"
            output = result.stderr  # ffmpeg outputs to stderr
            in_video_section = False
            for line in output.split("\n"):
                if "DirectShow video devices" in line:
                    in_video_section = True
                    continue
                if "DirectShow audio devices" in line:
                    break
                if in_video_section and ']  "' in line:
                    # Extract name between quotes
                    start = line.find('"') + 1
                    end = line.rfind('"')
                    if start > 0 and end > start:
                        camera_names.append(line[start:end])
        except FileNotFoundError:
            pass  # ffmpeg not installed

        # Fallback: Use PowerShell WMI query
        if not camera_names:
            ps_command = """
            Get-CimInstance Win32_PnPEntity |
            Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } |
            Select-Object -ExpandProperty Name
            """

            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                ),
            )

            if result.returncode == 0 and result.stdout.strip():
                camera_names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]

    except Exception as e:
        logger.debug(f"Failed to get camera names: {e}")

    return camera_names


def _wake_up_miniscope(device_index: int) -> bool:
    """
    Wake up miniscope hardware using v4l2-ctl commands.

    Miniscopes require specific initialization to turn on the LED
    and configure the sensor properly.

    Args:
        device_index: Camera device index (e.g., 2 for /dev/video2)

    Returns:
        True if wake-up commands succeeded
    """
    import subprocess

    device_path = f"/dev/video{device_index}"

    try:
        # Set exposure time
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--set-ctrl=exposure_time_absolute=100"],
            capture_output=True,
            timeout=2,
        )
        # Reset saturation to kick the LED
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--set-ctrl=saturation=0"],
            capture_output=True,
            timeout=2,
        )
        time.sleep(0.1)
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--set-ctrl=saturation=128"],
            capture_output=True,
            timeout=2,
        )
        # Set brightness
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--set-ctrl=brightness=50"],
            capture_output=True,
            timeout=2,
        )
        logger.info(f"Miniscope wake-up sequence completed for {device_path}")
        return True
    except FileNotFoundError:
        logger.warning("v4l2-ctl not found - miniscope wake-up skipped")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Miniscope wake-up timed out")
        return False
    except Exception as e:
        logger.warning(f"Miniscope wake-up failed: {e}")
        return False


def _apply_miniscope_controls(device_index: int, settings: "CameraSettings") -> bool:
    """
    Apply all miniscope camera controls via v4l2-ctl.

    Args:
        device_index: Camera device index (e.g., 2 for /dev/video2)
        settings: Camera settings containing control values

    Returns:
        True if controls were applied successfully
    """
    import subprocess

    device_path = f"/dev/video{device_index}"

    # Map settings to v4l2-ctl control names
    controls = {
        "brightness": settings.brightness,
        "contrast": settings.contrast,
        "saturation": settings.saturation,
        "hue": settings.hue,
        "gamma": settings.gamma,
        "gain": settings.gain,
        "sharpness": settings.sharpness,
        "exposure_time_absolute": settings.exposure_time,
        "focus_absolute": settings.focus,
        "zoom_absolute": settings.zoom,
        "iris_absolute": settings.iris,
    }

    try:
        for ctrl_name, ctrl_value in controls.items():
            result = subprocess.run(
                ["v4l2-ctl", "-d", device_path, f"--set-ctrl={ctrl_name}={ctrl_value}"],
                capture_output=True,
                timeout=2,
                text=True,
            )
            if result.returncode != 0 and result.stderr:
                # Some controls may not be supported - log but don't fail
                logger.debug(f"Control {ctrl_name} not applied: {result.stderr.strip()}")

        logger.info(f"Applied miniscope controls to {device_path}")
        return True
    except FileNotFoundError:
        logger.warning("v4l2-ctl not found - miniscope controls not applied")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Miniscope control application timed out")
        return False
    except Exception as e:
        logger.warning(f"Failed to apply miniscope controls: {e}")
        return False


def _send_miniscope_i2c_command(
    device_index: int, contrast: int, gamma: int, sharpness: int
) -> bool:
    """
    Send an I2C command to Miniscope V4 via UVC control tunnel.

    The Miniscope DAQ uses Contrast/Gamma/Sharpness UVC controls as a tunnel
    to send arbitrary I2C commands to the hardware. The byte layout is:
    - Sharpness: data2 | data1
    - Gamma: data0 | reg0
    - Contrast: packet_length | i2c_address

    Note: Contrast and Gamma values must be divided by 100 due to UVC scaling.

    Args:
        device_index: Camera device index
        contrast: Contrast value (packet_length << 8 | i2c_address)
        gamma: Gamma value (data0 << 8 | reg0)
        sharpness: Sharpness value (data2 << 8 | data1)

    Returns:
        True if command was sent successfully
    """
    import subprocess

    device_path = f"/dev/video{device_index}"

    # UVC controls are scaled by 100 for contrast and gamma
    contrast_scaled = contrast / 100.0
    gamma_scaled = gamma / 100.0

    try:
        # Set sharpness first (no scaling needed)
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, f"--set-ctrl=sharpness={sharpness}"],
            capture_output=True,
            timeout=2,
        )
        # Set gamma (scaled)
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, f"--set-ctrl=gamma={gamma_scaled}"],
            capture_output=True,
            timeout=2,
        )
        # Set contrast last (triggers the I2C transaction)
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, f"--set-ctrl=contrast={contrast_scaled}"],
            capture_output=True,
            timeout=2,
        )
        return True
    except Exception as e:
        logger.debug(f"I2C command failed: {e}")
        return False


def _set_miniscope_led(device_index: int, power_percent: int) -> bool:
    """
    Set Miniscope V4 LED power.

    LED control requires two I2C commands:
    1. MCU (address 0x20): Enable/disable LED driver
    2. Digital potentiometer (address 0x58): Set brightness

    Args:
        device_index: Camera device index
        power_percent: LED power 0-100 (0=off, 100=max brightness)

    Returns:
        True if LED was set successfully
    """
    # Convert percentage to hardware value (0-255, where 255=off, 0=brightest)
    # Clamp to valid range
    power_percent = max(0, min(100, power_percent))

    if power_percent == 0:
        # Turn off LED
        led_value = 255
    else:
        # Map 1-100% to 254-0 (inverted, brighter = lower value)
        led_value = int(255 - (power_percent * 255 / 100))
        led_value = max(0, min(254, led_value))

    logger.debug(f"Setting LED power: {power_percent}% -> hardware value {led_value}")

    # Command 1: MCU (address 0x20)
    # Sets LED enable state - register 0x01
    # Format: contrast=0x0320, gamma=(led_value<<8)|0x01, sharpness=0x0000
    contrast1 = 0x0320  # length=3, address=0x20
    gamma1 = (led_value << 8) | 0x01  # led_value, register 0x01
    sharpness1 = 0x0000

    success1 = _send_miniscope_i2c_command(device_index, contrast1, gamma1, sharpness1)

    # Small delay between commands
    time.sleep(0.05)

    # Command 2: Digital potentiometer (address 0x58)
    # Sets brightness via voltage divider
    # Format: contrast=0x0458, gamma=0x7200, sharpness=led_value
    # The 0x72 (114) is a hardcoded value for the voltage divider reference
    contrast2 = 0x0458  # length=4, address=0x58
    gamma2 = 0x7200  # hardcoded 0x72, register 0x00
    sharpness2 = led_value

    success2 = _send_miniscope_i2c_command(device_index, contrast2, gamma2, sharpness2)

    if success1 and success2:
        logger.info(f"Miniscope LED set to {power_percent}%")
    else:
        logger.warning(f"Failed to set Miniscope LED (cmd1={success1}, cmd2={success2})")

    return success1 and success2


def _set_miniscope_ewl_focus(device_index: int, focus_value: int) -> bool:
    """
    Set Miniscope V4 electrowetting lens (EWL) focus.

    Args:
        device_index: Camera device index
        focus_value: Focus position 0-255

    Returns:
        True if focus was set successfully
    """
    # Clamp to valid range
    focus_value = max(0, min(255, focus_value))

    logger.debug(f"Setting EWL focus to {focus_value}")

    # EWL focus command
    # Address 0xEE, register 0x08
    # Format: contrast=0x04EE, gamma=(focus_value<<8)|0x08, sharpness=0x0002
    contrast = 0x04EE  # length=4, address=0xEE
    gamma = (focus_value << 8) | 0x08  # focus value, register 0x08
    sharpness = 0x0002  # data for EWL

    success = _send_miniscope_i2c_command(device_index, contrast, gamma, sharpness)

    if success:
        logger.info(f"Miniscope EWL focus set to {focus_value}")
    else:
        logger.warning("Failed to set Miniscope EWL focus")

    return success


def _apply_miniscope_hardware_controls(device_index: int, settings: "CameraSettings") -> bool:
    """
    Apply Miniscope V4 hardware controls (LED and EWL).

    Args:
        device_index: Camera device index
        settings: Camera settings with led_power and ewl_focus

    Returns:
        True if controls were applied successfully
    """
    success = True

    # Set LED power
    if not _set_miniscope_led(device_index, settings.led_power):
        success = False

    # Small delay between commands
    time.sleep(0.1)

    # Set EWL focus
    if not _set_miniscope_ewl_focus(device_index, settings.ewl_focus):
        success = False

    return success


# =============================================================================
# Windows Miniscope Control via OpenCV
# =============================================================================
# On Windows, miniscope I2C commands are sent by encoding them into
# Contrast/Gamma/Sharpness camera properties. This matches the Bonsai.Miniscope
# implementation and works with the Cypress FX3 USB controller.
# =============================================================================


def _create_miniscope_command(i2c_addr: int, *data_bytes: int) -> int:
    """
    Create a 64-bit command for miniscope firmware.

    Format matches Bonsai.Miniscope Helpers.CreateCommand() exactly.

    Args:
        i2c_addr: I2C device address (e.g., 32 for MCU, 88 for pot, 238 for EWL)
        data_bytes: Up to 5 data bytes to send

    Returns:
        64-bit command word
    """
    command = i2c_addr

    if len(data_bytes) == 5:
        # Full 6-byte package: set address LSB to 1
        command |= 0x01
        for i, byte in enumerate(data_bytes):
            command |= (byte & 0xFF) << (8 * (i + 1))
    else:
        # Partial package: encode (length + 1) in second byte
        command |= ((len(data_bytes) + 1) << 8)
        for i, byte in enumerate(data_bytes):
            command |= (byte & 0xFF) << (8 * (i + 2))

    return command


def _send_miniscope_config_opencv(cap: cv2.VideoCapture, command: int) -> bool:
    """
    Send command to miniscope by splitting across Contrast/Gamma/Sharpness.

    This matches the Bonsai.Miniscope SendConfig implementation exactly.

    Args:
        cap: OpenCV VideoCapture object
        command: 64-bit command word

    Returns:
        True if all properties were set successfully
    """
    # Split 64-bit command into three 16-bit values
    # Bonsai order: Contrast (bits 0-15), Gamma (bits 16-31), Sharpness (bits 32-47)
    contrast_val = command & 0xFFFF
    gamma_val = (command >> 16) & 0xFFFF
    sharpness_val = (command >> 32) & 0xFFFF

    # Send via camera properties in correct order (Contrast, Gamma, Sharpness)
    r1 = cap.set(cv2.CAP_PROP_CONTRAST, contrast_val)
    r2 = cap.set(cv2.CAP_PROP_GAMMA, gamma_val)
    r3 = cap.set(cv2.CAP_PROP_SHARPNESS, sharpness_val)

    return r1 and r2 and r3


def _set_miniscope_led_opencv(cap: cv2.VideoCapture, power_percent: int) -> bool:
    """
    Set Miniscope V4 LED power using OpenCV (Windows).

    Args:
        cap: OpenCV VideoCapture object
        power_percent: LED power 0-100 (0=off, 100=max brightness)

    Returns:
        True if LED was set successfully
    """
    power_percent = max(0, min(100, power_percent))

    # Convert to 0-255 and invert (miniscope uses inverted scale)
    led_value = int(255 - (power_percent * 2.55))

    logger.debug(f"Setting LED power: {power_percent}% -> hardware value {led_value}")

    # Command 1: MCU (I2C address 32 = 0x20)
    cmd1 = _create_miniscope_command(32, 1, led_value)
    success1 = _send_miniscope_config_opencv(cap, cmd1)

    # Command 2: Digital potentiometer (I2C address 88 = 0x58)
    cmd2 = _create_miniscope_command(88, 0, 114, led_value)
    success2 = _send_miniscope_config_opencv(cap, cmd2)

    if success1 and success2:
        logger.info(f"Miniscope LED set to {power_percent}%")
    else:
        logger.warning(f"Failed to set Miniscope LED (cmd1={success1}, cmd2={success2})")

    return success1 and success2


def _init_miniscope_ewl_opencv(cap: cv2.VideoCapture) -> bool:
    """
    Initialize the EWL (Electrowetting Lens) driver.

    Must be called before set_ewl_focus().

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        True if initialization was successful
    """
    # Initialize MAX14574 EWL driver (I2C address 238 = 0xEE)
    cmd = _create_miniscope_command(238, 3, 3)
    success = _send_miniscope_config_opencv(cap, cmd)

    if success:
        logger.debug("EWL driver initialized")
    else:
        logger.warning("Failed to initialize EWL driver")

    return success


def _set_miniscope_ewl_opencv(cap: cv2.VideoCapture, focus_value: int) -> bool:
    """
    Set Miniscope V4 EWL focus using OpenCV (Windows).

    Args:
        cap: OpenCV VideoCapture object
        focus_value: Focus position 0-255 (128 is neutral)

    Returns:
        True if focus was set successfully
    """
    focus_value = max(0, min(255, focus_value))

    logger.debug(f"Setting EWL focus to {focus_value}")

    # EWL focus command: address 238 (0xEE), register 8, value, mode 2
    cmd = _create_miniscope_command(238, 8, focus_value, 2)
    success = _send_miniscope_config_opencv(cap, cmd)

    if success:
        logger.info(f"Miniscope EWL focus set to {focus_value}")
    else:
        logger.warning("Failed to set Miniscope EWL focus")

    return success


# Picamera2 is imported lazily to avoid crashes from numpy version conflicts
_picamera2_available = None  # None = not checked yet, True/False = checked
_Picamera2 = None  # Will hold the class if available


@dataclass
class CameraInfo:
    """Information about an available camera."""

    index: int
    name: str
    resolutions: list[tuple[int, int]] = field(default_factory=list)
    max_fps: float = 30.0
    is_available: bool = True

    def __str__(self) -> str:
        return f"{self.name} (Index {self.index})"


@dataclass
class CameraSettings:
    """Camera configuration settings."""

    camera_index: int = 0
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    exposure: int = -1  # -1 = auto
    brightness: int = 128
    contrast: int = 128
    saturation: int = 128
    auto_focus: bool = True
    auto_exposure: bool = True
    connection_timeout: float = 5.0  # Seconds to wait for camera connection
    force_backend: Optional[str] = None  # "v4l2", "picamera2", or None for auto
    pixel_format: Optional[str] = None  # "YUYV", "MJPG", or None for auto
    miniscope_mode: bool = False  # Enable miniscope-specific initialization
    buffer_size: int = 1  # Frame buffer size (1 = lowest latency)
    # Miniscope-specific controls (v4l2-ctl)
    hue: int = 0  # -32768 to 32767
    gamma: int = 0  # 0 to 65535
    gain: int = 0  # 0 to 65535
    sharpness: int = 0  # 0 to 65535
    exposure_time: int = 100  # Exposure time absolute
    focus: int = 0  # 0 to 65535
    zoom: int = 0  # 0 to 65535
    iris: int = 0  # 0 to 65535
    # Miniscope V4 hardware controls (sent via UVC I2C tunnel)
    led_power: int = 0  # 0-100 (percentage, 0=off, 100=max brightness)
    ewl_focus: int = 128  # 0-255 (electrowetting lens focus)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "camera_index": self.camera_index,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "exposure": self.exposure,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "auto_focus": self.auto_focus,
            "auto_exposure": self.auto_exposure,
            "connection_timeout": self.connection_timeout,
            "force_backend": self.force_backend,
            "pixel_format": self.pixel_format,
            "miniscope_mode": self.miniscope_mode,
            "buffer_size": self.buffer_size,
            "hue": self.hue,
            "gamma": self.gamma,
            "gain": self.gain,
            "sharpness": self.sharpness,
            "exposure_time": self.exposure_time,
            "focus": self.focus,
            "zoom": self.zoom,
            "iris": self.iris,
            "led_power": self.led_power,
            "ewl_focus": self.ewl_focus,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraSettings":
        """Deserialize from dictionary."""
        return cls(
            camera_index=data.get("camera_index", 0),
            resolution=tuple(data.get("resolution", [640, 480])),
            fps=data.get("fps", 30),
            exposure=data.get("exposure", -1),
            brightness=data.get("brightness", 128),
            contrast=data.get("contrast", 128),
            saturation=data.get("saturation", 128),
            auto_focus=data.get("auto_focus", True),
            auto_exposure=data.get("auto_exposure", True),
            connection_timeout=data.get("connection_timeout", 5.0),
            force_backend=data.get("force_backend", None),
            pixel_format=data.get("pixel_format", None),
            miniscope_mode=data.get("miniscope_mode", False),
            buffer_size=data.get("buffer_size", 1),
            hue=data.get("hue", 0),
            gamma=data.get("gamma", 0),
            gain=data.get("gain", 0),
            sharpness=data.get("sharpness", 0),
            exposure_time=data.get("exposure_time", 100),
            focus=data.get("focus", 0),
            zoom=data.get("zoom", 0),
            iris=data.get("iris", 0),
            led_power=data.get("led_power", 0),
            ewl_focus=data.get("ewl_focus", 128),
        )


class CameraState(Enum):
    """State of the camera connection."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    STREAMING = auto()
    ERROR = auto()


class CameraManager:
    """
    Manages camera devices for video capture.

    Features:
    - Enumerate available webcams
    - Thread-safe frame capture
    - Configurable camera settings
    - Frame callbacks for processing pipelines
    """

    # Common resolutions to test
    COMMON_RESOLUTIONS = [
        (640, 480),
        (720, 540),
        (800, 600),
        (1280, 720),
        (1920, 1080),
    ]

    def __init__(self):
        """Initialize the camera manager."""
        self._capture: Optional[cv2.VideoCapture] = None
        self._picamera2 = None  # Picamera2 instance for Pi cameras
        self._using_picamera2 = False
        self._using_ffmpeg = False  # FFmpeg fallback for Y800 cameras
        self._settings = CameraSettings()
        self._state = CameraState.DISCONNECTED
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_queue: Queue = Queue(maxsize=2)  # Double buffer
        self._running = False
        self._frame_callbacks: list[Callable[[np.ndarray, float], None]] = []
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_timestamp: float = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0

    @property
    def state(self) -> CameraState:
        """Current camera state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether a camera is connected."""
        return self._state in (CameraState.CONNECTED, CameraState.STREAMING)

    @property
    def is_streaming(self) -> bool:
        """Whether camera is actively streaming."""
        return self._state == CameraState.STREAMING

    @property
    def settings(self) -> CameraSettings:
        """Current camera settings."""
        return self._settings

    @property
    def current_fps(self) -> float:
        """Current frames per second."""
        return self._current_fps

    @staticmethod
    def enumerate_cameras(max_cameras: int = 10) -> list[CameraInfo]:
        """
        Enumerate all available camera devices.

        This does a quick scan - just checks if cameras can be opened,
        without trying to read frames. Format negotiation happens during connect.

        Args:
            max_cameras: Maximum number of camera indices to check

        Returns:
            List of available cameras
        """
        cameras = []
        found_indices = set()  # Track which indices we've already found

        # Suppress OpenCV warnings during enumeration
        import io
        import os
        import sys

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        # Also suppress OpenCV's internal logging
        old_log_level = os.environ.get("OPENCV_LOG_LEVEL", "")
        os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

        try:
            # Get camera names on Windows
            camera_names = _get_windows_camera_names() if sys.platform == "win32" else []

            # On Windows, try multiple backends to find all cameras
            if sys.platform == "win32":
                # Use DirectShow first - it's more stable than MSMF for enumeration
                # MSMF can cause obsensor errors and crashes with some cameras
                backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_ANY]
            else:
                backends_to_try = [_get_camera_backend()]

            for backend in backends_to_try:
                for i in range(max_cameras):
                    # Skip if we already found this camera index
                    if i in found_indices:
                        continue

                    try:
                        # Quick check - just see if camera opens
                        # Don't try to read frames here (too slow for problematic cameras)
                        cap = cv2.VideoCapture(i, backend)
                        if cap.isOpened():
                            # Camera found - add it to the list
                            # Use detected name if available, otherwise fallback
                            if i < len(camera_names) and camera_names[i]:
                                name = camera_names[i]
                            else:
                                name = f"Camera {i}"

                            # Get basic info without reading frames
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                            max_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                            cameras.append(
                                CameraInfo(
                                    index=i,
                                    name=name,
                                    resolutions=[(w, h)] + CameraManager.COMMON_RESOLUTIONS,
                                    max_fps=max_fps,
                                    is_available=True,
                                )
                            )
                            found_indices.add(i)
                            cap.release()
                            break  # Found with this backend, move to next index

                        cap.release()
                    except Exception as e:
                        # Some cameras/backends can throw exceptions during enumeration
                        logger.debug(f"Error enumerating camera {i} with backend {backend}: {e}")
        finally:
            # Restore stderr and log level
            sys.stderr = old_stderr
            if old_log_level:
                os.environ["OPENCV_LOG_LEVEL"] = old_log_level
            elif "OPENCV_LOG_LEVEL" in os.environ:
                del os.environ["OPENCV_LOG_LEVEL"]

        logger.info(f"Found {len(cameras)} camera(s)")
        return cameras

    def _try_connect_with_backend(
        self, backend: int, pixel_format: Optional[str] = None, quick_test: bool = False
    ) -> bool:
        """
        Try to connect to the camera with a specific backend and pixel format.

        Args:
            backend: OpenCV backend (e.g., cv2.CAP_V4L2, cv2.CAP_ANY)
            pixel_format: Optional pixel format to try (e.g., "MJPG", "YUY2")
            quick_test: If True, use shorter timeout for faster fallback iteration

        Returns:
            True if connection successful and frames can be read
        """
        try:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

            self._capture = cv2.VideoCapture(self._settings.camera_index, backend)

            if not self._capture.isOpened():
                logger.debug(f"Failed to open camera with backend {backend}")
                return False

            # Set buffer size (must be set early, before reading frames)
            buffer_size = self._settings.buffer_size
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

            # Try the specified pixel format if provided (overrides settings)
            format_to_use = pixel_format if pixel_format else self._settings.pixel_format
            if format_to_use:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*format_to_use[:4])
                    self._capture.set(cv2.CAP_PROP_FOURCC, fourcc)

                    # For grayscale formats, disable RGB conversion
                    if format_to_use in ("Y800", "GREY", "Y8  ", "Y16 "):
                        self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                except Exception as e:
                    logger.debug(f"Failed to set pixel format {format_to_use}: {e}")

            # Apply settings (resolution, etc.)
            self._apply_camera_settings_basic()

            # Use shorter timeout for quick tests during fallback iteration
            if quick_test:
                timeout = 1.0  # 1 second for quick test
                max_attempts = 5
            else:
                timeout = self._settings.connection_timeout
                max_attempts = max(10, int(timeout / 0.1))

            wait_time = timeout / max_attempts

            # Warmup and verify we can actually read frames
            success_count = 0
            for _attempt in range(max_attempts):
                try:
                    if self._capture.grab():
                        ret, frame = self._capture.retrieve()
                        if ret and frame is not None:
                            success_count += 1
                            if success_count >= 2:  # Reduced from 3 to 2
                                backend_name = self._get_backend_name(backend)
                                format_desc = format_to_use or "default"
                                logger.info(
                                    f"Camera working with backend {backend_name}, format {format_desc}"
                                )
                                return True
                except Exception as e:
                    logger.debug(f"Frame grab error: {e}")
                time.sleep(wait_time)

            logger.debug(f"Camera opened but failed to read frames with backend {backend}")
            self._capture.release()
            self._capture = None
            return False

        except Exception as e:
            logger.debug(f"Connection attempt failed: {e}")
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
            return False

    def _try_connect_with_fallbacks(self) -> bool:
        """
        Try connecting with multiple backends and pixel formats.

        On Windows, some USB cameras (like ANYMAZE) require specific backends
        or pixel formats to work properly. Grayscale cameras (Y800 format)
        need special handling.

        Returns:
            True if connection successful
        """
        # Get backends to try based on platform
        # Note: We exclude MSMF from grayscale attempts as it can crash with Y800 cameras
        if sys.platform == "win32":
            grayscale_backends = _get_windows_fallback_backends()  # DirectShow + AUTO only
        else:
            grayscale_backends = [_get_camera_backend()]

        # FIRST: Try specialized grayscale camera connection methods
        # This is most likely to work for scientific cameras like ANYMAZE
        # that use Y800/GREY format and don't support RGB conversion
        logger.info("Trying specialized grayscale camera connection methods...")
        for backend in grayscale_backends:
            backend_name = self._get_backend_name(backend)

            # Try the dedicated grayscale method
            try:
                logger.debug(f"Trying grayscale connection with {backend_name}")
                if self._try_connect_grayscale_camera(backend):
                    return True
            except Exception as e:
                logger.debug(f"Grayscale connection with {backend_name} failed: {e}")
                self._cleanup_capture()

            # Try the MODE-based connection
            try:
                logger.debug(f"Trying MODE-based connection with {backend_name}")
                if self._try_connect_with_mode_setting(backend):
                    return True
            except Exception as e:
                logger.debug(f"MODE-based connection with {backend_name} failed: {e}")
                self._cleanup_capture()

        # SECOND: Try standard connection with various pixel formats
        # Use only DirectShow for this - MSMF can crash with problematic cameras
        logger.info("Grayscale methods failed, trying standard connection...")

        # If user specified a pixel format, only try that
        if self._settings.pixel_format:
            formats_to_try = [self._settings.pixel_format]
        else:
            # Limit formats to try for faster iteration
            formats_to_try = [None, "MJPG", "YUY2"]

        # Only use DirectShow for standard connection attempts on Windows
        if sys.platform == "win32":
            standard_backends = [cv2.CAP_DSHOW]
        else:
            standard_backends = [_get_camera_backend()]

        for backend in standard_backends:
            backend_name = self._get_backend_name(backend)
            for pixel_format in formats_to_try:
                format_desc = pixel_format or "default"

                try:
                    logger.debug(f"Trying backend {backend_name} with format {format_desc}")
                    if self._try_connect_with_backend(backend, pixel_format, quick_test=True):
                        # Store the working format for future reference
                        if pixel_format and not self._settings.pixel_format:
                            logger.info(f"Auto-detected working pixel format: {pixel_format}")
                        return True
                except Exception as e:
                    logger.debug(f"Backend {backend_name} with {format_desc} failed: {e}")
                    self._cleanup_capture()
                    continue

        return False

    def _cleanup_capture(self) -> None:
        """Safely release the capture object."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def _try_connect_grayscale_camera(self, backend: int) -> bool:
        """
        Specialized connection method for grayscale cameras like ANYMAZE.

        Grayscale cameras (Y800/GREY format) often fail with OpenCV's default
        format negotiation because the backends request RGB32 which these
        cameras don't support.

        Args:
            backend: OpenCV backend to use (CAP_DSHOW recommended for grayscale)

        Returns:
            True if connection successful
        """
        backend_name = self._get_backend_name(backend)
        logger.info(f"Trying grayscale connection with {backend_name}...")

        try:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

            # Open the camera
            self._capture = cv2.VideoCapture(self._settings.camera_index, backend)

            if not self._capture.isOpened():
                logger.info(
                    f"Camera {self._settings.camera_index} failed to open with {backend_name}"
                )
                return False

            logger.info(f"Camera opened with {backend_name}, configuring...")

            # Disable hardware acceleration first (MSMF issue workaround)
            try:
                self._capture.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
            except Exception:
                pass  # Property might not exist in older OpenCV

            # Set resolution
            width, height = self._settings.resolution
            logger.info(f"Setting resolution to {width}x{height}...")
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._capture.set(cv2.CAP_PROP_FPS, self._settings.fps)

            # Try setting Y800/GREY format explicitly
            for fourcc_str in ["Y800", "GREY"]:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    if self._capture.set(cv2.CAP_PROP_FOURCC, fourcc):
                        logger.info(f"Set FOURCC to {fourcc_str}")
                        break
                except Exception as e:
                    logger.debug(f"Failed to set FOURCC {fourcc_str}: {e}")

            # Disable RGB conversion - critical for Y800 cameras
            self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)

            # Set buffer size - try larger buffer for problematic cameras
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)

            # Give camera time to initialize stream
            time.sleep(1.0)

            # Log current camera state
            actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
            actual_fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
            fourcc_chars = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            logger.info(
                f"Camera reports: {actual_w}x{actual_h} @ {actual_fps}fps, format={fourcc_chars}"
            )

            # Try to read frames - try both read() and grab()/retrieve()
            logger.info("Reading test frames...")
            success_count = 0

            # First try direct read()
            for attempt in range(15):
                try:
                    ret, frame = self._capture.read()
                    if ret and frame is not None and frame.size > 0:
                        success_count += 1
                        logger.info(
                            f"Frame {success_count} via read(): shape={frame.shape}, dtype={frame.dtype}"
                        )
                        if success_count >= 2:
                            logger.info(
                                f"Grayscale camera connected via {backend_name}: "
                                f"{actual_w}x{actual_h}, format={fourcc_chars}, frame_shape={frame.shape}"
                            )
                            return True
                    else:
                        if attempt < 3:
                            logger.debug(f"read() returned empty on attempt {attempt + 1}")
                except Exception as e:
                    logger.debug(f"read() error on attempt {attempt + 1}: {e}")
                time.sleep(0.1)

            # If read() failed, try grab()/retrieve()
            if success_count == 0:
                logger.info("Trying grab/retrieve method...")
                for _attempt in range(10):
                    try:
                        if self._capture.grab():
                            ret, frame = self._capture.retrieve()
                            if ret and frame is not None and frame.size > 0:
                                success_count += 1
                                logger.info(
                                    f"Frame {success_count} via grab/retrieve: shape={frame.shape}"
                                )
                                if success_count >= 2:
                                    logger.info(
                                        f"Grayscale camera connected via {backend_name}: "
                                        f"{actual_w}x{actual_h}, format={fourcc_chars}"
                                    )
                                    return True
                    except Exception as e:
                        logger.debug(f"grab/retrieve error: {e}")
                    time.sleep(0.1)

            logger.info(
                f"Failed to read frames with {backend_name} (got {success_count} successful reads)"
            )

            # Last resort: try minimal configuration (just open and read)
            logger.info("Trying minimal configuration (no property changes)...")
            self._capture.release()
            self._capture = cv2.VideoCapture(self._settings.camera_index, backend)

            if self._capture.isOpened():
                time.sleep(0.5)
                for _attempt in range(10):
                    ret, frame = self._capture.read()
                    if ret and frame is not None and frame.size > 0:
                        logger.info(
                            f"Minimal config SUCCESS: frame shape={frame.shape}, dtype={frame.dtype}"
                        )
                        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.resolution[0])
                        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.resolution[1])
                        return True
                    time.sleep(0.1)
                logger.info("Minimal configuration also failed")

            # Final attempt: try WITH RGB conversion enabled (opposite of grayscale mode)
            logger.info("Trying with forced RGB conversion...")
            self._capture.release()
            self._capture = cv2.VideoCapture(self._settings.camera_index, backend)

            if self._capture.isOpened():
                self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Force RGB conversion
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.resolution[0])
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.resolution[1])
                time.sleep(0.5)
                for _attempt in range(10):
                    ret, frame = self._capture.read()
                    if ret and frame is not None and frame.size > 0:
                        logger.info(
                            f"RGB conversion SUCCESS: frame shape={frame.shape}, dtype={frame.dtype}"
                        )
                        return True
                    time.sleep(0.1)
                logger.info("RGB conversion mode also failed")

            # Try FFmpeg DirectShow device input via OpenCV
            if sys.platform == "win32":
                camera_names = _get_windows_camera_names()
                if self._settings.camera_index < len(camera_names):
                    device_name = camera_names[self._settings.camera_index]
                    logger.info(f"Trying FFmpeg DirectShow input: video={device_name}")
                    self._capture = cv2.VideoCapture(f"video={device_name}", cv2.CAP_FFMPEG)
                    if self._capture.isOpened():
                        time.sleep(0.5)
                        for _attempt in range(10):
                            ret, frame = self._capture.read()
                            if ret and frame is not None and frame.size > 0:
                                logger.info(f"FFmpeg dshow SUCCESS: frame shape={frame.shape}")
                                return True
                            time.sleep(0.1)
                        logger.info("FFmpeg dshow input also failed")
                        self._capture.release()

                    # Final fallback: Use external FFmpeg process
                    logger.info(f"Trying external FFmpeg capture for '{device_name}'...")
                    width, height = self._settings.resolution
                    ffmpeg_cap = FFmpegCapture(device_name, width, height, self._settings.fps)
                    if ffmpeg_cap.open():
                        # Test reading frames
                        for _attempt in range(5):
                            ret, frame = ffmpeg_cap.read()
                            if ret and frame is not None:
                                logger.info(
                                    f"FFmpeg external capture SUCCESS: frame shape={frame.shape}"
                                )
                                self._capture = ffmpeg_cap  # Use FFmpegCapture as capture source
                                self._using_ffmpeg = True
                                return True
                            time.sleep(0.2)
                        logger.info("External FFmpeg capture failed to read frames")
                        ffmpeg_cap.release()
                    else:
                        logger.info(
                            "External FFmpeg capture failed to start (is FFmpeg installed?)"
                        )

            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
            self._capture = None
            return False

        except Exception as e:
            logger.debug(f"Grayscale camera connection failed: {e}")
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
            return False

    def _try_connect_with_mode_setting(self, backend: int) -> bool:
        """
        Try connecting with CAP_PROP_MODE set for grayscale.

        Some cameras respond to the MODE property for format selection.

        Args:
            backend: OpenCV backend to use

        Returns:
            True if connection successful
        """
        try:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

            self._capture = cv2.VideoCapture(self._settings.camera_index, backend)

            if not self._capture.isOpened():
                return False

            # Try different mode values that might select grayscale
            # Mode values are backend-specific but worth trying
            for mode in [0, 1, 2]:
                self._capture.set(cv2.CAP_PROP_MODE, mode)
                self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)

                # Try to read a frame
                ret, frame = self._capture.read()
                if ret and frame is not None:
                    logger.info(
                        f"Camera connected with MODE={mode} via {self._get_backend_name(backend)}"
                    )
                    return True

            self._capture.release()
            self._capture = None
            return False

        except Exception as e:
            logger.debug(f"Mode setting connection failed: {e}")
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
            return False

    @staticmethod
    def _get_backend_name(backend: int) -> str:
        """Get human-readable name for a backend ID."""
        backend_names = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "MediaFoundation",
            cv2.CAP_FFMPEG: "FFmpeg",
            cv2.CAP_V4L2: "V4L2",
            cv2.CAP_ANY: "Auto",
        }
        return backend_names.get(backend, f"Backend_{backend}")

    def _try_connect_picamera2(self) -> bool:
        """
        Try to connect using picamera2 (for Raspberry Pi camera modules).

        Returns:
            True if connection successful and frames can be read
        """
        global _picamera2_available, _Picamera2

        # Lazy import of picamera2 to avoid crashes from numpy version conflicts
        if _picamera2_available is None:
            try:
                from picamera2 import Picamera2

                _Picamera2 = Picamera2
                _picamera2_available = True
                logger.debug("picamera2 imported successfully")
            except (ImportError, ValueError) as e:
                logger.debug(f"picamera2 not available: {e}")
                _picamera2_available = False

        if not _picamera2_available:
            logger.debug("picamera2 not available")
            return False

        try:
            # Clean up any existing capture
            if self._capture is not None:
                self._capture.release()
                self._capture = None

            if self._picamera2 is not None:
                self._picamera2.close()
                self._picamera2 = None

            # Create Picamera2 instance
            self._picamera2 = _Picamera2()

            # Configure for video capture
            width, height = self._settings.resolution
            config = self._picamera2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": self._settings.fps},
            )
            self._picamera2.configure(config)

            # Start the camera
            self._picamera2.start()
            time.sleep(0.5)  # Give camera time to warm up

            # Verify we can capture frames
            for _ in range(5):
                frame = self._picamera2.capture_array()
                if frame is not None and frame.size > 0:
                    # Update resolution to actual
                    self._settings.resolution = (frame.shape[1], frame.shape[0])
                    self._using_picamera2 = True
                    logger.info(f"picamera2 working: {self._settings.resolution}")
                    return True
                time.sleep(0.1)

            # Failed to get frames
            self._picamera2.close()
            self._picamera2 = None
            return False

        except Exception as e:
            logger.debug(f"picamera2 failed: {e}")
            if self._picamera2 is not None:
                try:
                    self._picamera2.close()
                except Exception:
                    pass
                self._picamera2 = None
            return False

    def connect(self, settings: Optional[CameraSettings] = None) -> bool:
        """
        Connect to a camera with given settings.

        Args:
            settings: Camera settings to apply, or None for defaults

        Returns:
            True if connection successful
        """
        if self._state == CameraState.STREAMING:
            self.stop_streaming()

        if self._capture is not None:
            self._capture.release()

        self._state = CameraState.CONNECTING

        if settings is not None:
            self._settings = settings

        try:
            # Miniscope wake-up sequence (must happen BEFORE opening the camera)
            if self._settings.miniscope_mode:
                logger.info("Miniscope mode enabled - running wake-up sequence")
                _wake_up_miniscope(self._settings.camera_index)

            # Use platform-appropriate backend
            backend = _get_camera_backend()
            force_backend = self._settings.force_backend

            # Check if user forced a specific backend
            if force_backend == "picamera2":
                logger.info("Forcing picamera2 backend")
                if self._try_connect_picamera2():
                    self._state = CameraState.CONNECTED
                    logger.info("Connected to Pi camera via picamera2")
                    return True
                self._state = CameraState.ERROR
                return False
            elif force_backend == "v4l2":
                logger.info("Forcing V4L2 backend")
                backend = cv2.CAP_V4L2
            elif force_backend == "msmf":
                logger.info("Forcing Microsoft Media Foundation backend")
                backend = cv2.CAP_MSMF
            elif force_backend == "dshow":
                logger.info("Forcing DirectShow backend")
                backend = cv2.CAP_DSHOW

            # For Windows with forced backend, try grayscale methods
            if sys.platform == "win32" and force_backend in ("msmf", "dshow"):
                logger.info(f"Trying {force_backend.upper()} with grayscale camera methods...")
                connected = False

                if self._try_connect_grayscale_camera(backend):
                    logger.info(f"Connected via {force_backend.upper()} grayscale method")
                    connected = True
                elif self._try_connect_with_mode_setting(backend):
                    logger.info(f"Connected via {force_backend.upper()} MODE setting")
                    connected = True
                elif self._try_connect_with_backend(backend):
                    logger.info(f"Connected via {force_backend.upper()} standard method")
                    connected = True
                # Fallback: try CAP_ANY
                elif self._try_connect_grayscale_camera(cv2.CAP_ANY):
                    logger.info("Connected via auto-detect grayscale method")
                    connected = True
                elif self._try_connect_with_backend(cv2.CAP_ANY):
                    logger.info("Connected via auto-detect standard method")
                    connected = True

                if connected:
                    pass  # Continue to miniscope controls
                else:
                    self._state = CameraState.ERROR
                    logger.error(f"Failed to connect with {force_backend.upper()}")
                    return False

            # On Raspberry Pi, only try picamera2 for camera index 0 (Pi Camera module)
            # USB cameras (miniscopes, webcams) should use V4L2 directly
            if _is_raspberry_pi() and force_backend is None:
                if self._settings.camera_index == 0:
                    logger.info("Raspberry Pi detected, trying picamera2 for camera 0")
                    if self._try_connect_picamera2():
                        self._state = CameraState.CONNECTED
                        logger.info("Connected to Pi camera via picamera2")
                        return True
                    logger.info("picamera2 failed, falling back to V4L2")
                else:
                    logger.info(
                        f"Camera index {self._settings.camera_index} - using V4L2 (USB camera)"
                    )

            # Try connecting with the preferred backend
            # On Windows, use fallback logic to try multiple backends/formats
            if sys.platform == "win32" and force_backend is None:
                logger.info("Windows detected - trying multiple backends and pixel formats")
                if not self._try_connect_with_fallbacks():
                    self._state = CameraState.ERROR
                    logger.error("Failed to connect camera with any backend/format combination")
                    return False
            elif sys.platform == "win32" and force_backend in ("msmf", "dshow"):
                pass  # Already handled above
            elif not self._try_connect_with_backend(backend):
                # If V4L2 failed on Linux, try CAP_ANY as fallback
                if backend == cv2.CAP_V4L2:
                    logger.info("V4L2 failed, trying auto-detect backend")
                    if not self._try_connect_with_backend(cv2.CAP_ANY):
                        self._state = CameraState.ERROR
                        return False
                else:
                    self._state = CameraState.ERROR
                    return False

            # Apply miniscope controls after camera is open
            if self._settings.miniscope_mode:
                _apply_miniscope_controls(self._settings.camera_index, self._settings)
                # Apply hardware controls (LED, EWL focus)
                if sys.platform == "win32" and self._capture is not None:
                    # Windows: Use OpenCV-based control
                    _init_miniscope_ewl_opencv(self._capture)
                    _set_miniscope_led_opencv(self._capture, self._settings.led_power)
                    _set_miniscope_ewl_opencv(self._capture, self._settings.ewl_focus)
                else:
                    # Linux: Use v4l2-ctl based control
                    _apply_miniscope_hardware_controls(self._settings.camera_index, self._settings)

            self._state = CameraState.CONNECTED
            logger.info(f"Connected to camera {self._settings.camera_index}")
            return True

        except Exception as e:
            logger.error(f"Camera connection error: {e}")
            self._state = CameraState.ERROR
            return False

    def disconnect(self) -> None:
        """Disconnect from the current camera."""
        if self._state == CameraState.STREAMING:
            self.stop_streaming()

        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
            self._using_ffmpeg = False

        if self._picamera2 is not None:
            try:
                self._picamera2.close()
            except Exception:
                pass
            self._picamera2 = None
            self._using_picamera2 = False

        self._state = CameraState.DISCONNECTED
        self._last_frame = None
        logger.info("Camera disconnected")

    def start_streaming(self) -> bool:
        """
        Start capturing frames in background thread.

        Returns:
            True if streaming started successfully
        """
        if self._state == CameraState.STREAMING:
            return True

        # Check if we have a valid capture source
        has_capture = self._capture is not None and self._capture.isOpened()
        has_picamera2 = self._using_picamera2 and self._picamera2 is not None

        if not has_capture and not has_picamera2:
            logger.error("Cannot start streaming: camera not connected")
            return False

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraCapture"
        )
        self._capture_thread.start()

        self._state = CameraState.STREAMING
        logger.info("Camera streaming started")
        return True

    def stop_streaming(self) -> None:
        """Stop frame capture."""
        if not self._running:
            return

        self._running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

        if self._state == CameraState.STREAMING:
            self._state = CameraState.CONNECTED

        logger.info("Camera streaming stopped")

    def get_frame(self) -> Optional[tuple[np.ndarray, float]]:
        """
        Get the latest frame (non-blocking).

        Returns:
            Tuple of (frame, timestamp) or None if no frame available
        """
        try:
            return self._frame_queue.get_nowait()
        except Empty:
            # Return last frame if available
            if self._last_frame is not None:
                return (self._last_frame.copy(), self._last_timestamp)
            return None

    def on_frame(self, callback: Callable[[np.ndarray, float], None]) -> None:
        """
        Register callback for each new frame.

        Args:
            callback: Function called with (frame, timestamp) for each frame
        """
        self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[np.ndarray, float], None]) -> None:
        """Remove a frame callback."""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    def apply_settings(self, settings: CameraSettings) -> None:
        """
        Apply new camera settings.

        Args:
            settings: New settings to apply
        """
        old_index = self._settings.camera_index
        self._settings = settings

        # If camera index changed, reconnect
        if old_index != settings.camera_index:
            was_streaming = self._state == CameraState.STREAMING
            self.disconnect()
            self.connect(settings)
            if was_streaming:
                self.start_streaming()
        elif self._capture is not None and self._capture.isOpened():
            self._apply_camera_settings()

    def _apply_camera_settings_basic(self) -> None:
        """Apply basic camera settings (resolution, fps) without changing pixel format."""
        if self._capture is None:
            return

        # Resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.resolution[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.resolution[1])

        # FPS
        self._capture.set(cv2.CAP_PROP_FPS, self._settings.fps)

        # Verify resolution
        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != self._settings.resolution:
            logger.debug(f"Requested {self._settings.resolution}, got ({actual_w}, {actual_h})")
            self._settings.resolution = (actual_w, actual_h)

    def _apply_camera_settings(self) -> None:
        """Apply current settings to the camera."""
        if self._capture is None:
            return

        # Set pixel format FIRST (before resolution) - important for miniscopes
        if self._settings.pixel_format:
            try:
                fourcc = cv2.VideoWriter_fourcc(*self._settings.pixel_format)
                self._capture.set(cv2.CAP_PROP_FOURCC, fourcc)
                logger.debug(f"Set pixel format to {self._settings.pixel_format}")
            except Exception as e:
                logger.warning(f"Failed to set pixel format {self._settings.pixel_format}: {e}")
        elif sys.platform == "linux":
            # On Linux with V4L2, try MJPEG format for better compatibility
            # Some cameras don't support it, so we don't fail if it doesn't work
            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._capture.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass  # Camera may not support MJPEG, that's okay

        # Resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.resolution[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.resolution[1])

        # FPS
        self._capture.set(cv2.CAP_PROP_FPS, self._settings.fps)

        # Exposure
        if self._settings.auto_exposure:
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        else:
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self._capture.set(cv2.CAP_PROP_EXPOSURE, self._settings.exposure)

        # Brightness, contrast, saturation
        self._capture.set(cv2.CAP_PROP_BRIGHTNESS, self._settings.brightness)
        self._capture.set(cv2.CAP_PROP_CONTRAST, self._settings.contrast)
        self._capture.set(cv2.CAP_PROP_SATURATION, self._settings.saturation)

        # Auto focus
        if self._settings.auto_focus:
            self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Verify resolution
        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != self._settings.resolution:
            logger.warning(f"Requested {self._settings.resolution}, got ({actual_w}, {actual_h})")
            self._settings.resolution = (actual_w, actual_h)

        logger.debug(
            f"Applied camera settings: {self._settings.resolution} @ {self._settings.fps}fps"
        )

    def set_led_power(self, power_percent: int) -> bool:
        """
        Set Miniscope LED power (0-100%).

        Can be called while streaming to adjust LED brightness on-the-fly.

        Args:
            power_percent: LED power 0-100 (0=off, 100=max)

        Returns:
            True if LED was set successfully
        """
        if not self._settings.miniscope_mode:
            logger.warning("LED control requires miniscope mode to be enabled")
            return False

        self._settings.led_power = power_percent

        # Use OpenCV method on Windows when capture is available
        if sys.platform == "win32" and self._capture is not None:
            return _set_miniscope_led_opencv(self._capture, power_percent)
        else:
            return _set_miniscope_led(self._settings.camera_index, power_percent)

    def set_ewl_focus(self, focus_value: int) -> bool:
        """
        Set Miniscope electrowetting lens focus (0-255).

        Can be called while streaming to adjust focus on-the-fly.

        Args:
            focus_value: Focus position 0-255

        Returns:
            True if focus was set successfully
        """
        if not self._settings.miniscope_mode:
            logger.warning("EWL focus control requires miniscope mode to be enabled")
            return False

        self._settings.ewl_focus = focus_value

        # Use OpenCV method on Windows when capture is available
        if sys.platform == "win32" and self._capture is not None:
            return _set_miniscope_ewl_opencv(self._capture, focus_value)
        else:
            return _set_miniscope_ewl_focus(self._settings.camera_index, focus_value)

    def init_ewl(self) -> bool:
        """
        Initialize the EWL driver.

        Should be called after connecting to the miniscope and before
        adjusting EWL focus for the first time.

        Returns:
            True if initialization was successful
        """
        if not self._settings.miniscope_mode:
            logger.warning("EWL init requires miniscope mode to be enabled")
            return False

        # Only available on Windows with OpenCV
        if sys.platform == "win32" and self._capture is not None:
            return _init_miniscope_ewl_opencv(self._capture)
        else:
            # Linux doesn't need explicit init
            return True

    def _capture_loop(self) -> None:
        """Background thread for frame capture."""
        logger.debug("Capture loop started")
        consecutive_failures = 0
        max_failures_before_log = 30  # Only log every 30 failures to avoid spam
        miniscope_check_interval = 30  # Check brightness every N frames
        miniscope_frame_count = 0

        while self._running:
            frame = None

            # Handle picamera2
            if self._using_picamera2 and self._picamera2 is not None:
                try:
                    frame = self._picamera2.capture_array()
                    # picamera2 returns RGB, convert to BGR for OpenCV compatibility
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    consecutive_failures += 1
                    if (
                        consecutive_failures == 1
                        or consecutive_failures % max_failures_before_log == 0
                    ):
                        logger.warning(f"picamera2 capture failed: {e}")
                    time.sleep(0.01)
                    continue
            # Handle OpenCV capture
            elif self._capture is not None and self._capture.isOpened():
                # Use grab() + retrieve() which works better with V4L2 on Linux
                if not self._capture.grab():
                    consecutive_failures += 1
                    if (
                        consecutive_failures == 1
                        or consecutive_failures % max_failures_before_log == 0
                    ):
                        logger.warning(f"Failed to grab frame (attempt {consecutive_failures})")
                    time.sleep(0.01)
                    continue

                ret, frame = self._capture.retrieve()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if (
                        consecutive_failures == 1
                        or consecutive_failures % max_failures_before_log == 0
                    ):
                        logger.warning(f"Failed to retrieve frame (attempt {consecutive_failures})")
                    time.sleep(0.01)
                    continue
            else:
                time.sleep(0.1)
                continue

            if frame is None:
                consecutive_failures += 1
                time.sleep(0.01)
                continue

            consecutive_failures = 0  # Reset on success

            # Normalize frame format for grayscale cameras (Y800/GREY)
            # These cameras output 2D frames (height, width) instead of 3D (height, width, 3)
            # Convert to 3-channel BGR for consistency with the rest of the pipeline
            if len(frame.shape) == 2:
                # Single channel grayscale - convert to 3-channel BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                # Single channel with explicit dimension - squeeze and convert
                frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)

            # Miniscope watchdog: kick LED if image goes dark
            if self._settings.miniscope_mode:
                miniscope_frame_count += 1
                if miniscope_frame_count % miniscope_check_interval == 0:
                    mean_brightness = np.mean(frame)
                    if mean_brightness < 1.0:
                        logger.warning(
                            f"Miniscope darkness detected ({mean_brightness:.2f}) - kicking LED"
                        )
                        _wake_up_miniscope(self._settings.camera_index)

            timestamp = time.time()

            # Update FPS counter
            self._fps_counter += 1
            elapsed = timestamp - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = timestamp

            # Store last frame
            with self._lock:
                self._last_frame = frame.copy()
                self._last_timestamp = timestamp

            # Update queue (drop old frames if full)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
            self._frame_queue.put((frame, timestamp))

            # Notify callbacks
            for callback in self._frame_callbacks:
                try:
                    callback(frame, timestamp)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")

        logger.debug("Capture loop ended")

    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame (blocking).

        Useful for taking snapshots without streaming.

        Returns:
            Frame as numpy array (BGR format), or None if capture failed
        """
        if self._capture is None or not self._capture.isOpened():
            return None

        ret, frame = self._capture.read()
        if not ret or frame is None:
            return None

        # Normalize grayscale frames to BGR for consistency
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)

        return frame

    def get_property(self, prop_id: int) -> float:
        """Get a camera property value."""
        if self._capture is None:
            return 0.0
        return self._capture.get(prop_id)

    def set_property(self, prop_id: int, value: float) -> bool:
        """Set a camera property value."""
        if self._capture is None:
            return False
        return self._capture.set(prop_id, value)
