"""Device resolution and GPU verification.

Ultralytics' default device selection is permissive: if CUDA isn't available
it silently falls back to CPU, which means a 30 fps inference job can secretly
take 10x longer than it should without any error. This module gives you:

* :func:`resolve_device` — normalize a device spec (None / "auto" / "cuda" /
  "cuda:0" / int / "mps" / "cpu") and validate it's actually available.
* :func:`require_gpu_or_raise` — convenience wrapper that errors if the
  resolved device is CPU.
* :func:`gpu_info` — return a dict describing the available accelerators
  (CUDA devices, VRAM, driver, MPS availability, torch version).

These are imported lazily so the rest of the package doesn't pay the torch
import cost unless inference is actually requested.
"""

from __future__ import annotations

import sys
from typing import Any


def _is_gpu_device(device: str) -> bool:
    return device.startswith("cuda") or device == "mps"


def resolve_device(
    device: str | int | None = None,
    *,
    require_gpu: bool = False,
) -> str:
    """Normalize a device spec to a concrete torch device string.

    Parameters
    ----------
    device
        - ``None`` or ``"auto"`` — pick best available (cuda > mps > cpu)
        - ``"cuda"``, ``"cuda:N"``, or an ``int`` — specific CUDA device
        - ``"mps"`` — Apple Silicon Metal Performance Shaders
        - ``"cpu"`` — force CPU
    require_gpu
        If True, raise ``RuntimeError`` when the resolved device is CPU.

    Returns
    -------
    str
        Concrete device string usable by torch and Ultralytics
        (e.g. ``"cuda:0"``).

    Raises
    ------
    RuntimeError
        If a GPU device was explicitly requested but isn't available, or if
        ``require_gpu`` is True and only CPU is available.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required to resolve devices. " "Install with: pip install glider[vision]"
        ) from e

    # Normalize input.
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            resolved = "cuda:0"
        elif _mps_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    elif isinstance(device, int):
        resolved = f"cuda:{device}"
    else:
        resolved = str(device).strip().lower()
        if resolved == "gpu":
            resolved = (
                "cuda:0" if torch.cuda.is_available() else ("mps" if _mps_available() else "cpu")
            )

    # Validate availability.
    if resolved.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device {resolved!r}, but torch.cuda.is_available() "
                "is False. Check your PyTorch install (was it built with CUDA?) "
                "and your NVIDIA driver. Use device='cpu' to fall back."
            )
        # Validate index if "cuda:N" with N out of range.
        if ":" in resolved:
            idx = int(resolved.split(":", 1)[1])
            n = torch.cuda.device_count()
            if idx >= n:
                raise RuntimeError(f"Requested cuda:{idx}, but only {n} CUDA device(s) detected.")
    elif resolved == "mps":
        if not _mps_available():
            raise RuntimeError(
                "Requested MPS, but torch.backends.mps.is_available() is False. "
                "MPS requires macOS 12.3+ on Apple Silicon."
            )
    elif resolved != "cpu":
        raise ValueError(f"unrecognized device: {device!r}")

    if require_gpu and not _is_gpu_device(resolved):
        raise RuntimeError(
            "GPU was required but none is available. "
            "torch.cuda.is_available() is False and MPS is unavailable. "
            "Either install a CUDA-enabled PyTorch (https://pytorch.org/get-started/locally/), "
            "use a machine with a GPU, or pass require_gpu=False to allow CPU inference."
        )

    return resolved


def require_gpu_or_raise(device: str | int | None = None) -> str:
    """Resolve ``device`` and raise unless it lands on a GPU."""
    return resolve_device(device, require_gpu=True)


def gpu_info() -> dict[str, Any]:
    """Return a dict describing the accelerators visible to torch.

    Includes diagnostics that help distinguish "no GPU on this machine" from
    "you installed a CPU-only torch wheel and the GPU is invisible to torch":

    * ``torch_cpu_only_build`` — the installed torch wheel was built without
      CUDA support (``torch.version.cuda is None`` or version string contains
      ``+cpu``). Even on a machine with a working CUDA driver, this build will
      report ``cuda_available=False``.
    * ``torch_build_cuda`` — the CUDA version torch was built against
      (e.g. ``"12.1"``); ``None`` for CPU-only builds.
    * ``nvidia_driver_visible`` — whether ``nvidia-smi`` runs successfully.
      Useful for catching the case where the OS sees the GPU but the Python
      env doesn't.
    """
    info: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "torch_build_cuda": None,
        "torch_cpu_only_build": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": False,
        "nvidia_driver_visible": _nvidia_smi_available(),
    }

    try:
        import torch
    except ImportError:
        return info

    info["torch_available"] = True
    info["torch_version"] = torch.__version__
    info["torch_build_cuda"] = getattr(torch.version, "cuda", None)
    info["torch_cpu_only_build"] = info["torch_build_cuda"] is None or "+cpu" in str(
        torch.__version__
    )
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
    info["mps_available"] = _mps_available()

    if info["cuda_available"]:
        info["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
        for i in range(info["cuda_device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

    return info


def _nvidia_smi_available() -> bool:
    """Return True if ``nvidia-smi`` runs and reports at least one GPU."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def diagnose() -> list[tuple[str, str, str]]:
    """Return a list of ``(check, status, detail)`` triples diagnosing GPU setup.

    ``status`` is one of ``"ok"``, ``"warn"``, ``"fail"``, ``"info"``. Used by
    the GLIDER diagnostics interface to print a friendly report.
    """
    info = gpu_info()
    checks: list[tuple[str, str, str]] = []

    # Torch installed?
    if not info["torch_available"]:
        checks.append(
            ("torch installed", "fail", "PyTorch is not importable. `pip install glider[vision]`.")
        )
        return checks
    checks.append(("torch installed", "ok", info["torch_version"]))

    # CPU-only build? On macOS every torch wheel is CUDA-less by definition
    # (MPS is the accelerator), so "CPU-only, reinstall from cu121" would be
    # misleading advice — report the platform fact instead.
    if sys.platform == "darwin":
        checks.append(
            (
                "torch CUDA build",
                "info",
                "macOS build — CUDA not applicable; MPS is the accelerator",
            )
        )
    elif info["torch_cpu_only_build"]:
        checks.append(
            (
                "torch CUDA build",
                "warn",
                "CPU-only torch wheel detected. Reinstall from "
                "https://download.pytorch.org/whl/cu121 (or matching CUDA index).",
            )
        )
    else:
        checks.append(
            (
                "torch CUDA build",
                "ok",
                f"built against CUDA {info['torch_build_cuda']}",
            )
        )

    # Driver visible?
    if info["nvidia_driver_visible"]:
        checks.append(("nvidia-smi", "ok", "driver responding"))
    else:
        checks.append(
            (
                "nvidia-smi",
                "info",
                "not found or no GPU visible to OS (fine on Mac/CPU box)",
            )
        )

    # Torch sees CUDA?
    if info["cuda_available"]:
        names = ", ".join(d["name"] for d in info["cuda_devices"])
        checks.append(
            (
                "torch.cuda.is_available",
                "ok",
                f"{info['cuda_device_count']} device(s): {names}",
            )
        )
    elif info["nvidia_driver_visible"] and info["torch_cpu_only_build"]:
        checks.append(
            (
                "torch.cuda.is_available",
                "fail",
                "GPU visible to OS but torch is CPU-only — see warning above.",
            )
        )
    elif info["nvidia_driver_visible"] and not info["torch_cpu_only_build"]:
        checks.append(
            (
                "torch.cuda.is_available",
                "fail",
                "GPU visible to OS, CUDA-enabled torch, but cuda.is_available()=False. "
                "Likely a CUDA-version mismatch between torch and driver, or "
                "CUDA_VISIBLE_DEVICES is set to an empty string.",
            )
        )
    else:
        checks.append(("torch.cuda.is_available", "info", "no CUDA GPU on this machine"))

    # MPS?
    if info["mps_available"]:
        checks.append(("MPS (Apple Silicon)", "ok", "available"))

    return checks


def _mps_available() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
    except ImportError:
        return False


def format_gpu_info(info: dict[str, Any] | None = None) -> str:
    """Pretty-print ``gpu_info()`` output as a multi-line string."""
    if info is None:
        info = gpu_info()
    if not info["torch_available"]:
        return "PyTorch is not installed. `pip install glider[vision]` to enable inference."

    lines = [f"PyTorch {info['torch_version']}"]
    if sys.platform == "darwin":
        # macOS torch is never a CUDA build; warning about it is noise.
        lines.append("torch macOS build — CUDA not applicable; MPS is the accelerator.")
    elif info["torch_cpu_only_build"]:
        lines.append("⚠  CPU-only torch build (torch.version.cuda is None or +cpu).")
        lines.append("   Reinstall from https://download.pytorch.org/whl/cu121 to enable CUDA.")
    else:
        lines.append(f"torch built against CUDA {info['torch_build_cuda']}")
    if info["cuda_available"]:
        lines.append(
            f"CUDA available: {info['cuda_device_count']} device(s) "
            f"(cuDNN {info.get('cudnn_version')})"
        )
        for d in info["cuda_devices"]:
            lines.append(
                f"  [{d['index']}] {d['name']}  "
                f"({d['total_memory_gb']} GB, sm_{d['compute_capability'].replace('.', '')})"
            )
    else:
        lines.append("CUDA available: no")
    lines.append(f"MPS available: {'yes' if info['mps_available'] else 'no'}")
    lines.append(f"nvidia-smi visible: {'yes' if info['nvidia_driver_visible'] else 'no'}")
    return "\n".join(lines)
