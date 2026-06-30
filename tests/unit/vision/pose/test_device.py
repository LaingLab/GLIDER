"""Tests for device resolution.

Tests that call ``resolve_device`` need torch (imported lazily inside the
function); we mark them with ``requires_torch``. Format and diagnose tests
work with monkeypatched gpu_info() and run regardless.
"""

from __future__ import annotations

import importlib.util

import pytest

from glider.vision.pose import device as device_mod
from glider.vision.pose.device import (
    format_gpu_info,
    gpu_info,
    require_gpu_or_raise,
    resolve_device,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)


@requires_torch
def test_resolve_device_cpu_explicit():
    assert resolve_device("cpu") == "cpu"


@requires_torch
def test_resolve_device_unknown_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


@requires_torch
def test_resolve_device_auto_selects_something(monkeypatch):
    """auto should always pick *something* — at minimum cpu."""
    out = resolve_device("auto")
    assert out in {"cuda:0", "mps", "cpu"} or out.startswith("cuda:")


@requires_torch
def test_resolve_device_none_equals_auto():
    assert resolve_device(None) == resolve_device("auto")


@requires_torch
def test_resolve_device_cuda_unavailable_raises(monkeypatch):
    """Asking for CUDA when torch reports no CUDA must raise."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


@requires_torch
def test_resolve_device_cuda_index_out_of_range(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="only 1"):
        resolve_device("cuda:5")


@requires_torch
def test_resolve_device_mps_unavailable_raises(monkeypatch):
    monkeypatch.setattr(device_mod, "_mps_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


@requires_torch
def test_resolve_device_int_becomes_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device(0) == "cuda:0"
    assert resolve_device(1) == "cuda:1"


@requires_torch
def test_require_gpu_raises_on_cpu(monkeypatch):
    """If the only available device is CPU, require_gpu must raise."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_mod, "_mps_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU was required"):
        resolve_device(None, require_gpu=True)
    with pytest.raises(RuntimeError, match="GPU was required"):
        require_gpu_or_raise("cpu")


@requires_torch
def test_require_gpu_passes_on_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device("cuda", require_gpu=True) == "cuda"


def test_gpu_info_keys():
    info = gpu_info()
    assert {"torch_available", "cuda_available", "mps_available"}.issubset(info.keys())


def test_format_gpu_info_no_torch(monkeypatch):
    monkeypatch.setattr(
        device_mod,
        "gpu_info",
        lambda: {
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
            "mps_available": False,
        },
    )
    text = format_gpu_info()
    assert "PyTorch is not installed" in text


def test_format_gpu_info_cpu_only(monkeypatch):
    info = {
        "torch_available": True,
        "torch_version": "2.3.0",
        "torch_build_cuda": "12.1",
        "torch_cpu_only_build": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": False,
        "nvidia_driver_visible": False,
    }
    text = format_gpu_info(info)
    assert "PyTorch 2.3.0" in text
    assert "CUDA available: no" in text
    assert "MPS available: no" in text


def test_format_gpu_info_with_cuda():
    info = {
        "torch_available": True,
        "torch_version": "2.3.0",
        "torch_build_cuda": "12.1",
        "torch_cpu_only_build": False,
        "cuda_available": True,
        "cudnn_version": 8907,
        "cuda_device_count": 1,
        "cuda_devices": [
            {
                "index": 0,
                "name": "NVIDIA A100",
                "total_memory_gb": 40.0,
                "compute_capability": "8.0",
            }
        ],
        "mps_available": False,
        "nvidia_driver_visible": True,
    }
    text = format_gpu_info(info)
    assert "NVIDIA A100" in text
    assert "40.0 GB" in text
    assert "sm_80" in text
    assert "CUDA 12.1" in text


def test_format_gpu_info_cpu_only_build_warns():
    info = {
        "torch_available": True,
        "torch_version": "2.3.0+cpu",
        "torch_build_cuda": None,
        "torch_cpu_only_build": True,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": False,
        "nvidia_driver_visible": True,
    }
    text = format_gpu_info(info)
    assert "CPU-only torch build" in text
    assert "download.pytorch.org" in text


def test_diagnose_flags_cpu_only_with_visible_gpu(monkeypatch):
    """The classic 'gpu shows up in nvidia-smi but torch is +cpu' case."""
    from glider.vision.pose import device as device_mod

    monkeypatch.setattr(
        device_mod,
        "gpu_info",
        lambda: {
            "torch_available": True,
            "torch_version": "2.3.0+cpu",
            "torch_build_cuda": None,
            "torch_cpu_only_build": True,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
            "mps_available": False,
            "nvidia_driver_visible": True,
        },
    )
    checks = device_mod.diagnose()
    statuses = {name: (status, detail) for name, status, detail in checks}
    assert statuses["torch CUDA build"][0] == "warn"
    assert statuses["torch.cuda.is_available"][0] == "fail"
    assert "CPU-only" in statuses["torch.cuda.is_available"][1]


def test_diagnose_no_gpu_machine(monkeypatch):
    """Pure CPU laptop — diagnose should not flag failures."""
    from glider.vision.pose import device as device_mod

    monkeypatch.setattr(
        device_mod,
        "gpu_info",
        lambda: {
            "torch_available": True,
            "torch_version": "2.3.0",
            "torch_build_cuda": "12.1",
            "torch_cpu_only_build": False,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
            "mps_available": False,
            "nvidia_driver_visible": False,
        },
    )
    checks = device_mod.diagnose()
    statuses = {name: status for name, status, _ in checks}
    assert "fail" not in statuses.values()
    assert statuses["torch.cuda.is_available"] == "info"


def test_diagnose_healthy_cuda(monkeypatch):
    from glider.vision.pose import device as device_mod

    monkeypatch.setattr(
        device_mod,
        "gpu_info",
        lambda: {
            "torch_available": True,
            "torch_version": "2.3.0+cu121",
            "torch_build_cuda": "12.1",
            "torch_cpu_only_build": False,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_devices": [
                {
                    "index": 0,
                    "name": "RTX 4090",
                    "total_memory_gb": 24.0,
                    "compute_capability": "8.9",
                }
            ],
            "cudnn_version": 8907,
            "mps_available": False,
            "nvidia_driver_visible": True,
        },
    )
    checks = device_mod.diagnose()
    statuses = {name: status for name, status, _ in checks}
    assert all(s in {"ok", "info"} for s in statuses.values())
