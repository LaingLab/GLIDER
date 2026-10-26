"""The CUDA PyTorch offer (#221): when it is made, and what it installs.

Nothing here touches the network, nvidia-smi or the real torch: every probe is
injected, so the tests mean the same on a GPU workstation and a CI runner.
"""

from __future__ import annotations

import pytest

from glider.vision import gpu_setup
from glider.vision.gpu_setup import (
    candidate_tags,
    find_wheel_tag,
    install_command,
    install_cuda_torch,
    needs_cuda_torch,
    parse_driver_cuda,
)

OLD_HEADER = "| NVIDIA-SMI 550.54   Driver Version: 550.54   CUDA Version: 12.4     |"
NEW_HEADER = "| NVIDIA-SMI 610.88     KMD Version: 610.88     CUDA UMD Version: 13.3     |"


# --- The driver --------------------------------------------------------------


@pytest.mark.parametrize(("header", "expected"), [(OLD_HEADER, (12, 4)), (NEW_HEADER, (13, 3))])
def test_both_nvidia_smi_header_styles_are_read(header, expected):
    assert parse_driver_cuda(header) == expected


def test_output_without_a_cuda_version_is_no_driver():
    assert parse_driver_cuda("NVIDIA-SMI has failed because it couldn't communicate") is None


def test_tags_never_exceed_what_the_driver_runs():
    tags = candidate_tags((12, 4))
    assert tags[0] == "cu124"
    assert "cu126" not in tags
    assert tags[-1] == "cu118"


def test_tags_cross_a_major_version():
    assert candidate_tags((13, 1))[:4] == ["cu131", "cu130", "cu129", "cu128"]


# --- When it is offered ------------------------------------------------------


def _needs(**overrides):
    probes = {
        "torch_info": lambda: ("2.10.0", None),
        "driver": lambda: (12, 8),
        "platform_name": "win32",
        "frozen": False,
    }
    probes.update(overrides)
    return needs_cuda_torch(**probes)


def test_offered_for_cpu_torch_on_a_gpu_machine():
    assert _needs() == (12, 8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"torch_info": lambda: ("2.10.0+cu128", "12.8")},  # already CUDA
        {"torch_info": lambda: None},  # no torch at all
        {"driver": lambda: None},  # no NVIDIA driver
        {"platform_name": "darwin"},  # no CUDA on a Mac
        {"frozen": True},  # the installer build ships CUDA torch already
    ],
)
def test_not_offered(overrides):
    assert _needs(**overrides) is None


def test_the_driver_is_not_asked_when_torch_already_has_cuda():
    """nvidia-smi is the slow probe; a working install must never start it."""

    def _driver():
        raise AssertionError("nvidia-smi was run")

    assert _needs(torch_info=lambda: ("2.10.0+cu128", "12.8"), driver=_driver) is None


# --- Which build -------------------------------------------------------------


def _index(*wheels):
    return "".join(f'<a href="/whl/{w}#sha256=abc">{w}</a>\n' for w in wheels)


def test_the_newest_tag_with_a_matching_wheel_wins():
    pages = {
        # cu129 exists but has no Windows wheel of this version.
        "cu129": _index("torch-2.10.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl"),
        "cu128": _index("torch-2.10.0%2Bcu128-cp312-cp312-win_amd64.whl"),
        "cu126": _index("torch-2.10.0%2Bcu126-cp312-cp312-win_amd64.whl"),
    }
    fetched = []

    def _fetch(url):
        fetched.append(url)
        return pages.get(url.rstrip("/").split("/")[-2])

    tag = find_wheel_tag("2.10.0", (12, 9), fetch=_fetch, python_tag="cp312", platform_name="win32")
    assert tag == "cu128"
    assert not any("cu126" in url for url in fetched), "kept looking after a match"


def test_the_local_version_of_the_installed_torch_is_ignored():
    page = _index("torch-2.10.0+cu126-cp311-cp311-win_amd64.whl")
    tag = find_wheel_tag(
        "2.10.0+cpu", (12, 6), fetch=lambda _url: page, python_tag="cp311", platform_name="win32"
    )
    assert tag == "cu126"


def test_another_python_or_version_is_not_a_match():
    page = _index(
        "torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl",
        "torch-2.9.0%2Bcu128-cp312-cp312-win_amd64.whl",
    )
    tag = find_wheel_tag(
        "2.10.0", (12, 8), fetch=lambda _url: page, python_tag="cp312", platform_name="win32"
    )
    assert tag is None


def test_linux_arm_takes_the_aarch64_wheel():
    page = _index("torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_aarch64.whl")
    found = find_wheel_tag(
        "2.10.0",
        (12, 8),
        fetch=lambda _url: page,
        python_tag="cp312",
        platform_name="linux",
        machine="aarch64",
    )
    assert found == "cu128"
    missed = find_wheel_tag(
        "2.10.0",
        (12, 8),
        fetch=lambda _url: page,
        python_tag="cp312",
        platform_name="linux",
        machine="x86_64",
    )
    assert missed is None


# --- What it installs --------------------------------------------------------


@pytest.fixture
def pip_installer(monkeypatch):
    """Pretend pip is importable, so the command shape is the pip one."""
    from glider.plugins import installer

    real = installer.installer_command
    monkeypatch.setattr(
        installer,
        "installer_command",
        lambda package, requirements=(): real(
            package, requirements, pip_available=lambda: True, externally_managed=lambda: False
        ),
    )


def test_the_install_pins_this_torch_version_with_the_cuda_build(pip_installer):
    args = install_command("cu128", "2.10.0", "0.25.0+cpu")

    assert "torch==2.10.0+cu128" in args
    assert "torchvision==0.25.0+cu128" in args
    assert args[-2:] == ["--extra-index-url", "https://download.pytorch.org/whl/cu128"]


async def test_install_runs_the_command_for_the_tag_it_found(pip_installer, monkeypatch):
    monkeypatch.setattr(gpu_setup, "installed_torch", lambda: ("2.10.0", None))
    ran = []

    async def _runner(args, on_output=None):
        ran.append(args)
        return 0, "ok"

    result = await install_cuda_torch(
        (12, 8), runner=_runner, find_tag=lambda _version, _driver: "cu128"
    )

    assert result.ok
    assert "cu128" in result.message
    assert any(arg == "torch==2.10.0+cu128" for arg in ran[0])


async def test_no_matching_build_says_to_update_the_driver(monkeypatch):
    monkeypatch.setattr(gpu_setup, "installed_torch", lambda: ("2.10.0", None))

    async def _runner(args, on_output=None):  # pragma: no cover - must not run
        raise AssertionError("installed with no build to install")

    result = await install_cuda_torch(
        (11, 2), runner=_runner, find_tag=lambda _version, _driver: None
    )

    assert not result.ok
    assert "driver" in result.message


async def test_a_failed_install_reports_the_tool_and_output(pip_installer, monkeypatch):
    monkeypatch.setattr(gpu_setup, "installed_torch", lambda: ("2.10.0", None))

    async def _runner(args, on_output=None):
        return 1, "ERROR: no space left on device"

    result = await install_cuda_torch(
        (12, 8), runner=_runner, find_tag=lambda _version, _driver: "cu128"
    )

    assert not result.ok
    assert "pip exited with code 1" in result.message
    assert "no space" in result.output


@pytest.mark.parametrize(
    ("overrides", "says"),
    [
        ({"frozen": True}, "already includes"),
        ({"platform_name": "darwin"}, "Macs have no CUDA"),
        ({"torch_info": lambda: None}, "vision extra"),
        ({"torch_info": lambda: ("2.10.0+cu128", "12.8")}, "already installed"),
        ({"driver": lambda: None}, "NVIDIA driver"),
    ],
)
def test_every_refusal_says_why(overrides, says):
    probes = {
        "torch_info": lambda: ("2.10.0", None),
        "driver": lambda: (12, 8),
        "platform_name": "win32",
        "frozen": False,
        **overrides,
    }
    driver, reason = gpu_setup.gpu_support(**probes)
    assert driver is None
    assert says in reason
