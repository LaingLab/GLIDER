"""The `--version` flag prints the version and exits 0.

The PyInstaller smoke build in CI runs `GLIDER.exe --version` and asserts a
clean exit, so the entry point must support the flag. ``parse_args()`` reads
from ``sys.argv``, so these tests patch it rather than passing an argv list.
"""

import sys

import pytest

from glider.__main__ import parse_args
from glider._version import __version__


def test_version_flag_prints_version_and_exits_zero(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["glider", "--version"])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_normal_args_still_parse(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["glider", "--runner"])
    args = parse_args()
    assert args.runner is True
    assert args.builder is False
