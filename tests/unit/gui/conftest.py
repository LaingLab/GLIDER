"""Shared fixtures for GUI panel unit tests.

The ``tiny_behavior_model`` fixture now lives in ``tests/unit/conftest.py`` so
both the GUI panel tests and the live/offline behavior-classify parity test can
share a single, byte-for-byte-consistent model definition. This module is kept
so pytest still collects the ``tests/unit/gui`` package cleanly.
"""

from __future__ import annotations
