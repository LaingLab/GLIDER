"""Single source of truth for the GLIDER version string.

This module exists so that the version is defined in exactly one place and is
readable both at runtime (``glider.__version__``) and at build time via
``pyproject.toml``'s ``[tool.setuptools.dynamic]`` section.

Bump the value here when cutting a release — CI reads it via ``setuptools``
dynamic resolution and the in-app updater compares it against GitHub Releases.
"""

__version__ = "0.3.0-dev"
