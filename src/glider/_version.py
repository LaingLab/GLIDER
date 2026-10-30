"""Single source of truth for the GLIDER version string.

This module exists so that the version is defined in exactly one place and is
readable both at runtime (``glider.__version__``) and at build time via
``pyproject.toml``'s ``[tool.setuptools.dynamic]`` section.

Bump the value here when cutting a release — CI reads it via ``setuptools``
dynamic resolution. The in-app updater compares it against GitHub Releases only
for a prebuilt app; a git checkout is compared with ``main`` instead.
"""

__version__ = "1.0.0"
