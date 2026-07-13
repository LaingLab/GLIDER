"""
Runner subpackage.

The old tabbed RunnerShell/RunnerPanel UI was removed when the dashboard
became the sole touch-facing mode. This package now provides the smaller
building blocks that the dashboard panels reuse rather than a standalone
runner UI:

- ``readiness.py`` — readiness computation for starting an experiment
- ``run_timer.py`` — elapsed-time formatting
- ``run_banner.py`` — the run status banner
- ``device_controls.py`` — manual device controls
- ``runner_setup_page.py`` — the experiment setup page

See ``glider.gui.dashboard`` for the quadrant dashboard that hosts these
pieces.
"""

__all__: list[str] = []
