"""
Runner subpackage.

Building blocks for the operator views: the runner-mode RunnerShell (Pi
touchscreen) and the desktop dashboard reuse these:

- ``readiness.py`` — readiness computation for starting an experiment
- ``run_timer.py`` — elapsed-time formatting
- ``run_banner.py`` — the run status banner
- ``device_controls.py`` — manual device controls
- ``runner_setup_page.py`` — the RunnerShell's Setup page
- ``runner_shell.py`` — the runner-mode tabbed shell

See ``glider.gui.dashboard`` for the fixed 2x2 desktop dashboard.
"""

__all__: list[str] = []
