"""
Runner subpackage.

Building blocks for the operator views: the runner-mode Pi pages (Pi
touchscreen) and the desktop dashboard reuse these:

- ``readiness.py`` — readiness computation for starting an experiment
- ``run_timer.py`` — elapsed-time formatting
- ``device_controls.py`` — manual device controls
- ``device_grid.py`` — the Pi Run page's device cards
- ``notice_bar.py`` — the Pi's notification bar
- ``pi_dashboard.py`` — the Pi's Dashboard tab (graph over panel tabs)
- ``run_page.py`` — the Pi's Run tab

See ``glider.gui.dashboard`` for the desktop camera-cockpit dashboard.
"""

__all__: list[str] = []
