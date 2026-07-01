"""GLIDER behavior-analysis GUI (Qt) surface.

Houses the PyQt6 windows for the ported behavior pipeline — the
active-learning clip annotator (:mod:`glider.gui.behavior.annotator`) and,
in a later phase, the analysis tool window. The Qt-free data and compute
layers live under :mod:`glider.analysis.behavior` and
:mod:`glider.vision.pose`; this package only holds the widgets.
"""
