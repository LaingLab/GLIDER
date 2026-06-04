"""
GLIDER Dialog Components.

Contains dialog windows for:
- Camera Settings
- Subject Editor
- Experiment Settings
- CSV Data Analysis
"""

from glider.gui.dialogs.analysis_dialog import AnalysisDialog
from glider.gui.dialogs.camera_settings_dialog import CameraSettingsDialog
from glider.gui.dialogs.experiment_dialog import ExperimentDialog
from glider.gui.dialogs.subject_dialog import SubjectDialog

__all__ = [
    "AnalysisDialog",
    "ExperimentDialog",
    "CameraSettingsDialog",
    "SubjectDialog",
]
