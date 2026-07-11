"""Interactive first-run walkthrough (spotlight coach-marks).

Public entry point: :func:`start_tour`. The rest of the app should only need
``tour_targets()`` on the main window plus this one call.
"""

from glider.gui.onboarding.tour import (
    Tour,
    TourStep,
    golden_path_steps,
    start_tour,
    tour_complete,
)

__all__ = ["Tour", "TourStep", "golden_path_steps", "start_tour", "tour_complete"]
