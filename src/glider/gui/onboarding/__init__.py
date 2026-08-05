"""Interactive first-run walkthrough (spotlight coach-marks).

Public entry points: :func:`start_tour` for the main window's golden path, and
:func:`offer_tour_once` for the tool windows, which are opened on demand and so
each carry their own walkthrough and their own "seen it" flag.
"""

from glider.gui.onboarding.tour import (
    BEHAVIOR_TOUR_COMPLETE_KEY,
    POSE_BATCH_TOUR_COMPLETE_KEY,
    SESSION_REVIEW_TOUR_COMPLETE_KEY,
    TOUR_COMPLETE_KEY,
    Tour,
    TourStep,
    behavior_steps,
    golden_path_steps,
    offer_tour_once,
    pose_batch_steps,
    session_review_steps,
    start_tour,
    tour_complete,
)

__all__ = [
    "BEHAVIOR_TOUR_COMPLETE_KEY",
    "POSE_BATCH_TOUR_COMPLETE_KEY",
    "SESSION_REVIEW_TOUR_COMPLETE_KEY",
    "TOUR_COMPLETE_KEY",
    "Tour",
    "TourStep",
    "behavior_steps",
    "golden_path_steps",
    "offer_tour_once",
    "pose_batch_steps",
    "session_review_steps",
    "start_tour",
    "tour_complete",
]
