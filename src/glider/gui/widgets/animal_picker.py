"""Pick which animal from the Experiment > Animals list a run is of.

The choice is the session's ``active_subject_id``, which the data recorder and
the tracking logger already write into their CSV headers -- so picking here is
all a run needs to carry the animal. Shared by the run header (desktop and Pi)
and the multi-camera window, which is why it reads the session through a
callable: a New or Open replaces the session object under it.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QComboBox

_NONE = "No animal"


def subject_label(subject) -> str:
    """How an animal is named in a list: its ID, then its name and group."""
    parts = [subject.subject_id or subject.name or "Unnamed"]
    if subject.name and subject.subject_id:
        parts.append(subject.name)
    if subject.group:
        parts.append(f"({subject.group})")
    return " ".join(parts)


class AnimalPicker(QComboBox):
    """A combo of the session's animals, bound to its active subject."""

    def __init__(self, session: Callable[[], object | None], parent=None) -> None:
        super().__init__(parent)
        self._session = session
        self.setToolTip("The animal this run is of. Add animals in Experiment > Animals.")
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.currentIndexChanged.connect(self._on_chosen)
        self.refresh()

    def _metadata(self):
        session = self._session()
        return getattr(session, "metadata", None) if session is not None else None

    def refresh(self) -> None:
        """Rebuild from the session; the Animals list may have changed."""
        md = self._metadata()
        subjects = list(getattr(md, "subjects", None) or [])
        self.blockSignals(True)
        self.clear()
        self.addItem(_NONE, None)
        for subject in subjects:
            self.addItem(subject_label(subject), subject.id)
        active = getattr(md, "active_subject_id", None)
        self.setCurrentIndex(max(0, self.findData(active)) if active else 0)
        self.blockSignals(False)

    def showPopup(self) -> None:  # noqa: N802 - Qt override
        # Animals added since the last refresh should be in the list.
        self.refresh()
        super().showPopup()

    def active_subject(self):
        md = self._metadata()
        return md.get_active_subject() if md is not None else None

    def _on_chosen(self, _index: int) -> None:
        md = self._metadata()
        if md is None:
            return
        chosen = self.currentData()
        if chosen == md.active_subject_id:
            return
        md.active_subject_id = chosen
        session = self._session()
        if hasattr(session, "mark_dirty"):
            session.mark_dirty()
