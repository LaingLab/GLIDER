"""Animal Annotator core: label frames and build training datasets (#224).

Qt-free, so it can be unit-tested and driven from a script. The GUI lives in
:mod:`glider.gui.animal_annotator`.

- :mod:`.labels` -- YOLO pose label lines, read and written.
- :mod:`.project` -- a studio project on disk (``studio.json``).
- :mod:`.frames` -- pick and extract frames from videos.
- :mod:`.dataset` -- split by video and write ``data.yaml``.
"""
