"""Batch pose tracking tool (Tools → Batch Pose Tracking…).

Runs a YOLO-pose model over directories of videos and writes a DeepLabCut
CSV beside each one, driving the Qt-free core in
:mod:`glider.vision.pose.batch` from a QThread.

Nothing heavy is imported here: :class:`~.window.PoseBatchWindow` pulls in
ultralytics/torch, so ``MainWindow._open_pose_batch`` imports it lazily inside
the handler. Only :mod:`.availability` is safe to import while building menus.
"""
