"""Back-compat tests for the serialized model/CNN bundle formats.

GLIDER renamed the yolo2pose-era serialization identifiers — the
``BehaviorModel`` bundle's ``yolo2pose_version`` key and the CNN sequence
``yolo2pose-seq-cnn-*`` format string — to glider-namespaced names. New
bundles emit the glider names; ``load()`` must still open bundles that were
written with the legacy yolo2pose names.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")


def _tiny_behavior_model():
    from sklearn.ensemble import RandomForestClassifier

    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.model import BehaviorModel

    clf = RandomForestClassifier(n_estimators=2, random_state=0).fit(
        [[0.0, 0.0], [1.0, 1.0]], ["a", "b"]
    )
    return BehaviorModel(
        classifier=clf,
        feature_names=["f0", "f1"],
        spec=FeatureSpec(),
        window=5,
        stats=("mean",),
        fps=30.0,
        classes=["a", "b"],
    )


def test_behavior_model_roundtrip_emits_glider_version_key(tmp_path):
    import joblib

    from glider.analysis.behavior.model import BehaviorModel

    model = _tiny_behavior_model()
    out = tmp_path / "model.pkl"
    model.save(out)

    # New bundles record the training version under the glider-namespaced key.
    payload = joblib.load(out)
    assert "glider_version" in payload
    assert "yolo2pose_version" not in payload

    restored = BehaviorModel.load(out)
    assert restored.glider_version == model.glider_version


def test_behavior_model_loads_legacy_yolo2pose_version_key(tmp_path):
    import joblib

    from glider.analysis.behavior.model import BehaviorModel

    model = _tiny_behavior_model()
    out = tmp_path / "legacy.pkl"
    model.save(out)

    # Rewrite the bundle so it looks like a legacy yolo2pose-saved file.
    payload = joblib.load(out)
    payload["yolo2pose_version"] = payload.pop("glider_version")
    joblib.dump(payload, out)

    restored = BehaviorModel.load(out)  # must not raise
    assert restored.glider_version  # populated from the legacy key


def test_seq_bundle_formats_emit_glider_and_accept_legacy():
    from glider.analysis.behavior import sequence

    # New CNN bundles emit the glider-namespaced format string...
    assert sequence.SEQ_BUNDLE_FORMAT == "glider-seq-cnn-v2"
    # ...while load() still accepts the new format plus both legacy ones.
    assert sequence.SEQ_BUNDLE_FORMAT in sequence._SEQ_BUNDLE_FORMATS
    assert "yolo2pose-seq-cnn-v2" in sequence._SEQ_BUNDLE_FORMATS
    assert "yolo2pose-seq-cnn-v1" in sequence._SEQ_BUNDLE_FORMATS
