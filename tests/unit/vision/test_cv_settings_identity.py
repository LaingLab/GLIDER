from glider.vision.cv_processor import CVSettings


def test_identity_defaults_off():
    s = CVSettings()
    assert s.identity_mode == "off" and s.animals == []


def test_identity_round_trip():
    s = CVSettings(
        identity_mode="appearance",
        animals=[{"name": "Dam", "class_name": "dam"}],
        identity_relabel_s=0.8,
    )
    back = CVSettings.from_dict(s.to_dict())
    assert back.identity_mode == "appearance"
    assert back.animals == [{"name": "Dam", "class_name": "dam"}]
    assert back.identity_relabel_s == 0.8


def test_old_dict_loads_off():
    assert CVSettings.from_dict({"backend": "YOLO_V8"}).identity_mode == "off"


def test_unknown_mode_falls_back_to_off():
    assert CVSettings.from_dict({"identity_mode": "magic"}).identity_mode == "off"


def test_copy_does_not_share_animals():
    s = CVSettings(animals=[{"name": "A", "class_name": None}])
    c = s.copy()
    c.animals[0]["name"] = "B"
    assert s.animals[0]["name"] == "A"
