from glider.analysis.behavior.vocabulary import Behavior, Vocabulary


def test_behavior_carries_tags_and_round_trips():
    v = Vocabulary(
        [
            Behavior(name="rest", hotkey="1", tags={"stationary"}),
            Behavior(name="locomote", hotkey="2", tags={"locomotory"}),
            Behavior(name="rear", hotkey="3"),  # untagged -> empty
        ]
    )
    assert v.behavior_for_name("rest").tags == frozenset({"stationary"})
    assert v.behavior_for_name("rear").tags == frozenset()
    round_tripped = Vocabulary.from_dict(v.to_dict())
    assert round_tripped.behavior_for_name("locomote").tags == frozenset({"locomotory"})


def test_tag_map_helper():
    v = Vocabulary([Behavior(name="rest", hotkey="1", tags={"stationary"})])
    assert v.tag_map() == {"rest": frozenset({"stationary"})}
