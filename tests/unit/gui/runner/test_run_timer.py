from glider.gui.runner.run_timer import format_elapsed


def test_zero():
    assert format_elapsed(0.0) == "00:00.00"


def test_sub_minute_truncates_toward_zero():
    assert format_elapsed(9.999) == "00:09.99"


def test_minutes_and_centiseconds():
    assert format_elapsed(83.45) == "01:23.45"


def test_switches_to_hours_past_one_hour():
    assert format_elapsed(3661.5) == "01:01:01.50"


def test_negative_clamped_to_zero():
    assert format_elapsed(-1.0) == "00:00.00"
