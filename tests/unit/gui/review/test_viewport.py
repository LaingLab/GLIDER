"""Visible-window arithmetic for the review timeline. No display needed."""

from __future__ import annotations

import pytest

from glider.gui.review.viewport import (
    Viewport,
    format_seconds,
    format_timecode,
    snap,
    tick_spacing,
)


def _vp():
    return Viewport(0.0, 1000.0, min_span=10.0)


def test_a_new_viewport_shows_everything():
    vp = _vp()
    assert (vp.start, vp.end) == (0.0, 1000.0)


def test_zoom_keeps_the_point_under_the_cursor():
    vp = _vp()
    before = vp.x_of(250.0, 500)
    vp.zoom(0.5, about=250.0)
    assert vp.span == pytest.approx(500.0)
    assert vp.x_of(250.0, 500) == pytest.approx(before)


def test_zoom_is_clamped_to_the_session():
    vp = _vp()
    vp.zoom(0.5, about=990.0)
    assert vp.end <= 1000.0
    vp.zoom(10.0, about=500.0)
    assert (vp.start, vp.end) == (0.0, 1000.0)


def test_zoom_stops_at_the_minimum_span():
    vp = _vp()
    vp.zoom(0.0001, about=500.0)
    assert vp.span == pytest.approx(10.0)


def test_pan_stops_at_the_ends():
    vp = _vp()
    vp.show(0.0, 100.0)
    vp.pan(-50.0)
    assert vp.start == 0.0
    vp.pan(5000.0)
    assert vp.end == 1000.0 and vp.span == pytest.approx(100.0)


def test_follow_pages_rather_than_scrolls():
    vp = _vp()
    vp.show(0.0, 100.0)
    vp.follow(50.0)
    assert vp.start == 0.0
    vp.follow(120.0)
    assert (vp.start, vp.end) == (120.0, 220.0)


def test_x_and_t_round_trip():
    vp = _vp()
    vp.show(200.0, 400.0)
    assert vp.t_at(vp.x_of(321.0, 800), 800) == pytest.approx(321.0)


def test_snap_takes_the_nearest_within_tolerance():
    edges = [10.0, 20.0, 30.0]
    assert snap(21.0, edges, 2.0) == 20.0
    assert snap(25.0, edges, 2.0) == 25.0
    assert snap(5.0, [], 2.0) == 5.0


def test_timecode_has_frames_and_a_sign():
    assert format_timecode(134.23, 30.0) == "00:02:14:06"
    assert format_timecode(-5.5, 30.0) == "-00:00:05:15"
    assert format_timecode(59.999, 30.0) == "00:00:59:29"


def test_seconds_read_out():
    assert format_seconds(125.5) == "2:05.50"
    assert format_seconds(-3.25) == "-0:03.25"


def test_tick_spacing_follows_the_span():
    assert tick_spacing(60) == (10, 1)
    assert tick_spacing(300) == (30, 5)
    assert tick_spacing(1800) == (60, 10)
    assert tick_spacing(7200) == (600, 60)
