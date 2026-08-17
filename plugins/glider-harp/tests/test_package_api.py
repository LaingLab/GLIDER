"""What ``glider_harp`` exports.

The package is what callers import, not the modules inside it. Pinning the
surface here means a file rename is a decision, not an outage in Task 11.
"""

import glider_harp


def test_the_reader_side_is_importable_from_the_package():
    """A cache is not a reader; ``from glider_harp.reader import RegisterCache`` reads wrong."""
    from glider_harp import HarpReader, RegisterCache

    assert glider_harp.RegisterCache is RegisterCache
    assert glider_harp.HarpReader is HarpReader


def test_the_frame_layer_is_importable_from_the_package():
    from glider_harp import ChecksumError, FrameError, FrameSplitter, TruncatedFrameError, decode

    assert issubclass(ChecksumError, FrameError)
    assert issubclass(TruncatedFrameError, FrameError)
    assert callable(decode)
    assert FrameSplitter().feed(b"") == []


def test_all_lists_exactly_what_is_exported():
    """__all__ that drifts from the module is worse than none: it lies to star imports."""
    assert sorted(glider_harp.__all__) == glider_harp.__all__
    for name in glider_harp.__all__:
        assert hasattr(glider_harp, name), name
