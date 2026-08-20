"""Fixtures shared by the whole ``glider-harp`` suite.

There is exactly one, and it exists because this package now reads a directory
that belongs to the person running the tests.
"""

import pytest

from glider_harp import derivation


@pytest.fixture(autouse=True)
def user_profiles(tmp_path_factory, monkeypatch):
    """Point the user profile directory at an empty directory of our own.

    Autouse and unconditional. ``~/.glider/harp_profiles`` is a real directory
    on a developer's machine, and a user profile *overrides a shipped one of
    the same name* -- so a ``licketysplit.json`` sitting in it would silently
    replace the shipped profile that a dozen assertions here are about, and the
    failures would appear on one machine and nowhere else. Isolating it is what
    keeps this suite a statement about the code.

    Returned, so a test that wants a user profile writes it here.
    """
    directory = tmp_path_factory.mktemp("harp-user-profiles")
    monkeypatch.setattr(derivation, "user_profile_dir", lambda: directory)
    return directory
