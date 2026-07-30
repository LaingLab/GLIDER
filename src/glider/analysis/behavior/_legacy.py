"""Load model bundles pickled when this package was still named ``yolo2pose``.

Pickle stores the *module path* of every class it serializes, so a bundle
saved before the rename references ``yolo2pose.*`` and dies with
``ModuleNotFoundError: No module named 'yolo2pose'`` on load — even though the
classes still exist under their current names. Retraining a cohort model just
to rename its imports is not a reasonable ask, so map the old paths onto their
current homes for the duration of the load.

Only modules that actually appear in real bundles are mapped. The list is
deliberately explicit rather than a blanket ``yolo2pose.* -> glider.*`` rewrite:
the two packages were not laid out the same way, so a guessed prefix would
resolve to the wrong class and unpickle silently-wrong state.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Old dotted path -> the module that holds those classes today.
_ALIASES: dict[str, str] = {
    "yolo2pose.train.embedding": "glider.analysis.behavior.embedding",
}

# Parent packages the aliases hang off. Pickle imports these on the way down,
# so they must exist even though nothing is read from them directly.
_STUB_PACKAGES: tuple[str, ...] = ("yolo2pose", "yolo2pose.train")


@contextmanager
def legacy_module_aliases():
    """Temporarily expose the pre-rename module paths to the unpickler.

    Scoped rather than installed at import time: the aliases exist only while a
    bundle is being read, so nothing else in the process can accidentally start
    importing from ``yolo2pose`` and take a dependency on the shim.
    """
    installed: list[str] = []
    try:
        for name in _STUB_PACKAGES:
            if name not in sys.modules:
                stub = types.ModuleType(name)
                stub.__path__ = []  # mark as a package so submodule import works
                sys.modules[name] = stub
                installed.append(name)
        for old, new in _ALIASES.items():
            if old not in sys.modules:
                sys.modules[old] = importlib.import_module(new)
                installed.append(old)
        yield
    finally:
        for name in installed:
            sys.modules.pop(name, None)


def is_legacy_bundle_error(exc: BaseException) -> bool:
    """Whether *exc* is the ModuleNotFoundError a pre-rename bundle raises."""
    return isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", "") == "yolo2pose"


# Reconstructing these triggers numba recompilation of the fitted nearest-
# neighbour index, which breaks across umap/pynndescent/numba versions.
_EMBEDDING_ONLY_PACKAGES = frozenset({"umap", "pynndescent", "numba"})


class _Unreconstructible:
    """Stands in for an object we deliberately declined to rebuild."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):  # swallow whatever pickle hands back
        pass


def _tolerant_unpickler(path, fileobj):
    """A joblib unpickler that refuses to rebuild the 3D-embedding internals."""
    from joblib.numpy_pickle import NumpyUnpickler

    class _Tolerant(NumpyUnpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in _EMBEDDING_ONLY_PACKAGES:
                return _Unreconstructible
            return super().find_class(module, name)

    # joblib grew a required ensure_native_byte_order argument; support both.
    try:
        return _Tolerant(path, fileobj, ensure_native_byte_order=True)
    except TypeError:
        return _Tolerant(path, fileobj)


def load_bundle(path) -> tuple[dict, bool]:
    """Read a joblib model bundle, escalating through compatibility fallbacks.

    Returns ``(payload, embedding_usable)``. Escalation order:

    1. Plain load — the normal case.
    2. Under the pre-rename module aliases, for bundles saved as ``yolo2pose``.
    3. Additionally declining to rebuild the optional 3D embedding, whose
       fitted umap/pynndescent index is version-fragile.

    Step 3 is worth doing because the embedding drives a visualization and
    nothing else: refusing to classify a whole cohort because a view cannot be
    reconstructed would be the wrong trade. The caller is told, so it can drop
    the half-built artifact rather than hand out a broken one.
    """
    import joblib

    try:
        return joblib.load(path), True
    except ModuleNotFoundError as e:
        if not is_legacy_bundle_error(e):
            raise
    except Exception:
        # Fall through to the embedding-tolerant attempt below.
        pass

    try:
        with legacy_module_aliases():
            payload = joblib.load(path)
        logger.info("loaded %s via the legacy yolo2pose module aliases", path)
        return payload, True
    except Exception as e:
        logger.info("%s needs the embedding-tolerant path: %s", path, e)

    with legacy_module_aliases(), open(path, "rb") as fileobj:
        payload = _tolerant_unpickler(path, fileobj).load()
    logger.warning(
        "%s loaded without its 3D embedding: the fitted umap/pynndescent index "
        "could not be rebuilt against the installed versions. Classification is "
        "unaffected; only the embedding view is unavailable.",
        path,
    )
    return payload, False
