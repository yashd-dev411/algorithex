"""Native indicator kernel.

A compiled Rust extension providing the numeric core for the indicator library.
It is built from source in ``native-kernel/`` and the per-platform binaries are
vendored under ``bin/``, so installing this package needs no external native
dependency.

Callers import symbols directly, e.g. ``from algorithex._native import sma``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

_BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
_MODULE_NAME = "algorithex_kernel"

# Filename stem per interpreter/platform. The binaries are built for the stable
# ABI, so one file serves every CPython 3.10+ on that platform.
_SUFFIXES = ("so", "pyd", "dylib")

# Every platform's binary ships in the same directory, so try this platform's
# suffix first instead of relying on a foreign binary failing to load.
_SUFFIXES_BY_PLATFORM = {"linux": "so", "win32": "pyd", "darwin": "dylib"}
_NATIVE_SUFFIX = _SUFFIXES_BY_PLATFORM.get(sys.platform)


def _candidates() -> tuple[str, ...]:
    ordered = (_NATIVE_SUFFIX,) if _NATIVE_SUFFIX else ()
    ordered += tuple(s for s in _SUFFIXES if s not in ordered)
    return tuple(f"{_MODULE_NAME}.{suffix}" for suffix in ordered)


def _load() -> ModuleType:
    # CPython resolves the init symbol from the module name, so the extension
    # must be loaded under exactly this name regardless of its file suffix.
    errors: list[str] = []
    for name in _candidates():
        path = os.path.join(_BIN_DIR, name)
        if not os.path.exists(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
            if spec is None or spec.loader is None:
                errors.append(f"{name}: no loader")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError(
        "No usable native indicator kernel for "
        f"{sys.platform}/Python {sys.version_info.major}.{sys.version_info.minor}. "
        f"Looked in {_BIN_DIR}. Tried: {errors}"
    )


_impl = _load()

__doc__ = _impl.__doc__
__all__ = [n for n in dir(_impl) if not n.startswith("_")]

globals().update({n: getattr(_impl, n) for n in __all__})
del _impl