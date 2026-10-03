import os
import re
import pytest
import numpy as np

from algorithex import _native


@pytest.mark.parametrize('platform, expected_first', [
    ('linux', 'algorithex_kernel.so'),
    ('win32', 'algorithex_kernel.pyd'),
    ('darwin', 'algorithex_kernel.dylib'),
])
def test_native_binary_for_platform_is_tried_first(platform, expected_first, monkeypatch):
    # Every platform's binary ships side by side in the same directory, so a
    # foreign binary must not be attempted before the right one.
    monkeypatch.setattr(_native, '_NATIVE_SUFFIX', _native._SUFFIXES_BY_PLATFORM.get(platform))

    candidates = _native._candidates()

    assert candidates[0] == expected_first
    assert sorted(candidates) == sorted(_native._candidates())


def test_every_platform_suffix_is_still_considered():
    # An unknown platform must still get a chance at every vendored binary
    # rather than failing outright.
    assert len(_native._candidates()) == len(_native._SUFFIXES)


def test_loaded_kernel_exposes_the_numeric_core():
    # Importing this module already loaded the binary; assert it is usable.
    candles = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert _native.sma(candles, 5)[-1] == pytest.approx(3.0)


def test_vendored_binary_exists_for_this_platform():
    expected = os.path.join(_native._BIN_DIR, _native._candidates()[0])

    assert os.path.exists(expected)


def test_vendored_binary_is_not_locked_to_one_python_version():
    # A build against a versioned interpreter hard-links that interpreter's
    # DLL (python3XY.dll) and then fails to load on every other CPython. Stable
    # ABI builds leave the Python symbols undefined instead.
    stale = re.compile(rb'python3\d', re.IGNORECASE)

    for name in sorted(os.listdir(_native._BIN_DIR)):
        with open(os.path.join(_native._BIN_DIR, name), 'rb') as handle:
            blob = handle.read()

        assert not stale.search(blob), f'{name} is linked to a specific Python version'