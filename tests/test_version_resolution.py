"""
The version number must resolve from a bare source tree.

`algorithex.cli` asks for it at import time, via the `@click.version_option`
decorator. That makes missing distribution metadata a whole-package failure
rather than a cosmetic one, and in CI it took 70 test modules down with it --
a job that installs `requirements.txt` and never installs the project itself.
The Docker image runs `pip install -e .` and so never noticed.

The failure mode reproduced here is `importlib.metadata.version('algorithex')`
raising `PackageNotFoundError`, which is precisely what an uninstalled checkout
raises. The child-process boundary matters: the parent has already imported
the package, so an in-process test would pass with the bug still in place.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Patched in *before* `algorithex.cli` is imported, because the lookup happens
# during that import and not when the CLI actually runs.
HIDE_DISTRIBUTION_METADATA = '''
import importlib.metadata as _md

def _version(name):
    if name == "algorithex":
        raise _md.PackageNotFoundError(name)
    return _md.version.__wrapped__(name) if hasattr(_md.version, "__wrapped__") else _md.Distribution.from_name(name).version

_md.version = _version
'''

RESOLVE_VERSION = (
    HIDE_DISTRIBUTION_METADATA
    + 'from algorithex.cli import _get_dist_version\n'
    + 'print("resolved:", _get_dist_version())\n'
)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_the_fixture_really_hides_the_distribution_metadata() -> None:
    """Guard the guard: if the patch stopped biting, the tests below pass for
    the wrong reason."""
    result = _run(
        'import importlib.metadata as md\n'
        + HIDE_DISTRIBUTION_METADATA
        + 'try:\n'
        '    md.version("algorithex")\n'
        'except md.PackageNotFoundError:\n'
        '    print("hidden")\n'
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == 'hidden', result.stdout


def test_the_fixture_still_finds_other_distributions() -> None:
    """The patch must be narrow. It is standing in for one missing package, not
    for an environment where `importlib.metadata` is simply broken."""
    result = _run(
        'import importlib.metadata as md\n'
        + HIDE_DISTRIBUTION_METADATA
        + 'print("pytest:", md.version("pytest"))\n'
    )
    assert result.returncode == 0, result.stderr
    assert 'pytest: ' in result.stdout, result.stdout


def test_importing_cli_survives_missing_distribution_metadata() -> None:
    result = _run(RESOLVE_VERSION)

    assert result.returncode == 0, (
        'importing algorithex.cli without installed distribution metadata '
        f'failed:\n{result.stderr}'
    )
    assert result.stdout.strip().splitlines()[-1].startswith('resolved: '), result.stdout


def test_the_fallback_reports_the_version_from_the_source_tree() -> None:
    from algorithex.version import __version__

    result = _run(RESOLVE_VERSION)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == f'resolved: {__version__}'


def test_the_fallback_reads_the_same_number_setup_py_publishes() -> None:
    """`setup.py` parses `algorithex/version.py` as text for the same reason --
    importing it during a build would recurse. If the two parsers drift, the
    package would report one version and PyPI would serve another."""
    from algorithex.version import __version__

    source = (REPO_ROOT / 'algorithex' / 'version.py').read_text(encoding='utf-8')
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", source, re.MULTILINE
    )

    assert match is not None, 'version.py no longer matches setup.py\'s regex'
    assert match.group(1) == __version__


def test_an_installed_distribution_still_wins_over_the_source_tree() -> None:
    """The fallback is a fallback, not a replacement: when metadata exists it is
    the authority, because an editable install can be behind the working tree."""
    # `import algorithex.cli as cli` binds the click Group, not the module:
    # `algorithex/__init__.py` does `from algorithex.cli import cli`, so the
    # attribute shadows the submodule on the package.
    from importlib import import_module

    cli = import_module('algorithex.cli')

    original = cli.get_version
    try:
        cli.get_version = lambda name: '9.9.9-from-metadata'
        assert cli._get_dist_version() == '9.9.9-from-metadata'
    finally:
        cli.get_version = original

    # And when there is real metadata, that real metadata is what comes back.
    # Only assertable where it exists: an uninstalled checkout is precisely the
    # state the fallback exists for.
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version('algorithex')
    except PackageNotFoundError:
        return

    assert cli._get_dist_version() == installed