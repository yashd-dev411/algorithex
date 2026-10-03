"""
Tests that the repository is deployable on its own.

This exists because of a specific, shipped defect: the deploy instructions told
readers to `cd my-bot/docker`, and `my-bot` is a separate repository. Anyone who
cloned this one had no compose file, no env template and no way to run the thing
they had just cloned. Nothing in the test suite caught it, because no test
looked at the README or asked whether the documented path exists.

So: every path the documentation names must be inside this repository.
"""

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / 'README.md'
DEPLOY = REPO / 'deploy'
WORKSPACE = REPO / 'workspace'


@pytest.fixture(scope='module')
def readme():
    return README.read_text(encoding='utf-8')


# --- the pieces a clone needs -------------------------------------------


def test_the_repo_ships_a_compose_file():
    """The single most important file, and the one that was missing."""
    compose = DEPLOY / 'docker-compose.yml'
    assert compose.is_file(), 'a fresh clone has nothing to run'
    assert 'services:' in compose.read_text(encoding='utf-8')


def test_the_repo_ships_an_env_template():
    assert (DEPLOY / '.env.example').is_file()


def test_the_env_template_documents_every_variable_the_compose_uses():
    compose = (DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8')
    template = (DEPLOY / '.env.example').read_text(encoding='utf-8')
    # Comments are stripped: the compose file explains `${VAR}` in prose, and a
    # literal `VAR` is not a variable anyone can set.
    code = '\n'.join(
        line for line in compose.splitlines() if not line.strip().startswith('#')
    )
    used = set(re.findall(r'\$\{([A-Z_][A-Z0-9_]*)', code))
    missing = sorted(v for v in used if f'{v}=' not in template)
    assert not missing, f'compose uses variables absent from .env.example: {missing}'


def test_the_workspace_skeleton_ships():
    """The app refuses to start without these two directories in its working
    directory (helpers.is_algorithex_project), so a clone that lacks them
    crash-loops on first run."""
    assert (WORKSPACE / 'strategies').is_dir()
    assert (WORKSPACE / 'storage').is_dir()


def test_the_workspace_ignores_generated_data():
    """Otherwise the first run commits candles, logs and results."""
    ignore = (WORKSPACE / '.gitignore').read_text(encoding='utf-8')
    for pattern in ('storage/', '.env', '__pycache__'):
        assert pattern in ignore, f'workspace/.gitignore does not cover {pattern}'


def test_the_real_env_file_is_not_tracked():
    import subprocess

    tracked = subprocess.run(
        ['git', 'ls-files'], cwd=REPO, capture_output=True, text=True,
    ).stdout.split()
    offenders = [p for p in tracked if p.endswith('.env') and not p.endswith('.env.example')]
    assert not offenders, f'committed env files: {offenders}'


# --- the compose stack is wired correctly --------------------------------


def test_the_env_file_is_mounted_where_the_app_reads_it():
    """The app reads a .env *file* in its working directory, not the process
    environment. With only `env_file:` it exits with '.env is missing or empty'
    and, under restart: unless-stopped, crash-loops. This was observed, not
    theorised."""
    compose = (DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8')
    assert re.search(r'\./\.env:/home/\.env', compose), (
        'the app needs deploy/.env mounted at /home/.env'
    )


def test_the_workspace_is_mounted_as_the_working_directory():
    compose = (DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8')
    assert '../workspace:/home' in compose


def test_startup_waits_for_the_databases_to_be_ready():
    """Started is not ready. Without this the app boots first, fails to connect
    and exits, which reads as a crash rather than a slow start."""
    compose = (DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8')
    assert compose.count('condition: service_healthy') >= 2


def test_every_service_declares_a_healthcheck():
    compose = (DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8')
    assert compose.count('healthcheck:') == 3, 'app, postgres and redis each need one'


def test_the_compose_file_parses():
    """A compose file that does not parse fails at the worst moment, which is
    on someone else's first run."""
    import shutil
    import subprocess

    if not shutil.which('docker'):
        pytest.skip('docker is not available')
    result = subprocess.run(
        ['docker', 'compose', '-f', str(DEPLOY / 'docker-compose.yml'), 'config', '--quiet'],
        capture_output=True, text=True, cwd=str(DEPLOY),
    )
    assert result.returncode == 0, result.stderr


# --- the documentation points at real paths -------------------------------


def test_the_readme_names_no_path_outside_this_repository(readme):
    """The defect this file exists for. A reader following the README must not
    be sent to a checkout they do not have."""
    offenders = [line.strip() for line in readme.splitlines() if 'my-bot' in line]
    assert not offenders, f'README points outside this repository: {offenders}'


def test_every_path_the_readme_uses_exists(readme):
    """Check the repo-relative paths the deploy section actually names."""
    for path in re.findall(r'`(deploy/[A-Za-z0-9_.-]+|workspace/[A-Za-z0-9_.-]*)`', readme):
        assert (REPO / path).exists(), f'README names {path}, which does not exist'


def test_the_readme_leads_with_a_clone_and_run(readme):
    assert 'git clone' in readme
    assert 'docker compose up -d' in readme


def test_the_readme_states_that_it_cannot_trade(readme):
    """The most expensive misunderstanding available is deploying this
    believing it places orders."""
    assert 'cannot place' in readme.lower() or 'cannot place,' in readme.lower()
