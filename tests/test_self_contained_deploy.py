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
import os
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / 'README.md'
DEPLOY = REPO / 'deploy'
WORKSPACE = REPO / 'workspace'


@pytest.fixture
def staged_deploy(tmp_path):
    """
    A copy of deploy/ laid out the way a first real run would have it.

    The compose file references `.env` twice -- as `env_file`, and as a bind
    mount -- and `.env` is deliberately untracked because it holds a password.
    So `docker compose config` cannot run against a checkout that has never been
    deployed: compose aborts with "env file ... not found" before it parses
    anything. That is precisely the state CI is in, and precisely the state the
    first person to clone this repository is in.

    These tests were written against a developer machine where `.env` happened
    to exist, which is the same mistake as writing them against the source tree
    alone: they passed for a reason the reader does not have. Staging the env
    file here makes the test assert what it claims -- that the compose file
    parses -- instead of asserting that one machine is already deployed.

    `COMPOSE_PROJECT_NAME` is stripped from the staged file on purpose. Left in,
    the test below would pass off the value it happens to find in `.env` and
    never prove that the default lives in the compose file, which is the thing
    that keeps two checkouts from colliding.
    """
    import shutil

    target = tmp_path / 'deploy'
    shutil.copytree(DEPLOY, target, ignore=shutil.ignore_patterns('.env'))

    env_lines = [
        line for line
        in (DEPLOY / '.env.example').read_text(encoding='utf-8').splitlines()
        if not line.startswith('COMPOSE_PROJECT_NAME=')
    ]
    (target / '.env').write_text('\n'.join(env_lines) + '\n', encoding='utf-8')

    # The compose file bind-mounts ../workspace, which is a sibling of deploy/.
    (tmp_path / 'workspace').mkdir()
    return target


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


def test_the_compose_file_parses(staged_deploy):
    """A compose file that does not parse fails at the worst moment, which is
    on someone else's first run."""
    import shutil
    import subprocess

    if not shutil.which('docker'):
        pytest.skip('docker is not available')
    result = subprocess.run(
        ['docker', 'compose', '-f', str(staged_deploy / 'docker-compose.yml'), 'config', '--quiet'],
        capture_output=True, text=True, cwd=str(staged_deploy),
    )
    assert result.returncode == 0, result.stderr


# --- the documentation points at real paths -------------------------------


def test_the_readme_names_no_path_outside_this_repository(readme):
    """The defect this file exists for. A reader following the README must not
    be sent to a checkout they do not have."""
    offenders = [line.strip() for line in readme.splitlines() if 'my-bot' in line]
    assert not offenders, f'README points outside this repository: {offenders}'


def test_every_path_the_readme_uses_exists(readme):
    """Check the repo-relative paths the deploy section actually names.

    One exception, and it is the exception the instructions themselves create:
    `deploy/.env` is not shipped, because it holds a password. It is acceptable
    for the README to name it only while it also tells the reader the exact
    command that produces it.
    """
    for path in sorted(set(
        re.findall(r'`(deploy/[A-Za-z0-9_.-]+|workspace/[A-Za-z0-9_.-]*)`', readme)
    )):
        if (REPO / path).exists():
            continue

        assert path == 'deploy/.env', f'README names {path}, which does not exist'
        assert 'cp .env.example .env' in readme, (
            'the README names deploy/.env without telling the reader how to '
            'create it'
        )


def test_the_readme_leads_with_a_clone_and_run(readme):
    assert 'git clone' in readme
    assert 'docker compose up -d' in readme


def test_the_compose_project_is_named_explicitly():
    """Compose derives the project name from the directory, so every checkout
    calls itself `deploy` -- and `docker compose down` in one silently tears
    down another. Observed the hard way: tearing down a clone's stack took the
    running one with it."""
    compose = (DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8')
    assert re.search(r'^name:\s*\$\{COMPOSE_PROJECT_NAME:-', compose, re.M), (
        'the compose file must name its project explicitly'
    )


def test_two_checkouts_get_distinct_projects(staged_deploy):
    """The override is only useful if it actually changes the project name."""
    import shutil
    import subprocess

    if not shutil.which('docker'):
        pytest.skip('docker is not available')

    def project_name(env=None):
        result = subprocess.run(
            ['docker', 'compose', 'config'],
            cwd=str(staged_deploy), capture_output=True, text=True,
            env={**os.environ, **(env or {})},
        )
        assert result.returncode == 0, result.stderr
        match = re.search(r'^name:\s*(\S+)', result.stdout, re.M)
        return match.group(1) if match else None

    assert project_name() == 'algorithex'
    assert project_name({'COMPOSE_PROJECT_NAME': 'second'}) == 'second'


def test_the_published_port_is_the_port_the_app_listens_on():
    """APP_PORT is read by the app from the .env file mounted at /home/.env, so
    it is the container's listen port. Mapping "${APP_PORT}:9000" would publish
    host 9100 to container 9000 while the app listened on 9100 inside: healthy
    and unreachable. Found by deploying the clone on a non-default port."""
    compose = (DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8')
    code = '\n'.join(l for l in compose.splitlines() if not l.strip().startswith('#'))
    # Each published port looks like "${VAR:-NNNN}:" followed by the container
    # side, which must name the same variable.
    mappings = re.findall(r'"(\$\{[A-Z_]+:-\d+\}):([^"]+)"', code)
    assert mappings, 'no host:container port mappings found'
    for host_side, container_side in mappings:
        assert host_side == container_side, (
            f'"{host_side}:{container_side}" publishes a host port the app does '
            f'not listen on. The app reads this variable from the .env file, so '
            f'both sides of the mapping must be the same expression.'
        )


def test_the_readme_states_that_it_cannot_trade(readme):
    """The most expensive misunderstanding available is deploying this
    believing it places orders."""
    assert 'cannot place' in readme.lower() or 'cannot place,' in readme.lower()
