"""
Tests for the deployment surface.

These are not testing the application's behaviour. They are testing the things
that make this repository safe to push to a public remote: that a CI run cannot
publish anything anywhere, that the healthcheck reports failures quietly, and
that no credential is hardcoded in a file that gets committed.

The failure these guard against is specific and has already happened once in
this repository's history: an inherited workflow published to PyPI and Docker
Hub on any `v*` tag, under a namespace this project does not own, using secrets
that do not exist here. Nothing about that was visible from the code.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / '.github' / 'workflows'
DOCKERFILE = REPO / 'Dockerfile'
HEALTHCHECK = REPO / 'docker_healthcheck.py'


def _without_yaml_comments(body: str) -> str:
    """Drop whole-line YAML comments.

    These workflows explain their own hazards in comments, which is the point --
    but it means a substring search over the raw file matches the explanation
    rather than the configuration. Search the configuration.
    """
    return '\n'.join(
        line for line in body.splitlines() if not line.strip().startswith('#')
    )


@pytest.fixture(scope='module')
def workflows():
    return {p.name: p.read_text(encoding='utf-8') for p in sorted(WORKFLOWS.glob('*.yml'))}


# --- nothing can publish somewhere it does not own ----------------------


def test_no_workflow_publishes_to_a_package_index(workflows):
    """PyPI is not a deployment target for this project. If a future edit adds
    it back, that edit is publishing under a name it does not control."""
    for name, body in workflows.items():
        code = _without_yaml_comments(body)
        for banned in ('pypa/gh-action-pypi-publish', 'twine upload', 'PYPI_API_TOKEN'):
            assert banned not in code, f'{name} publishes to PyPI via {banned}'


def test_no_workflow_pushes_to_a_hardcoded_external_registry(workflows):
    """The registry must come from the repository, not a literal. A hardcoded
    name is how the previous workflow ended up aiming at someone else's
    Docker Hub namespace."""
    for name, body in workflows.items():
        code = _without_yaml_comments(body)
        assert 'docker.io' not in code, f'{name} pushes to a literal docker.io path'
        assert 'docker/login-action' not in code or 'registry: ${{ env.REGISTRY }}' in code, (
            f'{name} logs in to a registry that is not derived from the repository'
        )


def test_the_image_workflow_uses_the_repository_namespace(workflows):
    body = workflows['docker-publish.yml']
    assert '${{ github.repository }}' in body
    assert 'ghcr.io' in body


def test_the_image_workflow_cannot_ship_the_test_stage(workflows):
    """The Dockerfile picks its final stage with
    `FROM algorithex_with_test_${TEST_BUILD}`, and stage 1's ENTRYPOINT is the
    test suite. Publishing with TEST_BUILD=1 ships a container whose entrypoint
    is pytest.

    Checked on the build arguments rather than on the word appearing anywhere,
    because the workflow deliberately explains this hazard in a comment.
    """
    body = _without_yaml_comments(workflows['docker-publish.yml'])
    assert not re.search(r'TEST_BUILD\s*[:=]\s*\S', body), (
        'the publish workflow passes a TEST_BUILD build-arg'
    )
    assert 'build-args' not in body, 'the publish workflow overrides build args'
    assert 'ARG TEST_BUILD=0' in DOCKERFILE.read_text(encoding='utf-8')


def test_every_workflow_pins_a_minimum_permission(workflows):
    for name, body in workflows.items():
        assert re.search(r'^permissions:', body, re.M), f'{name} declares no permissions block'


def test_no_workflow_uses_a_deprecated_action_major(workflows):
    """checkout@v2 and cache@v2 are disabled by GitHub: a workflow pinning them
    silently does not run, which reads as coverage that is not there."""
    for name, body in workflows.items():
        for action, ref in re.findall(
            r'uses:\s*([\w-]+/[\w.-]+)@(\S+)', _without_yaml_comments(body)
        ):
            major = ref.lstrip('v').split('.')[0]
            assert int(major) >= 3, f'{name} uses {action}@{ref}, which is disabled'


# --- the test workflow claims only what has been run ---------------------


def test_the_test_matrix_matches_the_verified_interpreters(workflows):
    """3.11 is the Dockerfile base image, 3.12 is what the suite is run against
    interactively. Anything else in the matrix is untested by definition."""
    body = workflows['python-package.yml']
    versions = re.findall(r"python-version:\s*\[([^\]]+)\]", body)
    assert versions, 'no matrix found'
    declared = set(re.findall(r"'([\d.]+)'", versions[0]))
    assert declared == {'3.11', '3.12'}, f'unverified versions in the matrix: {declared}'


def test_the_test_workflow_preserves_pytest_exit_status(workflows):
    """`pytest | tail` returns tail's status. Without pipefail a red suite is
    reported as green, which is the worst possible bug in a CI gate."""
    body = workflows['python-package.yml']
    assert 'set -o pipefail' in body


def test_the_test_workflow_provides_its_databases(workflows):
    body = workflows['python-package.yml']
    assert 'postgres:14-alpine' in body
    assert 'redis:6-alpine' in body


# --- the image can be deployed and observed ------------------------------


def test_the_image_declares_a_healthcheck():
    body = DOCKERFILE.read_text(encoding='utf-8')
    assert 'HEALTHCHECK' in body
    assert 'docker_healthcheck.py' in body


def _code_without_prose(path: Path) -> str:
    """Source with comments and the module docstring removed.

    Several assertions below are about what the *code* does, and a substring
    search over the file would match the explanation in the docstring instead.
    """
    import ast
    import io
    import tokenize

    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    docstring_end = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        docstring_end = tree.body[0].end_lineno

    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if docstring_end and token.start[0] <= docstring_end and token.type != tokenize.NEWLINE:
            continue
        kept.append(token.string)
    return ' '.join(kept)


def test_the_healthcheck_cannot_dump_a_traceback():
    """A cold boot fails every probe for the first 90 seconds. One traceback
    per probe buries the real error in the logs."""
    code = _code_without_prose(HEALTHCHECK)
    assert 'except Exception' in code, 'an unreachable probe must be handled'
    assert 'traceback' not in code.lower(), 'the script must not print one'
    assert 'print_exc' not in code
    # And the failure path must actually exit non-zero, not merely log.
    assert 'return 1' in code


def test_the_healthcheck_reports_unreachable_quietly(tmp_path):
    """Point it at a closed port and check it exits 1 with a single line."""
    import socket

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        closed_port = probe.getsockname()[1]

    source = HEALTHCHECK.read_text(encoding='utf-8')
    # Same script, unreachable target.
    script = tmp_path / 'hc.py'
    script.write_text(source.replace('9000', str(closed_port)), encoding='utf-8')

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1
    lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
    assert len(lines) == 1, f'expected one line of output, got {len(lines)}:\n{result.stderr}'
    assert 'Traceback' not in result.stderr
    assert 'unreachable' in lines[0]


def test_the_image_declares_its_ports():
    body = DOCKERFILE.read_text(encoding='utf-8')
    for port in ('9000', '9001', '9002'):
        assert port in body, f'port {port} is not declared in the Dockerfile'


# --- the healthcheck follows the configured port --------------------------


def test_the_healthcheck_does_not_hardcode_a_port():
    """Found by deploying on a non-default port: the probe was pointed at 9000
    while the app listened on APP_PORT, so a correctly working container was
    reported permanently dead. The env template invites changing APP_PORT."""
    code = _code_without_prose(HEALTHCHECK)
    assert 'APP_PORT' in code
    assert "9000/" not in code, 'the probe URL must not hardcode a port'


def _load_healthcheck_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location('ah_healthcheck', HEALTHCHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_healthcheck_prefers_the_environment_variable(monkeypatch):
    module = _load_healthcheck_module()
    monkeypatch.setenv('APP_PORT', '9100')
    assert module.resolve_port() == 9100


def test_the_healthcheck_falls_back_to_the_env_file(tmp_path, monkeypatch):
    """A bare `docker run` injects no environment, but the app still reads the
    .env file in its working directory."""
    module = _load_healthcheck_module()
    monkeypatch.delenv('APP_PORT', raising=False)
    env_file = tmp_path / '.env'
    env_file.write_text('# comment\nAPP_PORT=9300\nOTHER=1\n', encoding='utf-8')
    monkeypatch.setattr(module, 'ENV_FILE', str(env_file))
    assert module.resolve_port() == 9300


@pytest.mark.parametrize('raw', ['', 'not-a-port', '0', '99999', None])
def test_a_nonsense_port_falls_back_to_the_default(raw, monkeypatch):
    module = _load_healthcheck_module()
    if raw is None:
        monkeypatch.delenv('APP_PORT', raising=False)
        monkeypatch.setattr(module, 'ENV_FILE', str(HEALTHCHECK.parent / 'does-not-exist'))
    else:
        monkeypatch.setenv('APP_PORT', raw)
    assert module.resolve_port() == 9000


def test_the_healthcheck_probes_the_configured_port(tmp_path, monkeypatch):
    """End to end on a closed port: the message must name the configured port,
    or an operator debugging a misconfiguration is told nothing useful."""
    import socket
    import subprocess

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        closed_port = probe.getsockname()[1]

    env = dict(os.environ, APP_PORT=str(closed_port))
    result = subprocess.run(
        [sys.executable, str(HEALTHCHECK)], capture_output=True, text=True,
        env=env, timeout=30,
    )
    assert result.returncode == 1
    assert str(closed_port) in result.stderr
    assert 'Traceback' not in result.stderr


# --- credentials are not committed ---------------------------------------


def test_the_compose_file_has_no_hardcoded_credential():
    """postgres-data/compose lives in the workspace repo, not the image, but it
    is the file most likely to be copied from a template into a real deploy."""
    compose = REPO.parent / 'my-bot' / 'docker' / 'docker-compose.yml'
    if not compose.exists():
        pytest.skip('workspace repo is not present')
    body = compose.read_text(encoding='utf-8')
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue  # prose about how to set it, not a setting
        if 'POSTGRES_PASSWORD' in stripped and ':-' not in stripped:
            pytest.fail(f'hardcoded password in compose: {stripped}')


def test_an_env_template_exists_and_documents_the_password():
    template = REPO.parent / 'my-bot' / '.env.example'
    if not template.exists():
        pytest.skip('workspace repo is not present')
    body = template.read_text(encoding='utf-8')
    assert 'PASSWORD=' in body
    assert 'CHANGE THIS' in body, 'the template must flag the values to replace'
