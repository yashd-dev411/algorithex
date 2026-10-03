import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hashlib import sha256

from algorithex.controllers.validation_controller import router
from algorithex.services.env import ENV_VALUES


CLEAN = """
def go(self):
    if self.close > self.price:
        return 1
    return 0
"""

CHEATING = """
def go(self):
    future = self.candles[-1]['close']
    return 1 if future > self.close else 0
"""


@pytest.fixture(autouse=True)
def _password_env():
    """
    The auth service reads ENV_VALUES['PASSWORD'] directly, and nothing
    populates it under pytest (the startup guard is skipped when unit testing).
    Seed it so the token can be derived the same way the service derives it.
    """
    previous = ENV_VALUES.get('PASSWORD')
    ENV_VALUES['PASSWORD'] = 'test-password'
    yield
    if previous is None:
        ENV_VALUES.pop('PASSWORD', None)
    else:
        ENV_VALUES['PASSWORD'] = previous


def auth_headers():
    """
    The token the service expects.

    `is_valid_token` compares the raw header value against sha256 of the
    configured password with no "Bearer " prefix, so the header is the bare
    digest. Deriving it from ENV_VALUES keeps the test honest if the password
    changes rather than hardcoding a digest that silently rots.
    """
    token = sha256(ENV_VALUES['PASSWORD'].encode('utf-8')).hexdigest()
    return {'Authorization': token}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def configs(n_configs=12, n_bars=600, seed=41):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.01, (n_configs, n_bars)) + np.linspace(0.006, 0.0015, n_configs)[:, None]).tolist()


# --- look-ahead -----------------------------------------------------------


def test_a_clean_strategy_passes(client):
    res = client.post('/validate/lookahead', json={'source': CLEAN}, headers=auth_headers())
    assert res.status_code == 200
    body = res.json()
    assert body['clean'] is True
    assert body['verdict'] == 'PASS'
    assert body['errors'] == []


def test_a_cheating_strategy_fails_and_names_the_line(client):
    res = client.post('/validate/lookahead', json={'source': CHEATING}, headers=auth_headers())
    assert res.status_code == 200
    body = res.json()
    assert body['clean'] is False
    assert body['verdict'] == 'FAIL'
    assert body['errors']
    assert 'negative indexing' in body['errors'][0]


def test_lookahead_rejects_empty_and_oversized_source(client):
    assert client.post('/validate/lookahead', json={'source': '   '}, headers=auth_headers()).status_code == 400
    # 2_400_000 characters, comfortably past the 2_000_000 cap.
    huge = client.post(
        '/validate/lookahead',
        json={'source': '# pad\n' * 400000},
        headers=auth_headers(),
    )
    assert huge.status_code == 400


def test_lookahead_requires_authentication(client):
    assert client.post('/validate/lookahead', json={'source': CLEAN}).status_code == 401


# --- overfitting ----------------------------------------------------------


def test_a_real_edge_passes_the_overfitting_checks(client):
    matrix = configs()
    res = client.post(
        '/validate/overfitting',
        json={'configurations': matrix, 'best': matrix[0]},
        headers=auth_headers(),
    )
    assert res.status_code == 200
    body = res.json()
    assert body['verdict'] == 'PASSES_OVERFITTING_CHECKS'
    assert body['pbo']['pbo'] < 0.5
    assert body['plateau']['is_plateau'] is True


def test_pure_noise_is_reported_as_failing(client):
    rng = np.random.default_rng(7)
    matrix = rng.normal(0, 0.01, (30, 800)).tolist()
    res = client.post(
        '/validate/overfitting',
        json={'configurations': matrix, 'best': matrix[0]},
        headers=auth_headers(),
    )
    assert res.status_code == 200
    assert res.json()['verdict'].startswith('FAILS')


def test_overfitting_requires_at_least_two_configurations(client):
    res = client.post(
        '/validate/overfitting',
        json={'configurations': [[0.0] * 50], 'best': [0.0] * 50},
        headers=auth_headers(),
    )
    assert res.status_code == 400


def test_mismatched_lengths_are_rejected_rather_than_guessed(client):
    matrix = configs()
    res = client.post(
        '/validate/overfitting',
        json={'configurations': matrix, 'best': matrix[0][:100]},
        headers=auth_headers(),
    )
    assert res.status_code == 400
    assert 'bars' in res.json()['detail']


def test_odd_n_blocks_is_rejected(client):
    matrix = configs()
    res = client.post(
        '/validate/overfitting',
        json={'configurations': matrix, 'best': matrix[0], 'n_blocks': 7},
        headers=auth_headers(),
    )
    assert res.status_code == 400
    assert 'even integer' in res.json()['detail']


def test_too_few_blocks_for_the_series_is_rejected(client):
    matrix = configs(n_bars=20)
    res = client.post(
        '/validate/overfitting',
        json={'configurations': matrix, 'best': matrix[0], 'n_blocks': 64},
        headers=auth_headers(),
    )
    assert res.status_code == 400


def test_non_finite_values_are_rejected(client):
    """
    NaN cannot be sent through strict JSON, but Starlette's decoder accepts the
    non-standard NaN literal, so a hand-rolled client can still smuggle one in.
    It must be refused with a 400 rather than producing a NaN report.
    """
    body = (
        '{"configurations": [[NaN, 0.01], [0.01, 0.02]], '
        '"best": [NaN, 0.01]}'
    )
    res = client.post(
        '/validate/overfitting',
        content=body,
        headers={**auth_headers(), 'Content-Type': 'application/json'},
    )
    assert res.status_code == 400
    assert 'non-finite' in res.json()['detail']


def test_overfitting_requires_authentication(client):
    matrix = configs()
    assert client.post(
        '/validate/overfitting', json={'configurations': matrix, 'best': matrix[0]}
    ).status_code == 401
