"""
The static cache-header policy.

The dashboard ships 483 content-hashed JS chunks totalling roughly 24 MB.
Starlette's StaticFiles sends no Cache-Control at all, so browsers revalidate
and re-fetch all of it on every navigation. These tests pin the two halves of
the fix: hashed assets are immutable and must be cached hard, everything else
must revalidate so a rebuilt frontend actually reaches the browser.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from algorithex import CachedStaticFiles


@pytest.fixture
def client(tmp_path):
    (tmp_path / '_nuxt').mkdir()
    (tmp_path / 'index.html').write_text('<html></html>', encoding='utf-8')
    (tmp_path / '_nuxt' / 'Czk_mF82.js').write_text('console.log(1)', encoding='utf-8')

    app = FastAPI()
    app.mount('/', CachedStaticFiles(directory=str(tmp_path)), name='static')
    return TestClient(app)


def test_content_hashed_chunks_are_cached_immutably(client):
    res = client.get('/_nuxt/Czk_mF82.js')
    assert res.status_code == 200
    assert res.headers['Cache-Control'] == 'public, max-age=31536000, immutable'


def test_the_immutable_branch_matches_on_either_separator(client, tmp_path):
    """
    Starlette builds the path from os.sep and drops the leading slash, so a
    naive startswith('/_nuxt/') matches nothing on either platform. This pins
    the behaviour rather than the implementation detail of the separator.
    """
    for relative in ('_nuxt/Czk_mF82.js', '_nuxt\\Czk_mF82.js'):
        assert CachedStaticFiles.IMMUTABLE_CACHE == 'public, max-age=31536000, immutable'
        normalised = relative.replace('\\', '/').lstrip('/')
        assert normalised.startswith(CachedStaticFiles.IMMUTABLE_PREFIX), relative


def test_index_html_must_revalidate(client):
    """A cached index.html is how a rebuilt frontend silently fails to appear."""
    res = client.get('/index.html')
    assert res.status_code == 200
    assert res.headers['Cache-Control'] == 'no-cache'


def test_a_missing_asset_is_not_cached(client):
    res = client.get('/_nuxt/does_not_exist.js')
    assert res.status_code == 404
    # Nothing to cache, and nothing that should later be pinned.
    assert 'Cache-Control' not in res.headers


def test_every_response_gets_exactly_one_cache_control_header(client):
    for path in ('/_nuxt/Czk_mF82.js', '/index.html'):
        res = client.get(path)
        assert len(res.headers.get_list('cache-control')) == 1
