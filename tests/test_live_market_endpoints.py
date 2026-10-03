"""
Tests for the read-only live market data endpoint.

The exchange clients talk to the network, so every test here replaces
`get_client` with a stub. What is under test is the HTTP layer: that the route
is authenticated, that each parameter is bounded and validated before it could
reach a venue, that exchange failures map to statuses a user can act on, and
that the response says plainly that it is read-only.

No test opens a socket.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hashlib import sha256

from algorithex.controllers import live_market_controller
from algorithex.controllers.live_market_controller import router
from algorithex.exchanges.live_data import (
    ExchangeError,
    InvalidSymbolError,
    InvalidTimeframeError,
    RateLimitError,
    ResponseFormatError,
    TransportError,
)
from algorithex.services.env import ENV_VALUES


class StubTicker:
    def __init__(self):
        self.last = 40500.0
        self.bid = 40499.5
        self.ask = 40500.5
        self.mid = 40500.0
        self.spread = 1.0
        self.spread_bps = 0.2469
        self.change_24h_pct = 1.25
        self.quote_volume_24h = 123456789.0


class StubBook:
    def __init__(self):
        from algorithex.execution.book import BookLevel, OrderBook

        self.bids = [BookLevel(price=40499.5, quantity=1.5), BookLevel(price=40499.0, quantity=2.5)]
        self.asks = [BookLevel(price=40500.5, quantity=1.2), BookLevel(price=40501.0, quantity=3.0)]
        self.best_bid = 40499.5
        self.best_ask = 40500.5
        self.mid = 40500.0
        self.spread = 1.0
        self.is_crossed = lambda: False


class StubSeries:
    def __init__(self):
        from algorithex.exchanges.live_data import Candle, CandleSeries

        self.symbol = 'BTCUSDT'
        self.interval = '1h'
        self.candles = [
            Candle(timestamp=1000, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0),
            Candle(timestamp=2000, open=1.5, high=2.5, low=1.0, close=2.0, volume=12.0),
        ]
        self.is_chronological = True
        self.is_consistent = True

    def __len__(self):
        return len(self.candles)

    def __iter__(self):
        return iter(self.candles)

    def column(self, name):
        import numpy as np

        return np.array([getattr(c, name) for c in self.candles])


class StubClient:
    """Records what it was asked for, so tests can assert on the call."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.fail_with = None

    def ticker(self, symbol):
        self.calls.append(('ticker', symbol, None))
        if self.fail_with:
            raise self.fail_with
        return StubTicker()

    def order_book(self, symbol, depth=None):
        self.calls.append(('order_book', symbol, depth))
        if self.fail_with:
            raise self.fail_with
        return StubBook()

    def candles(self, symbol, interval, limit):
        self.calls.append(('candles', symbol, (interval, limit)))
        if self.fail_with:
            raise self.fail_with
        return StubSeries()


@pytest.fixture(autouse=True)
def _password_env():
    """Seed PASSWORD, which the auth service reads and nothing populates under pytest."""
    previous = ENV_VALUES.get('PASSWORD')
    ENV_VALUES['PASSWORD'] = 'test-password'
    yield
    if previous is None:
        ENV_VALUES.pop('PASSWORD', None)
    else:
        ENV_VALUES['PASSWORD'] = previous


def auth_headers():
    token = sha256(ENV_VALUES['PASSWORD'].encode('utf-8')).hexdigest()
    return {'Authorization': token}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class StubFactory:
    """Stands in for ``get_client``; hands out recording stubs.

    ``fail_with`` is read when a client is built rather than assigned to one,
    because the client does not exist until the request under test runs.
    """

    def __init__(self):
        self.created = []
        self.fail_with = None

    def __call__(self, exchange='binance', **kwargs):
        stub_client = StubClient(**kwargs)
        stub_client.fail_with = self.fail_with
        self.created.append(stub_client)
        return stub_client


@pytest.fixture
def stub(monkeypatch):
    """Swap the network client for a recording stub."""
    factory = StubFactory()
    monkeypatch.setattr(live_market_controller, 'get_client', factory)
    return factory


BODY = {'exchange': 'binance', 'symbol': 'BTC-USDT', 'interval': '1h', 'candles': 2}


# --- auth ----------------------------------------------------------------


def test_endpoint_requires_authentication(client, stub):
    assert client.post('/live/market', json=BODY).status_code == 401
    assert stub.created == [], 'an unauthenticated request must not reach an exchange'


def test_a_bogus_token_is_refused(client, stub):
    res = client.post('/live/market', json=BODY, headers={'Authorization': 'not-the-token'})
    assert res.status_code == 401
    assert stub.created == []


# --- success -------------------------------------------------------------


def test_market_returns_a_complete_payload(client, stub):
    res = client.post('/live/market', json=BODY, headers=auth_headers())
    assert res.status_code == 200
    d = res.json()

    assert d['read_only'] is True
    assert 'public market data only' in d['notice']
    assert d['exchange'] == 'binance'
    assert d['symbol'] == 'BTCUSDT'
    assert d['interval'] == '1h'

    assert d['ticker']['last'] == 40500.0
    assert d['ticker']['mid'] == 40500.0
    assert d['ticker']['spread_bps'] == pytest.approx(0.2469)

    assert d['order_book']['depth'] == 2
    assert d['order_book']['best_bid'] == 40499.5
    assert d['order_book']['crossed'] is False
    assert d['order_book']['bids'][0] == {'price': 40499.5, 'quantity': 1.5}
    assert d['order_book']['asks'][0] == {'price': 40500.5, 'quantity': 1.2}

    assert d['candles']['count'] == 2
    assert d['candles']['chronological'] is True
    assert d['candles']['closes'] == [1.5, 2.0]
    assert d['candles']['timestamps'] == [1000, 2000]
    assert d['candles']['rows'][0]['open'] == 1.0


def test_market_reports_per_call_latency(client, stub):
    d = client.post('/live/market', json=BODY, headers=auth_headers()).json()
    assert set(d['latency_ms']) == {'ticker', 'order_book', 'candles'}
    assert all(v >= 0 for v in d['latency_ms'].values())
    assert d['fetched_at_ms'] > 1_600_000_000_000


def test_the_symbol_is_forwarded_verbatim_to_the_client(client, stub):
    client.post('/live/market', json=BODY, headers=auth_headers())
    assert stub.created[0].calls[0] == ('ticker', 'BTC-USDT', None)


def test_depth_is_left_to_the_venue_default_when_unspecified(client, stub):
    """Bybit does not offer 20 levels, so the controller must not hard-code one."""
    client.post('/live/market', json=BODY, headers=auth_headers())
    assert stub.created[0].calls[1] == ('order_book', 'BTC-USDT', None)


def test_bybit_is_accepted(client, stub):
    body = dict(BODY, exchange='BYBIT')
    res = client.post('/live/market', json=body, headers=auth_headers())
    assert res.status_code == 200
    assert res.json()['exchange'] == 'bybit'


# --- input validation ----------------------------------------------------


def test_an_unknown_exchange_is_refused_before_any_request(client, stub):
    res = client.post('/live/market', json=dict(BODY, exchange='kraken'), headers=auth_headers())
    assert res.status_code == 400
    assert 'unknown exchange' in res.json()['detail']
    assert stub.created == []


def test_a_malformed_symbol_is_a_400(client, stub):
    stub.fail_with = InvalidSymbolError('BTC USDT is not a valid symbol')
    res = client.post('/live/market', json=BODY, headers=auth_headers())
    assert res.status_code == 400
    assert 'not a valid symbol' in res.json()['detail']


def test_an_unsupported_interval_is_a_400(client, stub):
    stub.fail_with = InvalidTimeframeError("'7m' is not a supported timeframe")
    res = client.post('/live/market', json=BODY, headers=auth_headers())
    assert res.status_code == 400


def test_a_depth_the_venue_does_not_offer_is_a_400_that_says_which(client, stub):
    stub.fail_with = ValueError('Bybit only supports depths [1, 50, 200], got 5')
    res = client.post('/live/market', json=BODY, headers=auth_headers())
    assert res.status_code == 400
    assert '[1, 50, 200]' in res.json()['detail']


@pytest.mark.parametrize('count', [0, -1, 501, 100000])
def test_the_candle_count_is_bounded(client, stub, count):
    res = client.post('/live/market', json=dict(BODY, candles=count), headers=auth_headers())
    assert res.status_code == 400
    assert 'candles must be between' in res.json()['detail']
    assert stub.created == []


def test_the_largest_allowed_candle_count_is_accepted(client, stub):
    res = client.post('/live/market', json=dict(BODY, candles=500), headers=auth_headers())
    assert res.status_code == 200


# --- exchange failures ---------------------------------------------------


def test_a_rate_limit_is_429(client, stub):
    stub.fail_with = RateLimitError('Too many requests', status=429)
    res = client.post('/live/market', json=BODY, headers=auth_headers())
    assert res.status_code == 429
    assert 'rate limit' in res.json()['detail'].lower()


def test_a_transport_failure_is_502(client, stub):
    stub.fail_with = TransportError('api.binance.com failed: connection reset')
    res = client.post('/live/market', json=BODY, headers=auth_headers())
    assert res.status_code == 502
    assert 'could not reach binance' in res.json()['detail']


def test_a_malformed_payload_is_502(client, stub):
    stub.fail_with = ResponseFormatError('lastPrice must be finite, got nan')
    res = client.post('/live/market', json=BODY, headers=auth_headers())
    assert res.status_code == 502


def test_an_exchange_rejection_is_502_and_quotes_the_exchange(client, stub):
    stub.fail_with = ExchangeError('Invalid symbol.', status=400)
    res = client.post('/live/market', json=BODY, headers=auth_headers())
    assert res.status_code == 502
    assert 'Invalid symbol.' in res.json()['detail']


def test_the_client_is_built_with_a_short_timeout_and_few_attempts(client, stub):
    client.post('/live/market', json=BODY, headers=auth_headers())
    kwargs = stub.created[0].kwargs
    assert kwargs['timeout'] == live_market_controller.UPSTREAM_TIMEOUT
    assert kwargs['timeout'] <= 15
    assert kwargs['max_attempts'] == 2


# --- the read-only surface ----------------------------------------------


def test_the_live_router_exposes_exactly_one_post_endpoint():
    """No second route can be added here without this test failing."""
    routes = [
        (r.path, tuple(sorted(r.methods)))
        for r in router.routes
        if hasattr(r, 'methods')
    ]
    assert routes == [('/live/market', ('POST',))]


def test_no_key_or_secret_is_accepted_by_the_request_model():
    """The model has no field a credential could ride in on."""
    res_fields = set(live_market_controller.MarketRequestJson.model_fields)
    assert res_fields == {'exchange', 'symbol', 'interval', 'candles', 'depth'}


def test_a_credential_in_the_body_is_dropped_and_never_forwarded(client, stub):
    """Pydantic ignores unknown keys, so prove they go nowhere rather than assume it.

    The risk this closes is specific: a client that echoes unknown fields, or a
    controller that splats the raw body into request kwargs, would happily ship
    a pasted API key to an exchange.
    """
    res = client.post(
        '/live/market',
        json=dict(BODY, api_key='leak-me', secret='leak-me-too', signature='leak-me-3'),
        headers=auth_headers(),
    )
    assert res.status_code == 200

    body_text = res.text
    for leak in ('leak-me', 'leak-me-too', 'leak-me-3'):
        assert leak not in body_text, 'a credential came back in the response'

    # Nothing the client was constructed with may look like a credential.
    forwarded = stub.created[0].kwargs
    assert not any(
        word in key.lower()
        for key in forwarded
        for word in ('key', 'secret', 'sign', 'token', 'auth', 'pass')
    )


def test_the_page_and_the_api_are_both_registered():
    from algorithex import fastapi_app

    paths = {r.path for r in fastapi_app.routes}
    assert '/live' in paths, 'the /live page route is missing'
    assert '/live/market' in paths, 'the /live/market API route is missing'


def test_the_page_file_exists_and_declares_itself_read_only():
    from pathlib import Path

    import algorithex

    html = (Path(algorithex.__file__).parent / 'static' / 'live.html').read_text(encoding='utf-8')
    assert 'READ ONLY' in html
    assert '/live/market' in html
    # The page must read the session the dashboard already writes.
    assert "localStorage.getItem('main')" in html
    assert 'authToken' in html
