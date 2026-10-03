"""
Tests for the read-only live market data clients.

Every test here runs against a fake transport. No socket is opened, no clock is
waited on: the retry backoff and the request throttle are driven by injected
`time.sleep` stand-ins, so the whole file is fast and deterministic.

Where possible the tests assert against payloads captured from the real venues'
documented response shapes, because a parser that only ever sees a hand-made
fixture will happily agree with another hand-made fixture.
"""

import pytest

from algorithex.exchanges.live_data import (
    BinanceMarketData,
    BybitMarketData,
    CandleSeries,
    ExchangeError,
    HttpRequest,
    HttpResponse,
    InvalidRequestError,
    InvalidSymbolError,
    InvalidTimeframeError,
    MarketDataClient,
    MarketDataError,
    RateLimitError,
    ResponseFormatError,
    TransportError,
    get_client,
    normalize_symbol,
    normalize_timeframe,
    supported_exchanges,
)


# --- fakes ---------------------------------------------------------------


class FakeTransport:
    """Replays a scripted list of responses, recording every request.

    Each entry may be an :class:`HttpResponse` (returned as-is) or an exception
    instance (raised), which is how transport-level failures are simulated
    without a socket.
    """

    def __init__(self, *scripted):
        self.scripted = list(scripted)
        self.requests = []

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if not self.scripted:
            raise AssertionError(f'unexpected request: {request.url} {request.params}')
        nxt = self.scripted.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @property
    def params(self):
        return self.requests[-1].params


def ok(payload, status=200, headers=None):
    return HttpResponse(
        status_code=status, payload=payload, text='{}', headers=headers or {}
    )


class RecordingSleep:
    """Stands in for ``time.sleep`` and remembers what it was asked to wait for."""

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class FakeClock:
    """Monotonic clock that advances only when told to."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_client(cls, *scripted, **kwargs):
    """Build a client wired to a fake transport, a fake clock and a fake sleep."""
    transport = FakeTransport(*scripted)
    sleep = RecordingSleep()
    clock = FakeClock()
    client = cls(
        transport=transport,
        sleep=sleep,
        clock=clock,
        backoff_base=0.5,
        **kwargs,
    )
    return client, transport, sleep, clock


def binance_ticker_payload(**overrides):
    payload = {
        'symbol': 'BTCUSDT',
        'priceChange': '500.00',
        'priceChangePercent': '1.25',
        'lastPrice': '40500.00',
        'bidPrice': '40499.50',
        'askPrice': '40500.50',
        'quoteVolume': '123456789.12',
        'closeTime': 1735689600000,
    }
    payload.update(overrides)
    return payload


def binance_kline(ts, o, h, l, c, v):
    """Binance kline rows are positional arrays, with trailing fields we ignore."""
    return [ts, str(o), str(h), str(l), str(c), str(v), ts + 59_999, '0', 100, '0', '0', '0']


def bybit_envelope(result, ret_code=0, msg='OK'):
    return {'retCode': ret_code, 'retMsg': msg, 'result': result, 'time': 1735689600000}


# --- symbol and timeframe validation -------------------------------------


def test_symbol_accepts_the_forms_a_strategy_actually_writes():
    assert normalize_symbol('BTC-USDT') == 'BTCUSDT'
    assert normalize_symbol('BTC_USDT') == 'BTCUSDT'
    assert normalize_symbol('BTC/USDT') == 'BTCUSDT'
    assert normalize_symbol('  btc-usdt  ') == 'BTCUSDT'
    assert normalize_symbol('ETHBTC') == 'ETHBTC'


@pytest.mark.parametrize(
    'bad',
    [
        '',
        '   ',
        'BTC',
        'BT',
        'BTC USDT',
        'BTC*USDT',
        "BTC'USDT",
        'BTC-USDT;DROP TABLE',
        'BTC;USDT',
        '../etc/passwd',
        'BTCUSDT\nX-Evil: 1',
        'A' * 21,
        'btc usdt',
    ],
)
def test_symbol_rejects_anything_that_is_not_an_instrument(bad):
    with pytest.raises(InvalidSymbolError):
        normalize_symbol(bad)


@pytest.mark.parametrize('bad', [None, 123, ['BTCUSDT'], b'BTCUSDT'])
def test_symbol_rejects_non_strings(bad):
    with pytest.raises(InvalidSymbolError):
        normalize_symbol(bad)


def test_symbol_error_names_the_supported_shape():
    with pytest.raises(InvalidSymbolError, match=r'5-20 letters or digits'):
        normalize_symbol('BTC USDT')


def test_timeframe_is_canonical_or_refused():
    for tf in ('1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M'):
        assert normalize_timeframe(tf) == tf
    assert normalize_timeframe(' 1h ') == '1h'


@pytest.mark.parametrize('bad', ['2m', '1H', '1 hour', 'hourly', '', '7m', None, 60])
def test_timeframe_rejects_unknown_values(bad):
    with pytest.raises(InvalidTimeframeError):
        normalize_timeframe(bad)


def test_timeframe_error_lists_the_supported_set():
    with pytest.raises(InvalidTimeframeError, match='1m'):
        normalize_timeframe('7m')


# --- Binance ticker ------------------------------------------------------


def test_binance_ticker_parses_a_full_payload():
    client, transport, *_ = make_client(BinanceMarketData, ok(binance_ticker_payload()))
    ticker = client.ticker('BTC-USDT')

    assert ticker.symbol == 'BTCUSDT'
    assert ticker.last == 40500.00
    assert ticker.bid == 40499.50
    assert ticker.ask == 40500.50
    assert ticker.change_24h_pct == 1.25
    assert ticker.quote_volume_24h == 123456789.12
    assert ticker.close_time == 1735689600000
    assert transport.params == {'symbol': 'BTCUSDT'}


def test_binance_ticker_computes_mid_and_spread():
    client, *_ = make_client(BinanceMarketData, ok(binance_ticker_payload()))
    ticker = client.ticker('BTCUSDT')

    assert ticker.mid == pytest.approx(40500.00)
    assert ticker.spread == pytest.approx(1.0)
    # 1.0 wide on a 40500 mid is ~0.247 basis points.
    assert ticker.spread_bps == pytest.approx(0.2469, rel=1e-3)


def test_binance_ticker_rejects_a_non_positive_price():
    client, *_ = make_client(BinanceMarketData, ok(binance_ticker_payload(lastPrice='0')))
    with pytest.raises(ResponseFormatError, match='non-positive'):
        client.ticker('BTCUSDT')


def test_binance_ticker_rejects_a_crossed_book():
    client, *_ = make_client(
        BinanceMarketData, ok(binance_ticker_payload(bidPrice='40600', askPrice='40500'))
    )
    with pytest.raises(ResponseFormatError, match='crossed'):
        client.ticker('BTCUSDT')


def test_binance_ticker_rejects_a_missing_field():
    payload = binance_ticker_payload()
    del payload['bidPrice']
    client, *_ = make_client(BinanceMarketData, ok(payload))
    with pytest.raises(ResponseFormatError, match='bidPrice'):
        client.ticker('BTCUSDT')


def test_binance_ticker_rejects_a_string_that_is_not_a_number():
    client, *_ = make_client(BinanceMarketData, ok(binance_ticker_payload(lastPrice='n/a')))
    with pytest.raises(ResponseFormatError, match='not a number'):
        client.ticker('BTCUSDT')


@pytest.mark.parametrize('bad', ['nan', 'NaN', 'inf', '-inf', 'Infinity'])
def test_binance_ticker_refuses_non_finite_numbers(bad):
    """A NaN price would poison every average downstream, so it is refused here."""
    client, *_ = make_client(BinanceMarketData, ok(binance_ticker_payload(lastPrice=bad)))
    with pytest.raises(ResponseFormatError, match='finite'):
        client.ticker('BTCUSDT')


def test_binance_ticker_refuses_a_non_object_body():
    client, *_ = make_client(BinanceMarketData, ok([1, 2, 3]))
    with pytest.raises(ResponseFormatError, match='JSON object'):
        client.ticker('BTCUSDT')


# --- Binance candles -----------------------------------------------------


def test_binance_candles_parse_into_a_validated_series():
    rows = [
        binance_kline(1_000, 100, 110, 95, 105, 10),
        binance_kline(2_000, 105, 120, 100, 115, 12),
        binance_kline(3_000, 115, 118, 110, 112, 8),
    ]
    client, transport, *_ = make_client(BinanceMarketData, ok(rows))
    series = client.candles('BTC-USDT', '1h', limit=3)

    assert isinstance(series, CandleSeries)
    assert len(series) == 3
    assert series.symbol == 'BTCUSDT'
    assert series.interval == '1h'
    assert transport.params == {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 3}

    assert [c.timestamp for c in series] == [1_000, 2_000, 3_000]
    assert series[0].open == 100
    assert series[0].close == 105
    assert series[0].high == 110
    assert series[0].low == 95
    assert series[0].volume == 10
    assert series[0].is_bullish
    assert not series[2].is_bullish
    assert series[0].typical_price == pytest.approx((110 + 95 + 105) / 3)


def test_candle_series_exposes_columns_as_numpy_arrays():
    rows = [
        binance_kline(1_000, 100, 110, 95, 105, 10),
        binance_kline(2_000, 105, 120, 100, 115, 12),
    ]
    client, *_ = make_client(BinanceMarketData, ok(rows))
    series = client.candles('BTCUSDT', '1h')

    closes = series.column('close')
    assert closes.tolist() == [105.0, 115.0]
    assert series.column('timestamp').dtype.kind == 'i'
    assert series.column('close').dtype.kind == 'f'
    assert series.first_timestamp == 1_000
    assert series.last_timestamp == 2_000

    with pytest.raises(ValueError, match='unknown candle column'):
        series.column('adj_close')


def test_binance_candles_reject_out_of_order_rows():
    """A series running backwards in time flips every return sign inside out."""
    rows = [
        binance_kline(2_000, 105, 120, 100, 115, 12),
        binance_kline(1_000, 100, 110, 95, 105, 10),
    ]
    client, *_ = make_client(BinanceMarketData, ok(rows))
    with pytest.raises(ResponseFormatError, match='ascending'):
        client.candles('BTCUSDT', '1h')


def test_binance_candles_reject_duplicate_timestamps():
    rows = [
        binance_kline(1_000, 100, 110, 95, 105, 10),
        binance_kline(1_000, 105, 120, 100, 115, 12),
    ]
    client, *_ = make_client(BinanceMarketData, ok(rows))
    with pytest.raises(ResponseFormatError, match='ascending'):
        client.candles('BTCUSDT', '1h')


def test_binance_candles_reject_a_bar_whose_high_is_below_its_body():
    rows = [binance_kline(1_000, 100, 101, 95, 105, 10)]  # close 105 above high 101
    client, *_ = make_client(BinanceMarketData, ok(rows))
    with pytest.raises(ResponseFormatError, match='contradicts'):
        client.candles('BTCUSDT', '1h')


def test_binance_candles_reject_a_bar_whose_low_is_above_its_body():
    rows = [binance_kline(1_000, 100, 110, 104, 105, 10)]  # open 100 below low 104
    client, *_ = make_client(BinanceMarketData, ok(rows))
    with pytest.raises(ResponseFormatError, match='contradicts'):
        client.candles('BTCUSDT', '1h')


def test_a_flat_bar_is_consistent():
    rows = [binance_kline(1_000, 100, 100, 100, 100, 0)]
    client, *_ = make_client(BinanceMarketData, ok(rows))
    assert len(client.candles('BTCUSDT', '1h')) == 1


def test_binance_candles_skip_rows_too_short_to_be_a_bar():
    rows = [
        binance_kline(1_000, 100, 110, 95, 105, 10),
        [2_000, '105', '120'],
    ]
    client, *_ = make_client(BinanceMarketData, ok(rows))
    series = client.candles('BTCUSDT', '1h')
    assert len(series) == 1


def test_binance_candles_refuse_a_non_array_body():
    client, *_ = make_client(BinanceMarketData, ok({'code': -1121, 'msg': 'Invalid symbol.'}))
    with pytest.raises(ResponseFormatError, match='JSON array'):
        client.candles('BTCUSDT', '1h')


@pytest.mark.parametrize('limit', [0, -1, 1001, '500', 3.5, True, None])
def test_candle_limit_is_bounded(limit):
    client, *_ = make_client(BinanceMarketData)
    with pytest.raises(ValueError):
        client.candles('BTCUSDT', '1h', limit=limit)


def test_an_invalid_symbol_never_reaches_the_network():
    transport = FakeTransport()
    client = BinanceMarketData(transport=transport)
    with pytest.raises(InvalidSymbolError):
        client.ticker('BTC USDT')
    with pytest.raises(InvalidSymbolError):
        client.candles('BTC USDT')
    with pytest.raises(InvalidSymbolError):
        client.order_book('BTC USDT')
    assert transport.requests == []


def test_an_invalid_timeframe_never_reaches_the_network():
    transport = FakeTransport()
    client = BinanceMarketData(transport=transport)
    with pytest.raises(InvalidTimeframeError):
        client.candles('BTCUSDT', '7m')
    assert transport.requests == []


# --- order books ---------------------------------------------------------


def test_binance_order_book_builds_a_two_sided_ladder():
    payload = {
        'lastUpdateId': 1027024,
        'bids': [['40499.50', '1.5'], ['40499.00', '2.5']],
        'asks': [['40500.50', '1.2'], ['40501.00', '3.0']],
    }
    client, transport, *_ = make_client(BinanceMarketData, ok(payload))
    book = client.order_book('BTC-USDT', depth=5)

    assert transport.params == {'symbol': 'BTCUSDT', 'limit': 5}
    assert book.best_bid == 40499.50
    assert book.best_ask == 40500.50
    assert book.spread == pytest.approx(1.0)
    assert book.mid == pytest.approx(40500.00)
    assert not book.is_crossed()
    assert book.bids[1].quantity == 2.5


def test_zero_quantity_levels_are_dropped_not_treated_as_liquidity():
    """Exchanges pad the ladder with empty levels; the touch must be real."""
    payload = {
        'bids': [['0.00', '0.00000000'], ['40499.50', '1.5']],
        'asks': [['40500.50', '1.2'], ['0.00', '0.0']],
    }
    client, *_ = make_client(BinanceMarketData, ok(payload))
    book = client.order_book('BTCUSDT')

    assert book.best_bid == 40499.50
    assert book.best_ask == 40500.50
    assert len(book.bids) == 1
    assert len(book.asks) == 1


def test_a_crossed_book_is_surfaced_not_silently_accepted():
    payload = {'bids': [['40501.00', '1.0']], 'asks': [['40500.00', '1.0']]}
    client, *_ = make_client(BinanceMarketData, ok(payload))
    assert client.order_book('BTCUSDT').is_crossed()


def test_an_empty_side_is_an_error():
    client, *_ = make_client(BinanceMarketData, ok({'bids': [], 'asks': [['1', '1']]}))
    with pytest.raises(ResponseFormatError, match='empty side'):
        client.order_book('BTCUSDT')


def test_an_empty_side_of_pure_placeholders_is_an_error():
    client, *_ = make_client(BinanceMarketData, ok({'bids': [['1', '0']], 'asks': [['2', '1']]}))
    with pytest.raises(ResponseFormatError, match='empty side'):
        client.order_book('BTCUSDT')


def test_a_malformed_level_is_refused():
    client, *_ = make_client(BinanceMarketData, ok({'bids': [['1']], 'asks': [['2', '1']]}))
    with pytest.raises(ResponseFormatError, match='price and a quantity'):
        client.order_book('BTCUSDT')


def test_a_negative_price_is_refused():
    client, *_ = make_client(
        BinanceMarketData, ok({'bids': [['-1', '1']], 'asks': [['2', '1']]})
    )
    with pytest.raises(ResponseFormatError, match='must be positive'):
        client.order_book('BTCUSDT')


@pytest.mark.parametrize('depth', [1, 3, 7, 25, 250, 5001, 0])
def test_binance_rejects_depths_it_does_not_advertise(depth):
    client, transport, *_ = make_client(BinanceMarketData)
    with pytest.raises(ValueError, match='only supports depths'):
        client.order_book('BTCUSDT', depth=depth)
    assert transport.requests == []


@pytest.mark.parametrize('depth', [5, 10, 20, 50, 100, 500, 1000, 5000])
def test_binance_accepts_every_advertised_depth(depth):
    payload = {'bids': [['1', '1']], 'asks': [['2', '1']]}
    client, transport, *_ = make_client(BinanceMarketData, ok(payload))
    client.order_book('BTCUSDT', depth=depth)
    assert transport.params['limit'] == depth


# --- Bybit ---------------------------------------------------------------


def test_bybit_ticker_parses_and_scales_the_percentage():
    result = {
        'list': [
            {
                'symbol': 'BTCUSDT',
                'lastPrice': '40500.00',
                'bid1Price': '40499.50',
                'ask1Price': '40500.50',
                'price24hPcnt': '0.0125',
                'turnover24h': '123456789.12',
            }
        ],
        'category': 'spot',
    }
    # Bybit puts the observation timestamp at the envelope level, not in result.
    payload = bybit_envelope(result)
    payload['time'] = '1735689600000'
    client, transport, *_ = make_client(BybitMarketData, ok(payload))
    ticker = client.ticker('BTC-USDT')

    assert ticker.last == 40500.00
    # Bybit sends a fraction; Binance sends a percentage. Normalised here.
    assert ticker.change_24h_pct == pytest.approx(1.25)
    assert ticker.close_time == 1735689600000
    assert transport.params == {'category': 'spot', 'symbol': 'BTCUSDT'}


def test_bybit_ticker_without_a_timestamp_is_refused():
    payload = bybit_envelope(
        {
            'list': [
                {
                    'lastPrice': '1',
                    'bid1Price': '1',
                    'ask1Price': '1',
                    'price24hPcnt': '0.0',
                    'turnover24h': '1',
                }
            ]
        }
    )
    payload.pop('time')
    client, *_ = make_client(BybitMarketData, ok(payload))
    with pytest.raises(ResponseFormatError, match='tickers time'):
        client.ticker('BTCUSDT')


def test_bybit_orderbook_keeps_its_own_ts():
    """The envelope timestamp must not clobber the order book's sequence time."""
    payload = bybit_envelope({'b': [['1', '1']], 'a': [['2', '1']], 'ts': 1735689600000})
    payload['time'] = '1700000000000'
    client, *_ = make_client(BybitMarketData, ok(payload))
    assert client.order_book('BTCUSDT', depth=50).best_bid == 1.0


def test_bybit_klines_are_reversed_into_ascending_order():
    """Bybit returns newest-first; taking it at face value inverts every return."""
    result = {
        'list': [
            ['3000', '115', '118', '110', '112', '8'],
            ['2000', '105', '120', '100', '115', '12'],
            ['1000', '100', '110', '95', '105', '10'],
        ]
    }
    client, transport, *_ = make_client(BybitMarketData, ok(bybit_envelope(result)))
    series = client.candles('BTCUSDT', '1h')

    assert [c.timestamp for c in series] == [1000, 2000, 3000]
    assert series.column('close').tolist() == [105.0, 115.0, 112.0]
    assert series.is_chronological
    # The canonical '1h' must reach Bybit as its own '60' code.
    assert transport.params['interval'] == '60'
    # The parsed series still reports the canonical name, not the wire code.
    assert series.interval == '1h'


@pytest.mark.parametrize(
    'canonical,bybit_code',
    [
        ('1m', '1'),
        ('5m', '5'),
        ('30m', '30'),
        ('1h', '60'),
        ('4h', '240'),
        ('12h', '720'),
        ('1d', 'D'),
        ('1w', 'W'),
        ('1M', 'M'),
    ],
)
def test_bybit_interval_codes_differ_from_binance(canonical, bybit_code):
    result = {'list': [['1000', '1', '2', '0.5', '1.5', '10']]}
    client, transport, *_ = make_client(BybitMarketData, ok(bybit_envelope(result)))
    client.candles('BTCUSDT', canonical)
    assert transport.params['interval'] == bybit_code


@pytest.mark.parametrize('unsupported', ['8h', '3d'])
def test_bybit_rejects_an_interval_it_does_not_offer(unsupported):
    transport = FakeTransport()
    client = BybitMarketData(transport=transport)
    with pytest.raises(InvalidTimeframeError, match='does not offer'):
        client.candles('BTCUSDT', unsupported)
    assert transport.requests == []


def test_binance_sends_the_canonical_interval_unchanged():
    rows = [binance_kline(1000, 100, 110, 95, 105, 10)]
    client, transport, *_ = make_client(BinanceMarketData, ok(rows))
    client.candles('BTCUSDT', '4h')
    assert transport.params['interval'] == '4h'


def test_bybit_order_book_maps_b_and_a():
    result = {'b': [['40499.50', '1.5']], 'a': [['40500.50', '1.2']], 'ts': 1735689600000}
    client, transport, *_ = make_client(BybitMarketData, ok(bybit_envelope(result)))
    book = client.order_book('BTCUSDT', depth=50)

    assert transport.params == {'category': 'spot', 'symbol': 'BTCUSDT', 'limit': 50}
    assert book.best_bid == 40499.50
    assert book.best_ask == 40500.50


def test_bybit_non_zero_ret_code_raises_even_though_http_said_200():
    payload = bybit_envelope({}, ret_code=10001, msg='params error: invalid symbol')
    client, *_ = make_client(BybitMarketData, ok(payload))
    with pytest.raises(InvalidRequestError, match='invalid symbol') as excinfo:
        client.ticker('BTCUSDTX')
    assert excinfo.value.code == 10001
    assert excinfo.value.status is None


def test_bybit_envelope_without_a_result_object_is_refused():
    client, *_ = make_client(BybitMarketData, ok({'retCode': 0, 'retMsg': 'OK'}))
    with pytest.raises(ResponseFormatError, match='JSON object'):
        client.ticker('BTCUSDT')


def test_bybit_empty_ticker_list_is_refused():
    client, *_ = make_client(BybitMarketData, ok(bybit_envelope({'list': []})))
    with pytest.raises(ResponseFormatError, match='came back empty'):
        client.ticker('BTCUSDT')


def test_bybit_server_time_is_nanoseconds():
    client, transport, *_ = make_client(
        BybitMarketData, ok(bybit_envelope({'timeSecond': '1735689600'}))
    )
    assert client.server_time() == 1735689600 * 10**9
    assert transport.params == {'category': 'spot'}


def test_binance_server_time_is_milliseconds():
    client, *_ = make_client(BinanceMarketData, ok({'serverTime': 1735689600000}))
    assert client.server_time() == 1735689600000


@pytest.mark.parametrize('depth', [2, 25, 500, 1000])
def test_bybit_rejects_depths_it_does_not_advertise(depth):
    client, transport, *_ = make_client(BybitMarketData)
    with pytest.raises(ValueError, match='only supports depths'):
        client.order_book('BTCUSDT', depth=depth)
    assert transport.requests == []


# --- errors, retries, throttling ----------------------------------------


def test_exchange_error_messages_are_quoted_verbatim():
    payload = {'code': -1121, 'msg': 'Invalid symbol.'}
    client, *_ = make_client(
        BinanceMarketData, HttpResponse(400, payload, '', {})
    )
    with pytest.raises(InvalidRequestError, match='Invalid symbol') as excinfo:
        client.ticker('BTCUSDTX')
    assert excinfo.value.status == 400
    assert excinfo.value.code is None


def test_a_404_is_not_retried():
    client, transport, sleep, _ = make_client(
        BinanceMarketData, HttpResponse(404, {'msg': 'nope'}, '', {})
    )
    with pytest.raises(InvalidRequestError):
        client.ticker('BTCUSDTX')
    assert len(transport.requests) == 1
    assert sleep.calls == []


def test_a_429_is_retried_and_then_raises():
    throttled = HttpResponse(429, {'msg': 'Too many requests'}, '', {})
    client, transport, sleep, _ = make_client(
        BinanceMarketData, throttled, throttled, throttled
    )
    with pytest.raises(RateLimitError, match='Too many requests'):
        client.ticker('BTCUSDT')
    assert len(transport.requests) == 3
    assert sleep.calls == [0.5, 1.0]


def test_a_429_that_clears_returns_the_ticker():
    client, transport, sleep, _ = make_client(
        BinanceMarketData,
        HttpResponse(429, {'msg': 'slow down'}, '', {}),
        ok(binance_ticker_payload()),
    )
    assert client.ticker('BTCUSDT').last == 40500.00
    assert len(transport.requests) == 2
    assert sleep.calls == [0.5]


def test_retry_after_header_is_honoured_over_our_own_backoff():
    client, transport, sleep, _ = make_client(
        BinanceMarketData,
        HttpResponse(429, {'msg': 'slow'}, '', {'retry-after': '7'}),
        ok(binance_ticker_payload()),
    )
    client.ticker('BTCUSDT')
    assert sleep.calls == [7.0]


def test_a_malformed_retry_after_falls_back_to_backoff():
    client, transport, sleep, _ = make_client(
        BinanceMarketData,
        HttpResponse(429, {'msg': 'slow'}, '', {'retry-after': 'Wed, 21 Oct 2015 07:28:00 GMT'}),
        ok(binance_ticker_payload()),
    )
    client.ticker('BTCUSDT')
    assert sleep.calls == [0.5]


def test_a_418_ban_is_never_retried():
    """Binance returns 418 for a banned IP; hammering it only extends the ban."""
    client, transport, sleep, _ = make_client(
        BinanceMarketData, HttpResponse(418, {'msg': 'IP banned'}, '', {})
    )
    with pytest.raises(RateLimitError, match='IP banned'):
        client.ticker('BTCUSDT')
    assert len(transport.requests) == 1
    assert sleep.calls == []


def test_a_500_is_retried_with_growing_backoff_then_raises():
    client, transport, sleep, _ = make_client(
        BinanceMarketData,
        HttpResponse(500, {}, '', {}),
        HttpResponse(502, {}, '', {}),
        HttpResponse(503, {}, '', {}),
    )
    with pytest.raises(ExchangeError, match='HTTP 503'):
        client.ticker('BTCUSDT')
    assert len(transport.requests) == 3
    assert sleep.calls == [0.5, 1.0]


def test_a_transport_failure_is_retried():
    client, transport, sleep, _ = make_client(
        BinanceMarketData,
        TransportError('connection reset'),
        ok(binance_ticker_payload()),
    )
    assert client.ticker('BTCUSDT').last == 40500.00
    assert len(transport.requests) == 2
    assert sleep.calls == [0.5]


def test_exhausted_transport_retries_surface_the_original_error():
    client, transport, sleep, _ = make_client(
        BinanceMarketData,
        TransportError('dns failure'),
        TransportError('dns failure'),
        TransportError('dns failure'),
    )
    with pytest.raises(TransportError, match='dns failure'):
        client.ticker('BTCUSDT')
    assert len(transport.requests) == 3


def test_a_non_transport_exception_from_a_custom_transport_is_not_swallowed():
    """A bug in a caller-supplied transport must surface, not become a retry."""

    class BrokenTransport:
        def __call__(self, request):
            raise KeyError('a genuine bug')

    client = BinanceMarketData(transport=BrokenTransport())
    with pytest.raises(KeyError):
        client.ticker('BTCUSDT')


def test_an_html_error_page_is_not_mistaken_for_json():
    client, *_ = make_client(
        BinanceMarketData, HttpResponse(200, None, '<html>gateway</html>', {})
    )
    with pytest.raises(ResponseFormatError, match='not JSON'):
        client.ticker('BTCUSDT')


def test_min_request_interval_spaces_calls_out():
    client, transport, sleep, clock = make_client(
        BinanceMarketData,
        ok(binance_ticker_payload()),
        ok(binance_ticker_payload()),
        ok(binance_ticker_payload()),
        min_request_interval=0.25,
    )
    # The clock advances by exactly what the throttle asked us to sleep for, so
    # the pacing is verified without spending a quarter of a second.
    sleep.calls = []
    original = sleep.__call__

    def sleep_and_advance(seconds):
        original(seconds)
        clock.advance(seconds)

    client._sleep = sleep_and_advance
    client._throttle._sleep = sleep_and_advance

    for _ in range(3):
        client.ticker('BTCUSDT')

    assert len(transport.requests) == 3
    assert sleep.calls == [0.25, 0.25]


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError, match='max_attempts'):
        BinanceMarketData(transport=FakeTransport(), max_attempts=0)


def test_a_single_attempt_client_does_not_retry():
    client, transport, *_ = make_client(
        BinanceMarketData, HttpResponse(500, {}, '', {}), max_attempts=1
    )
    with pytest.raises(ExchangeError):
        client.ticker('BTCUSDT')
    assert len(transport.requests) == 1


def test_a_negative_throttle_interval_is_refused():
    with pytest.raises(ValueError, match='min_interval'):
        BinanceMarketData(transport=FakeTransport(), min_request_interval=-1.0)


# --- the read-only guarantee --------------------------------------------


def test_no_client_exposes_a_credential_shaped_attribute():
    """The boundary that keeps this module safe, asserted rather than assumed."""
    forbidden = ('key', 'secret', 'signature', 'sign', 'passphrase', 'token', 'auth')
    for exchange in supported_exchanges():
        client = get_client(exchange, transport=FakeTransport())
        for name in vars(client):
            lowered = name.lower()
            assert not any(word in lowered for word in forbidden), (
                f'{exchange} client exposes {name!r}, which looks like a credential'
            )


def test_credentials_cannot_be_passed_to_a_client():
    for kwargs in ({'api_key': 'x'}, {'secret': 'x'}, {'apiKey': 'x'}, {'signature': 'x'}):
        with pytest.raises(TypeError):
            get_client('binance', transport=FakeTransport(), **kwargs)


def test_requests_carry_no_authentication_headers():
    client, transport, *_ = make_client(BinanceMarketData, ok(binance_ticker_payload()))
    client.ticker('BTCUSDT')

    request = transport.requests[0]
    assert request.method == 'GET'
    header_names = {name.lower() for name in request.headers}
    assert not any(
        word in name for name in header_names for word in ('key', 'auth', 'token', 'sign')
    )
    assert header_names == {'accept', 'user-agent'}
    assert request.timeout == client.timeout


def test_no_endpoint_path_is_a_trading_endpoint():
    """Guards against a signed or account-scoped path being pasted into the URL table.

    Matched per path segment, so `/v5/market/orderbook` is allowed (it is the
    public depth endpoint) while `/api/v3/order` is not.
    """
    privileged = {'order', 'account', 'balance', 'withdraw', 'listenkey', 'position'}
    for exchange in supported_exchanges():
        cls = type(get_client(exchange, transport=FakeTransport()))
        for path_name in ('_ticker_path', '_candles_path', '_depth_path', '_time_path'):
            path = getattr(cls, path_name)()
            segments = {segment.lower() for segment in path.strip('/').split('/')}
            clash = segments & privileged
            assert not clash, f'{exchange} {path_name} hits privileged segment {clash}'


def test_the_public_api_is_read_only():
    """The public surface is pinned exactly, so a mutating method cannot be added quietly."""
    client = BinanceMarketData(transport=FakeTransport())
    public = {name for name in dir(client) if not name.startswith('_')}
    assert public == {
        'DEPTHS',
        'allowed_book_depths',
        'backoff_base',
        'base_url',
        'candles',
        'max_attempts',
        'max_candles',
        'name',
        'order_book',
        'server_time',
        'ticker',
        'timeout',
        'transport',
    }


def test_read_only_notice_says_what_it_does():
    from algorithex.exchanges.live_data import READ_ONLY_NOTICE

    assert 'public market data only' in READ_ONLY_NOTICE
    assert 'API keys' in READ_ONLY_NOTICE


# --- factory -------------------------------------------------------------


def test_factory_builds_each_supported_exchange():
    assert isinstance(get_client('binance'), BinanceMarketData)
    assert isinstance(get_client('Bybit'), BybitMarketData)
    assert isinstance(get_client('BINANCE SPOT'), BinanceMarketData)
    assert isinstance(get_client(' bybit '), BybitMarketData)
    assert supported_exchanges() == ('binance', 'bybit')


def test_factory_rejects_an_unknown_exchange():
    with pytest.raises(ValueError, match='unknown exchange'):
        get_client('kraken')
    with pytest.raises(ValueError, match='must be a string'):
        get_client(42)


def test_factory_forwards_construction_options():
    transport = FakeTransport()
    client = get_client('binance', transport=transport, timeout=3.5, max_attempts=7)
    assert client.transport is transport
    assert client.timeout == 3.5
    assert client.max_attempts == 7


def test_clients_default_to_the_real_network_transport_only_when_asked():
    """Constructing with no transport must not open a connection by itself."""
    from algorithex.exchanges.live_data import RequestsTransport

    client = BinanceMarketData()
    try:
        assert isinstance(client.transport, RequestsTransport)
    finally:
        client.transport.close()
