"""
Read-only live market data over HTTP.

`exchanges.live_data` can read real prices; this controller is what makes them
reachable from the dashboard without importing a strategy into the web process
or rebuilding the Nuxt bundle. A symbol and an exchange go in, a ticker, an
order book and a run of candles come out.

The read-only boundary is inherited, not restated: this layer holds no client
that can place an order, because `MarketDataClient` has no method that could.
What it does add is the obligation to not become an open proxy to an exchange
on a user's behalf, so every parameter is validated against a whitelist before
it reaches the wire -- exchange from a fixed set, symbol by pattern, interval
by the venue's own list, depth by the venue's advertised ladder, candle count
bounded. An authenticated caller gets a small, bounded, read-only surface.

Exchange-side failures are mapped rather than leaked as 500s. A rejected symbol
is the caller's problem (400). A rate limit is 429. A transport failure or a
malformed payload is an upstream problem (502). Telling those apart is the
difference between a user fixing their input and a user filing a bug.
"""

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from algorithex.exchanges.live_data import (
    READ_ONLY_NOTICE,
    ExchangeError,
    InvalidSymbolError,
    InvalidTimeframeError,
    RateLimitError,
    ResponseFormatError,
    TransportError,
    get_client,
    supported_exchanges,
)
from algorithex.services.auth import require_auth

router = APIRouter(prefix='/live', tags=['Live Market'], dependencies=[Depends(require_auth)])

#: Upper bound on candles in one response. The chart needs a screenful; more
#: than this is bandwidth spent on pixels nobody can see.
MAX_CANDLES = 500

#: How long a single upstream call may take before the web request gives up.
#: Short on purpose: this sits behind a browser fetch, and a hung upstream
#: should surface as an error rather than pin a worker indefinitely.
UPSTREAM_TIMEOUT = 12.0


class MarketRequestJson(BaseModel):
    """Which instrument, on which venue, over which timeframe."""

    exchange: str = 'binance'
    symbol: str = 'BTC-USDT'
    interval: str = '1h'
    candles: int = 120
    depth: Optional[int] = None


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(400, str(exc))


@router.post('/market')
def market(json_request: MarketRequestJson) -> JSONResponse:
    """
    Live ticker, order book and candles for one instrument, read-only.

    All three come from one request because the dashboard wants them together
    and a user watching a price move should not see the quote, the depth and the
    chart arrive at three different moments. The per-call timings are returned
    so a slow venue is visible rather than mysterious.
    """
    exchange = (json_request.exchange or '').strip().lower()
    if exchange not in supported_exchanges():
        raise HTTPException(
            400,
            f'unknown exchange {json_request.exchange!r}; supported: '
            f'{", ".join(supported_exchanges())}',
        )

    candles = int(json_request.candles)
    if not 1 <= candles <= MAX_CANDLES:
        raise HTTPException(400, f'candles must be between 1 and {MAX_CANDLES}, got {candles}')

    client = get_client(
        exchange, timeout=UPSTREAM_TIMEOUT, max_attempts=2, min_request_interval=0.05
    )

    try:
        ticker, ticker_ms = _timed(lambda: client.ticker(json_request.symbol))
        book, book_ms = _timed(lambda: client.order_book(json_request.symbol, json_request.depth))
        series, series_ms = _timed(
            lambda: client.candles(json_request.symbol, json_request.interval, candles)
        )
    except (InvalidSymbolError, InvalidTimeframeError, ValueError) as exc:
        # Includes the depth check: asking Bybit for 20 levels is a caller
        # error, and saying which sizes are allowed is more useful than a 500.
        raise _bad_request(exc)
    except RateLimitError as exc:
        raise HTTPException(429, f'exchange rate limit reached: {exc.message}')
    except (TransportError, ResponseFormatError) as exc:
        raise HTTPException(502, f'could not reach {exchange}: {exc}')
    except ExchangeError as exc:
        raise HTTPException(502, f'{exchange} rejected the request: {exc.message}')

    return JSONResponse(
        {
            'read_only': True,
            'notice': READ_ONLY_NOTICE,
            'exchange': exchange,
            'symbol': series.symbol,
            'interval': series.interval,
            'fetched_at_ms': int(time.time() * 1000),
            'latency_ms': {
                'ticker': ticker_ms,
                'order_book': book_ms,
                'candles': series_ms,
            },
            'ticker': {
                'last': ticker.last,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'mid': ticker.mid,
                'spread': ticker.spread,
                'spread_bps': ticker.spread_bps,
                'change_24h_pct': ticker.change_24h_pct,
                'quote_volume_24h': ticker.quote_volume_24h,
            },
            'order_book': {
                'depth': len(book.bids),
                'best_bid': book.best_bid,
                'best_ask': book.best_ask,
                'mid': book.mid,
                'spread': book.spread,
                'crossed': book.is_crossed(),
                # Cumulative size is what a ladder is actually read for: the
                # headline depth is a number, the running total is liquidity.
                'bids': [
                    {'price': lvl.price, 'quantity': lvl.quantity} for lvl in book.bids
                ],
                'asks': [
                    {'price': lvl.price, 'quantity': lvl.quantity} for lvl in book.asks
                ],
            },
            'candles': {
                'count': len(series),
                'chronological': series.is_chronological,
                'consistent': series.is_consistent,
                'closes': series.column('close').tolist(),
                'timestamps': series.column('timestamp').tolist(),
                'rows': [
                    {
                        'timestamp': c.timestamp,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume,
                    }
                    for c in series
                ],
            },
        },
        status_code=200,
    )


def _timed(fn) -> Any:
    """Run ``fn`` and return ``(result, elapsed_ms)``.

    Measured per upstream call rather than for the whole request, so a venue
    being slow shows up on the specific call that was slow.
    """
    start = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - start) * 1000, 1)
