"""
Read-only live market data.

This module is the first step toward the execution layer, and it stops
deliberately short of it. It reads public market data — tickers, order books,
candles — and it cannot do anything else. There is no order placement, no
request signing, no HMAC, no API-key or secret handling, and no account
endpoints anywhere in this file.

That boundary is not squeamishness, it is three separate reasons.

Safety
    Code that can place orders but has never been pointed at a real account
    will eventually be pointed at a real account. Public market data cannot
    lose anybody money no matter how it is misconfigured.

Jurisdiction
    Offering automated order placement to a third party is a regulated
    activity in most of the jurisdictions this software can be run from, and
    it drags in KYC, suitability, best-execution and order-reporting obligations
    that no open-source backtesting framework has any business taking on.
    Read-only data has none of that surface.

Honesty
    A framework that both simulates and executes quietly invites the belief
    that the two are equivalent. They are not. Simulation is a model of a
    market; execution is a participant in one. Keeping the reader here means
    the backtest results keep their meaning, and the gap between model and
    reality stays visible instead of being papered over by a lucky demo.

What it does give you
    Real prices to mark a strategy against, real depth to score it against, and
    real candles to detect that the market has stopped resembling the one the
    strategy was fitted on. That is enough to make a backtest honest.

Transport is injectable
    Every network call goes through a single ``Transport`` callable that maps an
    :class:`HttpRequest` to an :class:`HttpResponse`. Tests substitute a fake
    and never touch a socket; nothing here makes an implicit network call at
    import time or inside a constructor.

Response data is validated, not trusted
    Exchange payloads are validated at the boundary and rejected loudly: prices
    must be finite and positive, candles must be chronological and internally
    consistent (high above body, low below it), and a book may not cross. A
    silent NaN that quietly poisons a backtest is the exact failure mode this
    codebase exists to prevent.
"""

from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import-time cost avoided at runtime
    from algorithex.execution.book import OrderBook

__all__ = [
    'READ_ONLY_NOTICE',
    'InvalidSymbolError',
    'InvalidTimeframeError',
    'MarketDataError',
    'TransportError',
    'ExchangeError',
    'RateLimitError',
    'InvalidRequestError',
    'ResponseFormatError',
    'HttpRequest',
    'HttpResponse',
    'RequestsTransport',
    'Ticker',
    'Candle',
    'CandleSeries',
    'normalize_symbol',
    'normalize_timeframe',
    'TIMEFRAME_TO_MILLISECONDS',
    'MarketDataClient',
    'BinanceMarketData',
    'BybitMarketData',
    'supported_exchanges',
    'get_client',
]

READ_ONLY_NOTICE = (
    'This module reads public market data only. It cannot place, modify or '
    'cancel orders, and it never accepts, stores or transmits API keys or '
    'secrets.'
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class MarketDataError(Exception):
    """Base class for every failure this module raises."""


class TransportError(MarketDataError):
    """The request never produced an HTTP response (DNS, TLS, timeout, reset)."""


class ExchangeError(MarketDataError):
    """The exchange answered with an error.

    :param message: the exchange's own wording, when it gave one
    :param status: HTTP status code, or ``None`` for an in-band error code
    :param code: the exchange's error code (``retCode`` on Bybit, ``code`` on
        Binance), or ``None`` when the failure was purely at the HTTP layer
    """

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        code: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


class RateLimitError(ExchangeError):
    """Too many requests. Retryable, unless the exchange has banned the IP."""


class InvalidRequestError(ExchangeError):
    """The exchange rejected the request itself (bad symbol, bad parameter)."""


class ResponseFormatError(MarketDataError):
    """The body parsed as JSON but is not the shape a real payload has."""


class InvalidSymbolError(ValueError):
    """The symbol is not a well-formed instrument identifier."""


class InvalidTimeframeError(ValueError):
    """The timeframe is not one this client can request."""


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

# Public endpoints need no credentials. Sending anything resembling a key here
# would be both pointless and a liability, so the header set is fixed and shared
# by every client rather than being configurable per instance.
_PUBLIC_HEADERS: Dict[str, str] = {
    'Accept': 'application/json',
    'User-Agent': 'Algorithex/1.0 (read-only market data)',
}


@dataclass(frozen=True)
class HttpRequest:
    """One outbound GET. Carries no body: every endpoint here is a query."""

    method: str
    url: str
    params: Dict[str, Any]
    headers: Dict[str, str]
    timeout: float


@dataclass(frozen=True)
class HttpResponse:
    """One inbound response, decoded but not yet validated.

    ``payload`` is ``None`` when the body was not valid JSON. ``headers`` is
    lower-cased by :class:`RequestsTransport` so ``Retry-After`` can be read
    without guessing at the exchange's capitalisation.
    """

    status_code: int
    payload: Any = None
    text: str = ''
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300


#: Maps one :class:`HttpRequest` to one :class:`HttpResponse`.
Transport = Callable[[HttpRequest], HttpResponse]


class RequestsTransport:
    """The real network transport, built on ``requests``.

    Follows the retry convention already used by the candle-import drivers: a
    ``requests.Session`` with a ``urllib3`` ``Retry`` mounted for both schemes.
    That inner retry is in addition to the client's own retry loop, which exists
    to handle rate limits and to be observable in tests.
    """

    def __init__(self, timeout: float = 15.0, retries: int = 3) -> None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self._timeout = float(timeout)
        self.session = requests.Session()
        retry = Retry(
            total=int(retries),
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        try:
            response = self.session.get(
                request.url,
                params=request.params,
                headers=request.headers,
                timeout=request.timeout,
            )
        except Exception as exc:  # requests raises a wide family of errors
            raise TransportError(f'{request.url} failed: {exc}') from exc

        try:
            payload = response.json()
        except ValueError:
            # A rate-limit page or a proxy error is HTML. Keep the body for the
            # error message but do not pretend it parsed.
            payload = None

        return HttpResponse(
            status_code=response.status_code,
            payload=payload,
            text=response.text or '',
            headers={k.lower(): v for k, v in response.headers.items()},
        )

    def close(self) -> None:
        self.session.close()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

#: Exchange symbols are upper-case alphanumerics: BTCUSDT, ETHBTC, 1000SHIBUSDT.
_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{5,20}$')

#: Separators a human or a data vendor might leave in a symbol.
_SYMBOL_SEPARATORS = ('-', '_', '/', ':')

#: Timeframes shared by Binance and Bybit, mapped to milliseconds.
TIMEFRAME_TO_MILLISECONDS: Dict[str, int] = {
    '1m': 60_000,
    '3m': 180_000,
    '5m': 300_000,
    '15m': 900_000,
    '30m': 1_800_000,
    '1h': 3_600_000,
    '2h': 7_200_000,
    '4h': 14_400_000,
    '6h': 21_600_000,
    '8h': 28_800_000,
    '12h': 43_200_000,
    '1d': 86_400_000,
    '3d': 259_200_000,
    '1w': 604_800_000,
    '1M': 2_592_000_000,
}


def normalize_symbol(symbol: str) -> str:
    """Return the exchange's own form of ``symbol``, upper-case and unseparated.

    Jesse strategies are written as ``BTC-USDT``; the exchanges want
    ``BTCUSDT``. Both are accepted, lower case is accepted, and anything that is
    not plausibly an instrument is rejected here rather than being interpolated
    into a query string further down.

    :raises InvalidSymbolError: if the result is not ``[A-Z0-9]{5,20}``
    """
    if not isinstance(symbol, str):
        raise InvalidSymbolError(f'symbol must be a string, got {type(symbol).__name__}')

    candidate = symbol.strip().upper()
    for separator in _SYMBOL_SEPARATORS:
        candidate = candidate.replace(separator, '')

    if not _SYMBOL_PATTERN.match(candidate):
        raise InvalidSymbolError(
            f'{symbol!r} is not a valid symbol; expected 5-20 letters or digits, '
            f"optionally separated by {' or '.join(_SYMBOL_SEPARATORS)}"
        )
    return candidate


#: Binance's kline interval codes, which happen to match the canonical names.
_BINANCE_INTERVALS = {tf: tf for tf in TIMEFRAME_TO_MILLISECONDS}

#: Bybit codes the same timeframes completely differently, and this is a
#: routine source of "Invalid period!" errors: hourly bars are minutes spelled
#: as bare integers, and the daily/weekly/monthly bars are single letters.
_BYBIT_INTERVALS = {
    '1m': '1',
    '3m': '3',
    '5m': '5',
    '15m': '15',
    '30m': '30',
    '1h': '60',
    '2h': '120',
    '4h': '240',
    '6h': '360',
    '12h': '720',
    '1d': 'D',
    '1w': 'W',
    '1M': 'M',
}


def normalize_timeframe(timeframe: str) -> str:
    """Return the canonical spelling of ``timeframe``.

    :raises InvalidTimeframeError: if the timeframe is unknown
    """
    if not isinstance(timeframe, str):
        raise InvalidTimeframeError(
            f'timeframe must be a string, got {type(timeframe).__name__}'
        )
    candidate = timeframe.strip()
    if candidate not in TIMEFRAME_TO_MILLISECONDS:
        supported = ', '.join(TIMEFRAME_TO_MILLISECONDS)
        raise InvalidTimeframeError(
            f'{timeframe!r} is not a supported timeframe; expected one of {supported}'
        )
    return candidate


def _validate_limit(limit: Any, maximum: int, name: str = 'limit') -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f'{name} must be an int, got {type(limit).__name__}')
    if not 1 <= limit <= maximum:
        raise ValueError(f'{name} must be between 1 and {maximum}, got {limit}')
    return limit


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any, field_name: str) -> float:
    """Coerce to a finite float, or refuse the payload.

    ``float('nan')`` parses happily and then poisons every average it touches,
    so it is rejected here rather than discovered three modules later.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResponseFormatError(f'{field_name} is not a number: {value!r}') from exc
    if not math.isfinite(number):
        raise ResponseFormatError(f'{field_name} must be finite, got {value!r}')
    return number


def _to_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResponseFormatError(f'{field_name} is not an integer: {value!r}') from exc


def _require_mapping(payload: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ResponseFormatError(
            f'{field_name} must be a JSON object, got {type(payload).__name__}'
        )
    return payload


def _require_sequence(payload: Any, field_name: str) -> Sequence[Any]:
    if isinstance(payload, (str, bytes, Mapping)) or not isinstance(payload, Sequence):
        raise ResponseFormatError(
            f'{field_name} must be a JSON array, got {type(payload).__name__}'
        )
    return payload


def _exchange_error_message(payload: Any) -> Tuple[Optional[str], Optional[Any]]:
    """Pull the human-readable reason and error code out of an error body.

    Binance uses ``msg``/``code``, Bybit uses ``retMsg``/``retCode``. Both are
    checked so the raised error quotes the exchange rather than a generic
    string.
    """
    if not isinstance(payload, Mapping):
        return None, None
    for message_key in ('msg', 'retMsg', 'message', 'error'):
        if message_key in payload:
            message = payload[message_key]
            return (message if isinstance(message, str) else str(message)), None
    return None, None


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------

class _Throttle:
    """Spaces requests by a minimum interval.

    Both the clock and the sleep function are injected so a test can exercise
    the timing without spending real seconds. Requests never overlap: the next
    allowed time is measured from the moment the previous one was issued.
    """

    def __init__(
        self,
        min_interval: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval < 0:
            raise ValueError(f'min_interval must be non-negative, got {min_interval}')
        self._min_interval = float(min_interval)
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = self._clock()

    def wait(self) -> None:
        delay = self._next_allowed - self._clock()
        if delay > 0:
            self._sleep(delay)
        self._next_allowed = max(self._next_allowed, self._clock()) + self._min_interval


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ticker:
    """A point-in-time snapshot of an instrument."""

    symbol: str
    last: float
    bid: float
    ask: float
    change_24h_pct: float
    quote_volume_24h: float
    close_time: int

    @property
    def mid(self) -> float:
        """Mid price, the fair reference and the right mark for a backtest."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points of mid — comparable across price levels.

        A $2 spread means something very different on BTC at $60,000 than on
        DOGE at $0.06, so raw spread comparisons across symbols are meaningless.
        """
        mid = self.mid
        return 0.0 if mid == 0 else (self.spread / mid) * 10_000.0


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar. ``timestamp`` is the bar's open time in milliseconds."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass
class CandleSeries:
    """A validated run of candles, oldest first.

    :param symbol: instrument the bars belong to
    :param interval: canonical timeframe string
    :param candles: bars in ascending timestamp order
    """

    symbol: str
    interval: str
    candles: List[Candle] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.candles)

    def __iter__(self):
        return iter(self.candles)

    def __getitem__(self, index):
        return self.candles[index]

    @property
    def is_chronological(self) -> bool:
        """True when every bar starts strictly after the one before it."""
        return all(
            later.timestamp > earlier.timestamp
            for earlier, later in zip(self.candles, self.candles[1:])
        )

    @property
    def is_consistent(self) -> bool:
        """True when no bar's high or low contradicts its own body."""
        return all(
            c.high >= max(c.open, c.close, c.low) and c.low <= min(c.open, c.close, c.high)
            for c in self.candles
        )

    @property
    def first_timestamp(self) -> Optional[int]:
        return self.candles[0].timestamp if self.candles else None

    @property
    def last_timestamp(self) -> Optional[int]:
        return self.candles[-1].timestamp if self.candles else None

    def column(self, name: str) -> np.ndarray:
        """Return one OHLCV column as a float array, aligned with the bars."""
        if name not in ('timestamp', 'open', 'high', 'low', 'close', 'volume'):
            raise ValueError(f'unknown candle column {name!r}')
        dtype = int if name == 'timestamp' else float
        return np.array([getattr(c, name) for c in self.candles], dtype=dtype)

    def validate(self) -> 'CandleSeries':
        """Return self after asserting ordering and internal consistency.

        :raises ResponseFormatError: if the bars are not strictly ascending or a
            bar's high/low contradicts its own open/close
        """
        if not self.is_chronological:
            raise ResponseFormatError(
                f'{self.symbol} {self.interval} candles are not in strictly '
                f'ascending time order'
            )
        if not self.is_consistent:
            bad = next(
                (
                    c
                    for c in self.candles
                    if not (
                        c.high >= max(c.open, c.close, c.low)
                        and c.low <= min(c.open, c.close, c.high)
                    )
                ),
                None,
            )
            raise ResponseFormatError(
                f'{self.symbol} {self.interval} candle at {bad.timestamp} has an '
                f'high/low that contradicts its own body'
            )
        return self


def _build_order_book(
    raw_bids: Any,
    raw_asks: Any,
    symbol: str,
) -> 'OrderBook':
    """Build a two-sided book, dropping the empty levels exchanges pad with.

    Both venues include levels with zero quantity at the edge of the ladder.
    They are placeholders, not liquidity, and keeping them would make
    ``best_bid`` report a price nobody will trade at.
    """
    from algorithex.execution.book import BookLevel, OrderBook

    book = OrderBook()
    for side, levels in (('bid', raw_bids), ('ask', raw_asks)):
        rows = _require_sequence(levels, f'{symbol} {side} levels')
        for row in rows:
            pair = _require_sequence(row, f'{symbol} {side} level')
            if len(pair) < 2:
                raise ResponseFormatError(
                    f'{symbol} {side} level needs a price and a quantity, got {pair!r}'
                )
            price = _to_float(pair[0], f'{symbol} {side} price')
            quantity = _to_float(pair[1], f'{symbol} {side} quantity')
            if quantity <= 0:
                continue
            if price <= 0:
                raise ResponseFormatError(
                    f'{symbol} {side} price must be positive, got {price}'
                )
            level = BookLevel(price=price, quantity=quantity)
            if side == 'bid':
                book.bids.append(level)
            else:
                book.asks.append(level)

    if not book.bids or not book.asks:
        raise ResponseFormatError(
            f'{symbol} order book came back with an empty side '
            f'({len(book.bids)} bids, {len(book.asks)} asks)'
        )
    return book


# ---------------------------------------------------------------------------
# Client base
# ---------------------------------------------------------------------------

class MarketDataClient(ABC):
    """Base class for the read-only public market data endpoints of a venue.

    Subclasses declare a base URL, a path table and a parser for each shape.
    They gain retry, backoff, throttling, symbol and timeframe validation and
    typed results for free.

    There is deliberately no ``place_order``, no ``cancel_order`` and no
    ``account`` method to override. Adding one to a subclass would be the first
    step towards the execution layer this module declines to be.
    """

    #: Public REST base URL. Overridable so a test or a regional mirror can
    #: substitute its own without touching the client.
    base_url: str = ''

    #: Hard cap on ``limit`` for candle requests, enforced before the call.
    max_candles: int = 1000

    def __init__(
        self,
        transport: Optional[Transport] = None,
        timeout: float = 15.0,
        min_request_interval: float = 0.0,
        max_attempts: int = 3,
        backoff_base: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f'max_attempts must be at least 1, got {max_attempts}')
        self._sleep = sleep
        self.transport: Transport = (
            transport if transport is not None else RequestsTransport(timeout=timeout)
        )
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.backoff_base = float(backoff_base)
        self._throttle = _Throttle(min_request_interval, clock=clock, sleep=sleep)

    @classmethod
    def _extra_params(cls) -> Dict[str, Any]:
        """Parameters every request to this venue must carry.

        Empty on Binance, whose v3 endpoints infer everything from the path.
        """
        return {}

    @classmethod
    def _interval_param(cls, interval: str) -> str:
        """Translate a canonical timeframe into this venue's interval code.

        Binance accepts the canonical spelling unchanged; Bybit does not, so it
        overrides this. Raising here rather than passing a bad code through is
        the difference between a clear message and an opaque "Invalid period!".
        """
        return interval

    # -- request plumbing ---------------------------------------------------

    def _send(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        """Issue a GET with throttling and retries, returning the parsed JSON.

        Retries transport failures, 5xx and 429. A 418 is *not* retried: on
        Binance that is a ban, and hammering a banned IP only lengthens it. A
        4xx is not retried either, because resending a request the exchange
        already called malformed just burns the rate limit.
        """
        request = HttpRequest(
            method='GET',
            url=f'{self.base_url}{path}',
            params=dict(params or {}),
            headers=dict(_PUBLIC_HEADERS),
            timeout=self.timeout,
        )

        delay = self.backoff_base
        last_transport_error: Optional[TransportError] = None

        for attempt in range(1, self.max_attempts + 1):
            self._throttle.wait()
            is_last = attempt == self.max_attempts

            try:
                response = self.transport(request)
            except MarketDataError as exc:
                # A custom transport is free to raise anything; only treat
                # TransportError as retryable and let real bugs surface.
                if isinstance(exc, TransportError):
                    last_transport_error = exc
                    if is_last:
                        raise
                else:
                    raise
            else:
                status = response.status_code

                if status == 418:
                    message, _ = _exchange_error_message(response.payload)
                    raise RateLimitError(
                        message or 'exchange has banned this IP (HTTP 418)',
                        status=status,
                    )

                if status == 429:
                    if is_last:
                        message, _ = _exchange_error_message(response.payload)
                        raise RateLimitError(
                            message or 'rate limit exceeded (HTTP 429)', status=status
                        )
                    self._backoff(delay, response)
                    delay *= 2
                    continue

                if 500 <= status < 600:
                    if is_last:
                        message, code = _exchange_error_message(response.payload)
                        raise ExchangeError(
                            message or f'exchange returned HTTP {status}',
                            status=status,
                            code=code,
                        )
                    self._backoff(delay, response)
                    delay *= 2
                    continue

                if not response.is_success:
                    message, code = _exchange_error_message(response.payload)
                    error = InvalidRequestError(
                        message or f'exchange rejected the request (HTTP {status})',
                        status=status,
                        code=code,
                    )
                    raise error

                if response.payload is None:
                    raise ResponseFormatError(
                        f'{request.url} returned HTTP {status} with a body that is '
                        f'not JSON: {response.text[:200]!r}'
                    )
                return response.payload

            if not is_last:
                sleep_for = delay
                delay *= 2
                self._sleep_for(sleep_for)

        # Unreachable in practice; the loop either returns or raises.
        raise last_transport_error or TransportError('request failed with no response')

    def _backoff(self, delay: float, response: HttpResponse) -> None:
        """Sleep before a retry, honouring ``Retry-After`` when the venue sets it."""
        retry_after = response.headers.get('retry-after')
        if retry_after:
            try:
                self._sleep_for(float(retry_after))
                return
            except (TypeError, ValueError):
                # A malformed or HTTP-date Retry-After falls through to backoff.
                pass
        self._sleep_for(delay)

    def _sleep_for(self, seconds: float) -> None:
        """Indirection so tests can count backoff sleeps without waiting."""
        self._sleep(seconds)

    # -- venue-specific ----------------------------------------------------

    @abstractmethod
    def _parse_ticker(self, symbol: str, payload: Any) -> Ticker:
        """Turn the venue's ticker payload into a :class:`Ticker`."""

    @abstractmethod
    def _parse_candles(self, symbol: str, interval: str, payload: Any) -> CandleSeries:
        """Turn the venue's klines payload into a validated :class:`CandleSeries`."""

    @abstractmethod
    def _parse_order_book(self, symbol: str, payload: Any) -> 'OrderBook':
        """Turn the venue's depth payload into a two-sided book."""

    @abstractmethod
    def _check_envelope(self, payload: Any) -> Any:
        """Raise on an in-band error, return the payload ready for parsing.

        Venues that wrap their answers in a ``{"retCode": ...}`` envelope raise
        here; venues that report only through the HTTP status pass the body
        straight through and let each parser check its own shape.
        """

    # -- public read-only API ----------------------------------------------

    def ticker(self, symbol: str) -> Ticker:
        """Latest price, top of book and 24h statistics for ``symbol``.

        :param symbol: ``BTC-USDT`` or ``BTCUSDT``
        :raises InvalidSymbolError: on a malformed symbol
        :raises RateLimitError: when throttled and out of retries
        :raises ExchangeError: on any other exchange-side failure
        """
        normalized = normalize_symbol(symbol)
        payload = self._send(
            self._ticker_path(), {**self._extra_params(), 'symbol': normalized}
        )
        return self._parse_ticker(normalized, self._check_envelope(payload))

    def candles(self, symbol: str, interval: str = '1m', limit: int = 500) -> CandleSeries:
        """Closed candles for ``symbol``, oldest first.

        :param symbol: ``BTC-USDT`` or ``BTCUSDT``
        :param interval: canonical timeframe, e.g. ``1h``
        :param limit: how many bars, 1 to ``max_candles``
        :raises InvalidSymbolError: on a malformed symbol
        :raises InvalidTimeframeError: on an unknown interval
        :raises ValueError: if ``limit`` is out of range
        :raises ResponseFormatError: if the returned bars are out of order or
            internally inconsistent
        """
        normalized = normalize_symbol(symbol)
        canonical_interval = normalize_timeframe(interval)
        _validate_limit(limit, self.max_candles)

        payload = self._send(
            self._candles_path(),
            {
                **self._extra_params(),
                'symbol': normalized,
                'interval': self._interval_param(canonical_interval),
                'limit': limit,
            },
        )
        series = self._parse_candles(
            normalized, canonical_interval, self._check_envelope(payload)
        )
        return series.validate()

    def order_book(self, symbol: str, depth: Optional[int] = None) -> 'OrderBook':
        """Top ``depth`` levels of the book, best price first.

        :param symbol: ``BTC-USDT`` or ``BTCUSDT``
        :param depth: one of the venue's advertised ladder sizes, or ``None``
            to use :meth:`default_depth`
        :raises ValueError: if ``depth`` is not advertised by this venue
        :raises ResponseFormatError: if a side comes back empty
        """
        normalized = normalize_symbol(symbol)
        allowed = self.allowed_book_depths()
        if depth is None:
            depth = self.default_depth()
        if depth not in allowed:
            raise ValueError(
                f'{self.name} only supports depths {sorted(allowed)}, got {depth}'
            )

        payload = self._send(
            self._depth_path(),
            {**self._extra_params(), 'symbol': normalized, 'limit': depth},
        )
        return self._parse_order_book(normalized, self._check_envelope(payload))

    def server_time(self) -> int:
        """Exchange clock in milliseconds.

        Useful for detecting local clock drift before trusting any timestamp.
        """
        payload = self._check_envelope(
            self._send(self._time_path(), self._extra_params())
        )
        return self._parse_server_time(payload)

    @property
    def name(self) -> str:
        return type(self).__name__.replace('MarketData', '')

    @classmethod
    def allowed_book_depths(cls) -> Tuple[int, ...]:
        """Ladder sizes this venue accepts, as documented."""
        raise NotImplementedError

    @classmethod
    def default_depth(cls) -> int:
        """A sensible ladder size for callers with no preference.

        Venues disagree about which sizes exist -- Bybit offers 1, 50 and 200
        but not 20 -- so a hard-coded default is a request that fails on one of
        them. This picks the smallest advertised size that is at least 20, and
        falls back to the smallest available.
        """
        depths = sorted(cls.allowed_book_depths())
        return next((d for d in depths if d >= 20), depths[0])

    @classmethod
    def _ticker_path(cls) -> str:
        raise NotImplementedError

    @classmethod
    def _candles_path(cls) -> str:
        raise NotImplementedError

    @classmethod
    def _depth_path(cls) -> str:
        raise NotImplementedError

    @classmethod
    def _time_path(cls) -> str:
        raise NotImplementedError

    @classmethod
    def _parse_server_time(cls, payload: Any) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------------

class BinanceMarketData(MarketDataClient):
    """Read-only public market data from Binance Spot (api/v3).

    Every endpoint used here is public and unauthenticated. The signed
    ``/api/v3/order`` family is deliberately not reachable from this class.
    """

    base_url = 'https://api.binance.com'
    max_candles = 1000

    #: Advertised by Binance; anything else is rejected with HTTP 400.
    DEPTHS: Tuple[int, ...] = (5, 10, 20, 50, 100, 500, 1000, 5000)

    @classmethod
    def allowed_book_depths(cls) -> Tuple[int, ...]:
        return cls.DEPTHS

    @classmethod
    def _ticker_path(cls) -> str:
        return '/api/v3/ticker/24hr'

    @classmethod
    def _candles_path(cls) -> str:
        return '/api/v3/klines'

    @classmethod
    def _depth_path(cls) -> str:
        return '/api/v3/depth'

    @classmethod
    def _time_path(cls) -> str:
        return '/api/v3/time'

    def _check_envelope(self, payload: Any) -> Any:
        # Binance has no result envelope: errors arrive as an HTTP status with a
        # body of {"code": ..., "msg": ...}. A successful body is therefore
        # passed through untouched and each parser validates its own shape,
        # because /api/v3/klines and /api/v3/depth answer with a bare array and
        # an object respectively rather than one common envelope.
        return payload

    @classmethod
    def _parse_server_time(cls, payload: Any) -> int:
        return _to_int(payload.get('serverTime'), 'serverTime')

    def _parse_ticker(self, symbol: str, payload: Any) -> Ticker:
        row = _require_mapping(payload, f'{symbol} ticker')
        last = _to_float(row.get('lastPrice'), f'{symbol} lastPrice')
        bid = _to_float(row.get('bidPrice'), f'{symbol} bidPrice')
        ask = _to_float(row.get('askPrice'), f'{symbol} askPrice')
        if bid <= 0 or ask <= 0 or last <= 0:
            raise ResponseFormatError(
                f'{symbol} ticker has a non-positive price (last={last}, '
                f'bid={bid}, ask={ask})'
            )
        if ask < bid:
            raise ResponseFormatError(
                f'{symbol} ticker is crossed: bid {bid} above ask {ask}'
            )
        return Ticker(
            symbol=symbol,
            last=last,
            bid=bid,
            ask=ask,
            change_24h_pct=_to_float(row.get('priceChangePercent'), 'priceChangePercent'),
            quote_volume_24h=_to_float(row.get('quoteVolume'), 'quoteVolume'),
            close_time=_to_int(row.get('closeTime'), 'closeTime'),
        )

    def _parse_candles(self, symbol: str, interval: str, payload: Any) -> CandleSeries:
        rows = _require_sequence(payload, f'{symbol} klines')
        candles = [
            Candle(
                timestamp=_to_int(row[0], f'{symbol} kline openTime'),
                open=_to_float(row[1], f'{symbol} kline open'),
                high=_to_float(row[2], f'{symbol} kline high'),
                low=_to_float(row[3], f'{symbol} kline low'),
                close=_to_float(row[4], f'{symbol} kline close'),
                volume=_to_float(row[5], f'{symbol} kline volume'),
            )
            for row in rows
            if len(_require_sequence(row, f'{symbol} kline')) >= 6
        ]
        # Binance already returns klines oldest-first; validate() in the caller
        # is what guarantees it rather than an assumption baked in here.
        return CandleSeries(symbol=symbol, interval=interval, candles=candles)

    def _parse_order_book(self, symbol: str, payload: Any) -> 'OrderBook':
        row = _require_mapping(payload, f'{symbol} depth')
        return _build_order_book(
            row.get('bids'), row.get('asks'), symbol
        )


# ---------------------------------------------------------------------------
# Bybit
# ---------------------------------------------------------------------------

class BybitMarketData(MarketDataClient):
    """Read-only public market data from Bybit v5 (spot category).

    Two things make Bybit's shapes different from Binance's, and both have
    bitten people before:

    Errors arrive with HTTP 200 and a non-zero ``retCode``.
    Klines arrive newest-first. Parsing them as-is yields a series running
    backwards in time, which silently turns every return sign inside out. This
    client reverses them and the tests assert the ordering.
    """

    base_url = 'https://api.bybit.com'
    max_candles = 1000
    CATEGORY = 'spot'

    #: Advertised by Bybit for the spot category.
    DEPTHS: Tuple[int, ...] = (1, 50, 200)

    @classmethod
    def allowed_book_depths(cls) -> Tuple[int, ...]:
        return cls.DEPTHS

    @classmethod
    def _ticker_path(cls) -> str:
        return '/v5/market/tickers'

    @classmethod
    def _candles_path(cls) -> str:
        return '/v5/market/kline'

    @classmethod
    def _depth_path(cls) -> str:
        return '/v5/market/orderbook'

    @classmethod
    def _time_path(cls) -> str:
        return '/v5/market/time'

    @classmethod
    def _extra_params(cls) -> Dict[str, Any]:
        # Every v5 endpoint requires a category. Omitting it is the single
        # most common Bybit error, so it is set here once for all calls.
        return {'category': cls.CATEGORY}

    @classmethod
    def _interval_param(cls, interval: str) -> str:
        try:
            return _BYBIT_INTERVALS[interval]
        except KeyError:
            supported = ', '.join(sorted(_BYBIT_INTERVALS))
            raise InvalidTimeframeError(
                f'Bybit does not offer a {interval} interval; it supports {supported}'
            ) from None

    def _check_envelope(self, payload: Any) -> Any:
        envelope = _require_mapping(payload, 'bybit envelope')
        code = _to_int(envelope.get('retCode'), 'retCode')
        if code != 0:
            raise InvalidRequestError(
                str(envelope.get('retMsg') or 'bybit returned a non-zero retCode'),
                code=code,
            )
        result = dict(_require_mapping(envelope.get('result'), 'bybit result'))
        # Bybit reports the observation timestamp at the envelope level, outside
        # `result`. Copy it in so every parser reads the clock from one place
        # instead of each one reaching back for the raw envelope.
        if 'time' not in result and envelope.get('time') is not None:
            result['time'] = envelope['time']
        return result

    @classmethod
    def _parse_server_time(cls, payload: Any) -> int:
        # Bybit reports time in nanoseconds as a string.
        return _to_int(_require_mapping(payload, 'server time')['timeSecond'], 'timeSecond') * 10**9

    def _parse_ticker(self, symbol: str, payload: Any) -> Ticker:
        result = _require_mapping(payload, f'{symbol} tickers result')
        rows = _require_sequence(result.get('list'), f'{symbol} tickers list')
        if not rows:
            raise ResponseFormatError(f'{symbol} ticker list came back empty')
        row = _require_mapping(rows[0], f'{symbol} ticker')

        last = _to_float(row.get('lastPrice'), f'{symbol} lastPrice')
        bid = _to_float(row.get('bid1Price'), f'{symbol} bid1Price')
        ask = _to_float(row.get('ask1Price'), f'{symbol} ask1Price')
        if bid <= 0 or ask <= 0 or last <= 0:
            raise ResponseFormatError(
                f'{symbol} ticker has a non-positive price (last={last}, '
                f'bid={bid}, ask={ask})'
            )
        if ask < bid:
            raise ResponseFormatError(
                f'{symbol} ticker is crossed: bid {bid} above ask {ask}'
            )
        return Ticker(
            symbol=symbol,
            last=last,
            bid=bid,
            ask=ask,
            change_24h_pct=_to_float(
                row.get('price24hPcnt', 0.0), 'price24hPcnt'
            ) * 100.0,
            quote_volume_24h=_to_float(row.get('turnover24h'), 'turnover24h'),
            close_time=_to_int(result.get('time'), 'tickers time'),
        )

    def _parse_candles(self, symbol: str, interval: str, payload: Any) -> CandleSeries:
        result = _require_mapping(payload, f'{symbol} kline result')
        rows = _require_sequence(result.get('list'), f'{symbol} kline list')

        candles = [
            Candle(
                timestamp=_to_int(row[0], f'{symbol} kline start'),
                open=_to_float(row[1], f'{symbol} kline open'),
                high=_to_float(row[2], f'{symbol} kline high'),
                low=_to_float(row[3], f'{symbol} kline low'),
                close=_to_float(row[4], f'{symbol} kline close'),
                volume=_to_float(row[5], f'{symbol} kline volume'),
            )
            for row in rows
            if len(_require_sequence(row, f'{symbol} kline')) >= 6
        ]
        # Newest-first upstream. Reversing here is the difference between a
        # timeline and an anti-timeline.
        candles.reverse()
        return CandleSeries(symbol=symbol, interval=interval, candles=candles)

    def _parse_order_book(self, symbol: str, payload: Any) -> 'OrderBook':
        result = _require_mapping(payload, f'{symbol} orderbook result')
        return _build_order_book(result.get('b'), result.get('a'), symbol)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CLIENTS: Dict[str, type] = {
    'binance': BinanceMarketData,
    'binance spot': BinanceMarketData,
    'bybit': BybitMarketData,
    'bybit spot': BybitMarketData,
}


def supported_exchanges() -> Tuple[str, ...]:
    """Canonical exchange names accepted by :func:`get_client`."""
    return ('binance', 'bybit')


def get_client(exchange: str = 'binance', **kwargs: Any) -> MarketDataClient:
    """Build a read-only market data client.

    :param exchange: ``binance`` or ``bybit`` (case-insensitive)
    :param kwargs: passed to the client; the only meaningful ones are
        ``transport``, ``timeout``, ``min_request_interval``, ``max_attempts``,
        ``backoff_base``, ``sleep`` and ``clock``
    :raises ValueError: on an unknown exchange
    """
    if not isinstance(exchange, str):
        raise ValueError(f'exchange must be a string, got {type(exchange).__name__}')
    key = exchange.strip().lower()
    try:
        client_class = _CLIENTS[key]
    except KeyError:
        raise ValueError(
            f'unknown exchange {exchange!r}; supported: {", ".join(supported_exchanges())}'
        ) from None
    return client_class(**kwargs)
