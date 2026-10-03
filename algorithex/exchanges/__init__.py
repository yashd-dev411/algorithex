"""Exchange drivers.

Two very different things live here.

`Exchange` / `sandbox`
    The order-placement interface the backtest engine drives, plus the
    simulated venue that fills against synthetic liquidity. Nothing in this
    package talks to a real exchange.

`live_data`
    Read-only public market data from real venues. See its module docstring for
    why it stops short of order placement.
"""

from .sandbox.Sandbox import Sandbox
from .live_data import (
    READ_ONLY_NOTICE,
    BinanceMarketData,
    BybitMarketData,
    Candle,
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
    RequestsTransport,
    ResponseFormatError,
    Ticker,
    TransportError,
    get_client,
    normalize_symbol,
    normalize_timeframe,
    supported_exchanges,
)
