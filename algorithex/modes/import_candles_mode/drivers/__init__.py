from algorithex.enums import exchanges
from algorithex.modes.import_candles_mode.drivers.Binance.BinanceSpot import BinanceSpot
from algorithex.modes.import_candles_mode.drivers.Binance.BinanceUSSpot import BinanceUSSpot
from algorithex.modes.import_candles_mode.drivers.Binance.BinancePerpetualFutures import BinancePerpetualFutures
from algorithex.modes.import_candles_mode.drivers.Bitfinex.BitfinexSpot import BitfinexSpot
from algorithex.modes.import_candles_mode.drivers.Coinbase.CoinbaseSpot import CoinbaseSpot
from algorithex.modes.import_candles_mode.drivers.Binance.BinancePerpetualFuturesTestnet import BinancePerpetualFuturesTestnet
from algorithex.modes.import_candles_mode.drivers.Bybit.BybitUSDTPerpetual import BybitUSDTPerpetual
from algorithex.modes.import_candles_mode.drivers.Bybit.BybitUSDTPerpetualTestnet import BybitUSDTPerpetualTestnet
from algorithex.modes.import_candles_mode.drivers.Bybit.BybitUSDCPerpetual import BybitUSDCPerpetual
from algorithex.modes.import_candles_mode.drivers.Bybit.BybitUSDCPerpetualTestnet import BybitUSDCPerpetualTestnet
from algorithex.modes.import_candles_mode.drivers.Bybit.BybitSpotTestnet import BybitSpotTestnet
from algorithex.modes.import_candles_mode.drivers.Bybit.BybitSpot import BybitSpot
from algorithex.modes.import_candles_mode.drivers.Apex.ApexOmniPerpetualTestnet import ApexOmniPerpetualTestnet
from algorithex.modes.import_candles_mode.drivers.Apex.ApexOmniPerpetual import ApexOmniPerpetual
from algorithex.modes.import_candles_mode.drivers.Gate.GateUSDTPerpetual import GateUSDTPerpetual
from algorithex.modes.import_candles_mode.drivers.Gate.GateSpot import GateSpot
from algorithex.modes.import_candles_mode.drivers.Hyperliquid.HyperliquidPerpetual import HyperliquidPerpetual
from algorithex.modes.import_candles_mode.drivers.Hyperliquid.HyperliquidPerpetualTestnet import HyperliquidPerpetualTestnet
from algorithex.modes.import_candles_mode.drivers.Lighter.LighterPerpetual import LighterPerpetual
from algorithex.modes.import_candles_mode.drivers.Lighter.LighterPerpetualTestnet import LighterPerpetualTestnet
from algorithex.modes.import_candles_mode.drivers.KuCoin.KuCoinSpot import KuCoinSpot
from algorithex.modes.import_candles_mode.drivers.KuCoin.KuCoinUSDTPerpetual import KuCoinUSDTPerpetual
from algorithex.modes.import_candles_mode.drivers.Kraken.KrakenSpot import KrakenSpot
from algorithex.modes.import_candles_mode.drivers.Kraken.KrakenPerpetual import KrakenPerpetual
from algorithex.modes.import_candles_mode.drivers.Kraken.KrakenPerpetualTestnet import KrakenPerpetualTestnet
from algorithex.services.historical_data import (
    HistoricalCandleProviderRegistry,
    MassiveCurrenciesProvider,
    MassiveFuturesProvider,
    MassiveIndicesProvider,
    MassiveStocksProvider,
)


drivers = {
    # Perpetual Futures
    exchanges.BINANCE_PERPETUAL_FUTURES: BinancePerpetualFutures,
    exchanges.BINANCE_PERPETUAL_FUTURES_TESTNET: BinancePerpetualFuturesTestnet,
    exchanges.BITFINEX_SPOT: BitfinexSpot,
    exchanges.COINBASE_SPOT: CoinbaseSpot,
    exchanges.BYBIT_USDT_PERPETUAL: BybitUSDTPerpetual,
    exchanges.BYBIT_USDT_PERPETUAL_TESTNET: BybitUSDTPerpetualTestnet,
    exchanges.BYBIT_USDC_PERPETUAL: BybitUSDCPerpetual,
    exchanges.BYBIT_USDC_PERPETUAL_TESTNET: BybitUSDCPerpetualTestnet,
    exchanges.APEX_OMNI_PERPETUAL_TESTNET: ApexOmniPerpetualTestnet,
    exchanges.APEX_OMNI_PERPETUAL: ApexOmniPerpetual,
    exchanges.GATE_USDT_PERPETUAL: GateUSDTPerpetual,
    exchanges.GATE_SPOT: GateSpot,
    exchanges.HYPERLIQUID_PERPETUAL: HyperliquidPerpetual,
    exchanges.HYPERLIQUID_PERPETUAL_TESTNET: HyperliquidPerpetualTestnet,
    exchanges.LIGHTER_PERPETUAL: LighterPerpetual,
    exchanges.LIGHTER_PERPETUAL_TESTNET: LighterPerpetualTestnet,
    exchanges.KUCOIN_USDT_PERPETUAL: KuCoinUSDTPerpetual,
    exchanges.KRAKEN_PERPETUAL: KrakenPerpetual,
    exchanges.KRAKEN_PERPETUAL_TESTNET: KrakenPerpetualTestnet,
    # Spot
    exchanges.BINANCE_SPOT: BinanceSpot,
    exchanges.BINANCE_US_SPOT: BinanceUSSpot,
    exchanges.BYBIT_SPOT_TESTNET: BybitSpotTestnet,
    exchanges.BYBIT_SPOT: BybitSpot,
    exchanges.KUCOIN_SPOT: KuCoinSpot,
    exchanges.KRAKEN_SPOT: KrakenSpot,
}


driver_names = list(drivers.keys())
historical_provider_classes = {
    **drivers,
    exchanges.MASSIVE_STOCKS: MassiveStocksProvider,
    exchanges.MASSIVE_CURRENCIES: MassiveCurrenciesProvider,
    exchanges.MASSIVE_INDICES: MassiveIndicesProvider,
    exchanges.MASSIVE_FUTURES: MassiveFuturesProvider,
}
historical_provider_names = list(historical_provider_classes)


def build_crypto_historical_provider_registry(
    provider_ids: tuple[str, ...] | None = None,
) -> HistoricalCandleProviderRegistry:
    """Build fresh crypto providers so sessions and rate-limit state are not shared between imports."""
    registry = HistoricalCandleProviderRegistry()
    selected_provider_ids = tuple(drivers) if provider_ids is None else provider_ids
    for provider_id in selected_provider_ids:
        provider_class = drivers.get(provider_id)
        if provider_class is None:
            # Use the registry's stable typed lookup error for unknown provider IDs.
            registry.get(provider_id)
        registry.register(provider_class())
    return registry


def build_historical_provider_registry(
    provider_ids: tuple[str, ...] | None = None,
) -> HistoricalCandleProviderRegistry:
    """Build all import providers without exposing data-only sources as live exchange drivers."""
    registry = HistoricalCandleProviderRegistry()
    selected_provider_ids = tuple(historical_provider_classes) if provider_ids is None else provider_ids
    for provider_id in selected_provider_ids:
        provider_class = historical_provider_classes.get(provider_id)
        if provider_class is None:
            registry.get(provider_id)
        registry.register(provider_class())
    return registry
