from .BybitMain import BybitMain
from algorithex.enums import exchanges


class BybitUSDTPerpetual(BybitMain):
    def __init__(self) -> None:
        super().__init__(
            name=exchanges.BYBIT_USDT_PERPETUAL,
            rest_endpoint='https://api.bybit.com',
            category='linear',
        )
