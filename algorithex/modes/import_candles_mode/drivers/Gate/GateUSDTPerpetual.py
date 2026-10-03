from .GateUSDTMain import GateUSDTMain
from algorithex.enums import exchanges


class GateUSDTPerpetual(GateUSDTMain):
    def __init__(self) -> None:
        super().__init__(
            name=exchanges.GATE_USDT_PERPETUAL,
            rest_endpoint='https://api.gateio.ws/api/v4/futures'
        )
