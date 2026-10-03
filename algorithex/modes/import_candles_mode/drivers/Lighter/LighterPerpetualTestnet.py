from .LighterMain import LighterMain
from algorithex.enums import exchanges


class LighterPerpetualTestnet(LighterMain):
    def __init__(self) -> None:
        super().__init__(
            name=exchanges.LIGHTER_PERPETUAL_TESTNET,
            rest_endpoint='https://testnet.zklighter.elliot.ai',
        )
