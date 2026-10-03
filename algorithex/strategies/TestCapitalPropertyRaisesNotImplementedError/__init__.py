from algorithex.strategies import Strategy
import algorithex.helpers as ah
from algorithex import utils


class TestCapitalPropertyRaisesNotImplementedError(Strategy):
    def should_long(self) -> bool:
        self.capital
        return False

    def go_long(self) -> None:
        pass

    def should_cancel_entry(self):
        return False
