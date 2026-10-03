from algorithex.strategies import Strategy
import algorithex.helpers as ah
from algorithex import utils


class TestOrderPriceCannotBeGreaterThanZero(Strategy):
    def should_long(self) -> bool:
        return self.price == 10

    def go_long(self) -> None:
        self.buy = 1, 0

    def should_cancel_entry(self):
        return False
