from algorithex.strategies import Strategy
import algorithex.helpers as ah
from algorithex import utils


class TestStrategyVariablesAreResetBeforeOpeningNewPosition(Strategy):
    def should_long(self) -> bool:
        return self.price in [10, 20]

    def go_long(self) -> None:
        self.buy = 1, self.price

    def update_position(self) -> None:
        if self.price == 11:
            self.liquidate()

    def on_close_position(self, order, closed_trade) -> None:
        self.take_profit = 1, self.price - 1
        self.stop_loss = 1, self.price - 1

    def should_cancel_entry(self):
        return False
