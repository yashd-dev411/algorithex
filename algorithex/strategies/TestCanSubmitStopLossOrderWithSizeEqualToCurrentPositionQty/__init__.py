from algorithex.strategies import Strategy
import algorithex.helpers as ah
from algorithex import utils


class TestCanSubmitStopLossOrderWithSizeEqualToCurrentPositionQty(Strategy):
    def should_long(self):
        return self.price == 10

    def go_long(self):
        self.buy = 1, self.price

    def on_open_position(self, order) -> None:
        self.stop_loss = self.position.qty, 8

    def on_close_position(self, order, closed_trade) -> None:
        assert order.is_stop_loss is True
        assert order.price == 8

    def should_cancel_entry(self):
        return False
