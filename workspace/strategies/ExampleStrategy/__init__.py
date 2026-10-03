from algorithex.strategies import Strategy, cached
import algorithex.indicators as ta
from algorithex import utils


class ExampleStrategy(Strategy):
    """EMA trend-following starter: longs when the fast EMA crosses above the slow EMA.

    Long-only so it runs on spot and futures. Tune via hyperparameters().
    """

    @property
    @cached
    def fast_ema(self):
        return ta.ema(self.candles, self.hp['fast_period'])

    @property
    @cached
    def slow_ema(self):
        return ta.ema(self.candles, self.hp['slow_period'])

    def hyperparameters(self):
        return [
            {'name': 'fast_period', 'type': int, 'min': 5, 'max': 25, 'default': 8},
            {'name': 'slow_period', 'type': int, 'min': 20, 'max': 60, 'default': 21},
            {'name': 'risk_pct', 'type': int, 'min': 1, 'max': 10, 'default': 5},
            {'name': 'take_profit_pct', 'type': int, 'min': 4, 'max': 30, 'default': 10},
            {'name': 'stop_loss_pct', 'type': int, 'min': 2, 'max': 15, 'default': 5},
        ]

    def should_long(self) -> bool:
        return self.fast_ema > self.slow_ema

    def should_short(self) -> bool:
        return False

    def should_cancel_entry(self) -> bool:
        return False

    def go_long(self):
        qty = utils.size_to_qty(self.balance * (self.hp['risk_pct'] / 100), self.price)
        self.buy = qty, self.price

    def go_short(self):
        pass

    def on_open_position(self, order):
        tp = self.price * (1 + self.hp['take_profit_pct'] / 100)
        sl = self.price * (1 - self.hp['stop_loss_pct'] / 100)
        self.take_profit = self.position.qty, tp
        self.stop_loss = self.position.qty, sl
