from algorithex.strategies import Strategy
import algorithex.helpers as ah
from algorithex import utils


class TestDailyBalancesProperty(Strategy):
    def should_long(self) -> bool:
        return False

    def go_long(self) -> None:
        pass

    def should_cancel_entry(self):
        return False

    def before_terminate(self):
        assert len(self.daily_balances) == 10
        for d in self.daily_balances:
            assert d == 10_000
