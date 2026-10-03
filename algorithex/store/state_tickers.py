from typing import List

import numpy as np

import algorithex.helpers as ah
from algorithex.libs import DynamicNumpyArray
from algorithex.models.Ticker import Ticker
from algorithex.routes import router


class TickersState:
    def __init__(self) -> None:
        self.storage = {}

    def init_storage(self) -> None:
        for ar in router.all_formatted_routes:
            exchange, symbol = ar['exchange'], ar['symbol']
            key = ah.key(exchange, symbol)
            self.storage[key] = DynamicNumpyArray((60, 5), drop_at=120)

    def add_ticker(self, ticker: np.ndarray, exchange: str, symbol: str) -> None:
        key = ah.key(exchange, symbol)

        # only process once per second
        if len(self.storage[key][:]) == 0 or ah.now_to_timestamp() - self.storage[key][-1][0] >= 1000:
            self.storage[key].append(ticker)

    def get_tickers(self, exchange: str, symbol: str) -> List[Ticker]:
        key = ah.key(exchange, symbol)
        return self.storage[key][:]

    def get_current_ticker(self, exchange: str, symbol: str) -> Ticker:
        key = ah.key(exchange, symbol)
        return self.storage[key][-1]

    def get_past_ticker(self, exchange: str, symbol: str, number_of_tickers_ago: int) -> Ticker:
        if number_of_tickers_ago > 120:
            raise ValueError('Max accepted value for number_of_tickers_ago is 120')

        number_of_tickers_ago = abs(number_of_tickers_ago)
        key = ah.key(exchange, symbol)
        return self.storage[key][-1 - number_of_tickers_ago]
