from typing import List, Dict, Deque
from collections import deque

import pandas as pd

from trading_api.bybit_api import ApiClient, Ohlc
from logger import Logger


class Singleton:
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class OhlcvCollector(Singleton):
    def __init__(self) -> None:
        self.api_client : ApiClient = ApiClient()
        self.ohlcs_2_5m : Deque[Ohlc] = Deque(maxlen=10)
        self.ohlcs_7_5m : Deque[Ohlc] = Deque(maxlen=10)
        
    def collect_ohlcv_2_5m(self) -> None:
        while True:
            ohlc : Ohlc = self.api_client.get_now_ohlc()
            self.ohlcs_2_5m.append(ohlc)

    def collect_ohlcv_7_5m(self) -> None:
        while True:
            pass

    def initialize_df(self) -> None:
        self.df_btc_15m: pd.DataFrame = None
        self.df_btc_5m: pd.DataFrame = None
        self.df_btc_7_5m: pd.DataFrame = None
        self.df_btc_2_5m: pd.DataFrame = None
        self.df_eth_15m: pd.DataFrame = None
        self.df_eth_5m: pd.DataFrame = None
