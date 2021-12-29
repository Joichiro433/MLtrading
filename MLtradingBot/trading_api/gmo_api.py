from typing import Dict, List, Union, Tuple
import time
from datetime import datetime, timedelta
import sys

import requests
import pandas as pd
import pybotters

import constants
from settings import gmo_settings as settings
from logger import Logger


logger = Logger()
symbol = settings.symbol
coin = symbol[:3]


class Ohlc:
    """ローソク足の情報をもつクラス

    Attributes
    ----------
    timestamp : datetime
        ローソク足の取得時刻
    open : float
        始値
    high : float
        高値
    low : float
        安値
    close : float
        終値
    volume : float
        出来高
    """
    def __init__(
            self, 
            timestamp: datetime, 
            open: float, 
            high: float, 
            low: float, 
            close: float,
            volume: float) -> None:
        self.timestamp : datetime = timestamp
        self.open : float = open
        self.high : float = high
        self.low : float = low
        self.close : float = close
        self.volume : float = volume

    def __str__(self) -> str:
        return str(self.__dict__)


class Order:
    """注文の情報をもつクラス

    Attributes
    ----------
    side : str = constants.BUY | constants.SELL
        買い、もしくは売り
    order_type : str = constans.LIMIT | constans.MARKET | constans.STOP
        成行、指値、逆指値
    qty : int
        ポジションサイズ
    price : float | None
        注文の値段。成行注文の場合はNoneとする
    created_at : datetime | None
        注文が作成された時刻。注文を出す際はNoneと指定
    updated_at : datetime | None
        注文が更新された時刻。注文を出す際はNoneと指定
    """
    def __init__(
            self, 
            side: str, 
            order_type: str, 
            qty: int, 
            price: Union[float, None], 
            created_at: Union[datetime, None] = None,
            updated_at: Union[datetime, None] = None) -> None:
        self.side : str = side
        self.order_type : str = order_type
        self.qty : int = qty
        self.price : Union[float, None] = price
        self.created_at : Union[datetime, None] = created_at
        self.updated_at : Union[datetime, None] = updated_at

    def __str__(self) -> str:
        return str(self.__dict__)


class Position:
    """ポジションの情報をもつクラス

    Attributes
    ----------
    side : str = constants.BUY | constants.SELL | 'None'
        買い、もしくは売り
    size : int
        ポジションサイズ
    entry_price : float
        ポジションをもった時点の値段
    leverage : float
        レバレッジの値
    liq_price : float
        強制ロスカットの値段
    created_at : datetime
        ポジションを持った際の時刻
    updated_at : datetime
        ポジションを更新した際の時刻
    """
    def __init__(
            self,
            side: str,
            size: int,
            entry_price: float,
            leverage: float,
            liq_price: float,
            created_at: datetime,
            updated_at: datetime) -> None:
        self.side : str = side
        self.size : int = size
        self.entry_price : float = entry_price
        self.leverage : float = leverage
        self.liq_price : float = liq_price
        self.created_at : datetime = created_at
        self.updated_at : datetime = updated_at
    
    def __str__(self) -> str:
        return str(self.__dict__)


class ApiClinet:
    def __init__(self) -> None:
        self.apis = {'gmocoin': [settings.api_key, settings.api_secret_key]}

    # def get_now_ohlc(self) -> Ohlc:
    #     resp : requests.Response = pybotters.get('https://api.coin.z.com/public/v1/ticker', params={'symbol': coin})
    #     ohlc_dict : Dict[str, str] = resp.json()['data'][0]
    #     now_ohlc : Ohlc = Ohlc(
    #         timestamp=pd.Timestamp(ohlc_dict['timestamp']),
    #         open=float(ohlc_dict[])
    #     )
    #     return now_ohlc