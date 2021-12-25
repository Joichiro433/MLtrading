from typing import List, Dict, Deque
from collections import deque
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import constants
from utils import utils
from trading_api.bybit_api import ApiClient, Ohlc, Order, Position
from trading_brain.bybit_brain import FeatureCreator, MLJudgement
from logger import Logger


logger = Logger()
logger.remove_oldlog()


class Singleton:
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class DataCollector(Singleton):
    def __init__(self) -> None:
        self.api_client : ApiClient = ApiClient()
        self.feature_creator : FeatureCreator = FeatureCreator()
        self.ohlcs_2_5m : Deque[Ohlc] = deque(maxlen=10)
        self.ohlcs_7_5m : Deque[Ohlc] = deque(maxlen=10)
        self.is_update_2_5m : bool = False
        self.is_update_7_5m : bool = False
        self.warning_counter : int = 0

    def get_df_features(self) -> pd.DataFrame:
        while (not self.is_update_2_5m) or (not self.is_update_7_5m):
            # データが更新されるまで待機
            time.sleep(0.5)
            self.warning_counter += 1
            assert self.warning_counter < 10, f'warning_counterが10に到達'
        
        ohlc_dfs : Dict[str, pd.DataFrame] = {
            'df_btc_15m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_15M, symbol='BTCUSD')]),
            'df_btc_5m':  pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_5M, symbol='BTCUSD')]),
            'df_btc_7_5m': pd.DataFrame([ohlc.__dict__ for ohlc in self.ohlcs_7_5m]),
            'df_btc_2_5m': pd.DataFrame([ohlc.__dict__ for ohlc in self.ohlcs_2_5m]),
            'df_eth_15m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_15M, symbol='ETHUSD')]),
            'df_eth_5m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_5M, symbol='ETHUSD', num_ohlcs=10)]),
        }
        logger.debug(ohlc_dfs['df_btc_7_5m'])
        logger.debug(ohlc_dfs['df_btc_2_5m'])
        df_features : pd.DataFrame = self.feature_creator.create_features(**ohlc_dfs)
        self.is_update_2_5m = False
        self.is_update_7_5m = False
        return df_features
        
    def collect_ohlcv_2_5m(self) -> None:
        while True:
            minute : int = datetime.now().minute
            second : int = datetime.now().second
            minute_time = minute + (second/60)
            if minute_time % 2.5 == 0:
                ohlc : Ohlc = self.api_client.get_now_ohlc()
                self.ohlcs_2_5m.append(ohlc)
                self.is_update_2_5m = True
                time.sleep(60 * 2)  # 2分間wait

    def collect_ohlcv_7_5m(self) -> None:
        while True:
            minute : int = datetime.now().minute
            second : int = datetime.now().second
            minute_time = minute + (second/60)
            if minute_time % 7.5 == 0:
                ohlc : Ohlc = self.api_client.get_now_ohlc()
                self.ohlcs_7_5m.append(ohlc)
                self.is_update_7_5m = True
                time.sleep(60 * 7)  # 7分間wait


def main_trade():
    data_collector : DataCollector = DataCollector()
    api_client : ApiClient = ApiClient()
    ml_judgement : MLJudgement = MLJudgement()

    while len(data_collector.ohlcs_7_5m) <= 5:
        # 特徴量計算に必要なデータ数が貯まるまで待機
        time.sleep(60 * 7.5)  # 7.5分wait
    logger.info('Trading start...!')
    while True:
        if datetime.now().minute % 15 == 0:
            while True:
                #有効注文がなくなるまで待機
                orders : List[Order] = api_client.get_active_orders()
                if len(orders) == 0:
                    break
                time.sleep(1)

            # 特徴量を算出
            df_features : pd.DataFrame = data_collector.get_df_features()
            # ML予測結果を取得
            df_pred : pd.DataFrame = ml_judgement.predict(df_features=df_features)

            pred_buy : float = df_pred['y_pred_buy'].iloc[-1]  # pred>0: shoud trade, pred<0: should not trade
            pred_sell : float = df_pred['y_pred_sell'].iloc[-1]  # pred>0: shoud trade, pred<0: should not trade
            buy_price : float = utils.round_num(df_features['buy_price'].iloc[-1])
            sell_price = utils.round_num(df_features['sell_price'].iloc[-1])
            logger.info('Prediction by ML')
            logger.info(f'pred_buy: {pred_buy}')
            logger.info(f'pred_sell: {pred_sell}')
            now_position : Position = api_client.get_position()
            qty : int = 100
            # entry
            if now_position.side == constants.NONE:
                if pred_buy > 0:
                    order : Order = Order(side=constants.BUY, order_type=constants.LIMIT, qty=qty, price=buy_price)
                    logger.info(f'Create entry order!: {order}')
                    api_client.create_order(order=order)
                if pred_sell > 0:
                    order : Order = Order(side=constants.SELL, order_type=constants.LIMIT, qty=qty, price=sell_price)
                    logger.info(f'Create entry order!: {order}')
                    api_client.create_order(order=order)
            # exit
            elif now_position.side == constants.BUY:
                if not (pred_sell < 0 and pred_buy > 0):  # ML予測が、「価格がまだ上がる」場合は保留
                    order : Order = Order(side=constants.SELL, order_type=constants.LIMIT, qty=now_position.size, price=sell_price)
                    logger.info(f'Create exit order!: {order}')
                    api_client.create_order(order=order)
            elif now_position == constants.SELL:
                if not (pred_buy < 0 and pred_sell > 0):  # ML予測が、「価格がまだ下がる」場合は保留
                    order : Order = Order(side=constants.BUY, order_type=constants.LIMIT, qty=now_position.size, price=buy_price)
                    logger.info(f'Create exit order!: {order}')
                    api_client.create_order(order=order)

            time.sleep(60 * 14)  # 14分wait


if __name__ == '__main__':
    try:
        logger.info('####### START #######')
        data_collector : DataCollector = DataCollector()
        # キリの良い時間まで待機
        while (datetime.now().minute%15 != 14) or (datetime.now().second != 50):
            pass
        # MLBot起動
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="thread") as executor:
            executor.submit(main_trade)
            executor.submit(data_collector.collect_ohlcv_2_5m)
            executor.submit(data_collector.collect_ohlcv_7_5m)
            
    except Exception as e:
        logger.error(e)

    finally:
        logger.info('######## END ########')
