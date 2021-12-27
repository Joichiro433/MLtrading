from typing import List, Dict, Deque
from collections import deque
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import constants
from settings import bybit_settings as settings
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
        self.ohlcs_2_5m : Deque[Ohlc] = deque(maxlen=30)
        self.ohlcs_7_5m : Deque[Ohlc] = deque(maxlen=30)

    def get_df_features(self) -> pd.DataFrame:
        ohlc_dfs : Dict[str, pd.DataFrame] = {
            'df_btc_15m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_15M, symbol='BTCUSD')]),
            'df_btc_5m':  pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_5M, symbol='BTCUSD')]),
            'df_btc_7_5m': pd.DataFrame([ohlc.__dict__ for ohlc in self.ohlcs_7_5m]),
            'df_btc_2_5m': pd.DataFrame([ohlc.__dict__ for ohlc in self.ohlcs_2_5m]),
            'df_eth_15m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_15M, symbol='ETHUSD')]),
            'df_eth_5m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_5M, symbol='ETHUSD', num_ohlcs=30)]),
        }
        df_features : pd.DataFrame = self.feature_creator.create_features(**ohlc_dfs)
        return df_features
        
    def collect_ohlcv_2_5m(self) -> None:
        try:
            while True:
                minute : int = datetime.now().minute
                second : int = datetime.now().second
                minute_time = minute + (second/60)
                if minute_time % 2.5 == 0:
                    ohlc : Ohlc = self.api_client.get_now_ohlc()
                    logger.info(f'2.5m ohlc: {ohlc}')
                    self.ohlcs_2_5m.append(ohlc)
                    time.sleep(60 * 2)  # 2分間wait
        except Exception as e:
            logger.error(e)

    def collect_ohlcv_7_5m(self) -> None:
        try:
            while True:
                minute : int = datetime.now().minute
                second : int = datetime.now().second
                minute_time = minute + (second/60)
                if minute_time % 7.5 == 0:
                    ohlc : Ohlc = self.api_client.get_now_ohlc()
                    logger.info(f'7.5m ohlc: {ohlc}')
                    self.ohlcs_7_5m.append(ohlc)
                    time.sleep(60 * 7)  # 7分間wait
        except Exception as e:
            logger.error(e)


def main_trade():
    try:
        data_collector : DataCollector = DataCollector()
        api_client : ApiClient = ApiClient()
        ml_judgement : MLJudgement = MLJudgement()

        while len(data_collector.ohlcs_7_5m) < 8:
            # 特徴量計算に必要なデータ数が貯まるまで待機
            time.sleep(60 * 7.5)  # 7.5分wait
        logger.info('Trading start...!')
        while True:
            if datetime.now().minute % 15 == 0:
                # 特徴量を算出
                df_features : pd.DataFrame = data_collector.get_df_features()
                #全注文をキャンセル
                api_client.cancel_all_active_orders()
                while True:
                    #有効注文がなくなるまで待機
                    orders : List[Order] = api_client.get_active_orders()
                    if len(orders) == 0:
                        break
                    time.sleep(1)
                # ML予測結果を取得
                df_pred : pd.DataFrame = ml_judgement.predict(df_features=df_features)

                pred_buy : float = df_pred['y_pred_buy'].iloc[-1]  # pred>0: shoud trade, pred<0: should not trade
                pred_sell : float = df_pred['y_pred_sell'].iloc[-1]  # pred>0: shoud trade, pred<0: should not trade
                buy_price : float = utils.round_num(df_pred['buy_price'].iloc[-1])
                sell_price : float = utils.round_num(df_pred['sell_price'].iloc[-1])
                logger.info('Prediction by ML')
                logger.info(f'pred_buy: {pred_buy}')
                logger.info(f'pred_sell: {pred_sell}')
                logger.info(f'buy_price: {buy_price}')
                logger.info(f'sell_price: {sell_price}')
                now_position : Position = api_client.get_position()
                logger.info(f'Now position: {now_position}')
                # entry
                if now_position.side == constants.NONE:
                    logger.info('Entry!')
                    qty : int = int(api_client.get_available_qty() * settings.leverage * 0.95)
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
                    logger.info('Exit buy position!')
                    if not (pred_sell < 0 and pred_buy > 0):  # ML予測が、「価格がまだ上がる」場合は保留
                        order : Order = Order(side=constants.SELL, order_type=constants.LIMIT, qty=now_position.size, price=sell_price)
                        logger.info(f'Create exit order!: {order}')
                        api_client.create_order(order=order)
                elif now_position.side == constants.SELL:
                    logger.info('Exit sell position!')
                    if not (pred_buy < 0 and pred_sell > 0):  # ML予測が、「価格がまだ下がる」場合は保留
                        order : Order = Order(side=constants.BUY, order_type=constants.LIMIT, qty=now_position.size, price=buy_price)
                        logger.info(f'Create exit order!: {order}')
                        api_client.create_order(order=order)

                time.sleep(60 * 14)  # 14分wait
    except Exception as e:
        logger.error(e)


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
