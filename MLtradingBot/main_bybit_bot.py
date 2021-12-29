from typing import List, Dict
from datetime import datetime
import time

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

    def get_df_features(self) -> pd.DataFrame:
        ohlc_dfs : Dict[str, pd.DataFrame] = {
            'df_btc_15m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_15M, symbol='BTCUSD')]),
            'df_btc_5m':  pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_5M, symbol='BTCUSD')]),
            'df_btc_3m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_3M, symbol='BTCUSD')]),
            'df_eth_15m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_15M, symbol='ETHUSD')]),
            'df_eth_5m': pd.DataFrame([ohlc.__dict__ for ohlc in self.api_client.get_ohlcs(time_interval=constants.DURATION_5M, symbol='ETHUSD')]),
        }
        df_features : pd.DataFrame = self.feature_creator.create_features(**ohlc_dfs)
        return df_features


def main_trade():
    data_collector : DataCollector = DataCollector()
    api_client : ApiClient = ApiClient()
    ml_judgement : MLJudgement = MLJudgement()

    logger.info('Trading start...!')
    while True:
        if not datetime.now().minute % 15 == 0:
            # キリの良い時間まで待機
            continue
    
        # 特徴量を算出
        df_features : pd.DataFrame = data_collector.get_df_features()
        #全注文をキャンセル
        logger.info('Cancel all active orders')
        api_client.cancel_all_active_orders()
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
            if (pred_buy > 0) and (pred_buy > pred_sell):
                order : Order = Order(side=constants.BUY, order_type=constants.LIMIT, qty=qty, price=buy_price)
                logger.info(f'Create entry order!: {order}')
                api_client.create_order(order=order)
            elif (pred_sell > 0) and (pred_sell > pred_buy):
                order : Order = Order(side=constants.SELL, order_type=constants.LIMIT, qty=qty, price=sell_price)
                logger.info(f'Create entry order!: {order}')
                api_client.create_order(order=order)
        # exit
        elif now_position.side == constants.BUY:
            logger.info('Exit buy position!')
            order : Order = Order(side=constants.SELL, order_type=constants.LIMIT, qty=now_position.size, price=sell_price)
            logger.info(f'Create exit order!: {order}')
            api_client.create_order(order=order)
        elif now_position.side == constants.SELL:
            logger.info('Exit sell position!')
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
        main_trade()
            
    except Exception as e:
        logger.error(e)

    finally:
        logger.info('######## END ########')
