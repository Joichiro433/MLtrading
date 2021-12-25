from typing import List, Dict, Tuple, Any
import os

import numpy as np
import pandas as pd
from nptyping import NDArray
import joblib
import talib

from constants import BYBIT_FEAURES, BYBIT_FINE_FEAURES
from logger import Logger

logger = Logger()

Model = Any


class MLJudgement:
    def __init__(self) -> None:
        features_btc5m : List[str] = [feature + '_btc5m' for feature in BYBIT_FINE_FEAURES]
        features_btc2_5m : List[str] = [feature + '_btc2_5m' for feature in BYBIT_FINE_FEAURES]
        features_btc7_5m : List[str] = [feature + '_btc7_5m' for feature in BYBIT_FINE_FEAURES]
        features_eth : List[str] = [feature + '_eth' for feature in BYBIT_FEAURES]
        features_eth5m : List[str] = [feature + '_eth5m' for feature in BYBIT_FINE_FEAURES]
        self.features : List[str] = BYBIT_FEAURES + features_btc5m + features_btc2_5m + features_btc7_5m + features_eth + features_eth5m  # 使用する特徴量
        self.models_dir_path : str = os.path.join('trained_models', 'bybit')
        self.regression_model_names : List[str] = ['gbdt', 'dart', 'goss', 'ridge']  # Blending重み最適化を行った順番
        self.classification_model_names : List[str] = ['gbdt_class', 'dart_class', 'goss_class']  # Blending重み最適化を行った順番
        self.regression_models, self.classification_models = self._load_models()
        self.blending_weights : NDArray[float] = np.load(os.path.join(self.models_dir_path, 'blending_weights.npy'), allow_pickle='TRUE')
        assert len(self.blending_weights) == 7, f'blending_weightsの要素数が7で無く不正. length of blendgin_weights: {self.blending_weights}'

    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        X = df_features[self.features]
        buy_preds : List[NDArray[int]] = []
        sell_preds : List[NDArray[int]] = []
        # 回帰モデルで予測
        for model_name in self.regression_model_names:
            buy_model : Model = self.regression_models['buy'][model_name]
            buy_pred : NDArray[float] = buy_model.predict(X)
            buy_preds.append(np.sign(buy_pred))  # 1: should trade, -1: should not trade
            sell_model : Model = self.regression_models['sell'][model_name]
            sell_pred : NDArray[float] = sell_model.predict(X)
            sell_preds.append(np.sign(sell_pred))  # 1: should trade, -1: should not trade
        # 分類モデルで予測
        for model_name in self.classification_model_names:
            buy_model : Model = self.classification_models['buy'][model_name]
            buy_pred : NDArray[int] = buy_model.predict(X)
            buy_preds.append(buy_pred)  # 1: should trade, -1: should not trade
            sell_model : Model = self.classification_models['sell'][model_name]
            sell_pred : NDArray[int] = sell_model.predict(X)
            sell_preds.append(sell_pred)  # 1: should trade, -1: should not trade
        
        df = df_features.copy()
        # order価格
        df['buy_price'] = df['close'] - df['ATR'] * 0.5
        df['buy_price'] = df['close'] + df['ATR'] * 0.5
        # Blendingで予測
        df['y_pred_buy'] = np.average(buy_preds, axis=0, weights=self.blending_weights)
        df['y_pred_sell'] = np.average(sell_preds, axis=0, weights=self.blending_weights)
        return df

    def _load_models(self) -> Tuple[Dict[str, Dict[str, Model]]]:
        regression_models : Dict[str, Dict[str, Model]] = {
            'buy': {},
            'sell': {},
        }
        classification_models : Dict[str, Dict[str, Model]] = {
            'buy': {},
            'sell': {},
        }
        for model_name in self.regression_model_names:
            regression_models['buy'][model_name] = joblib.load(os.path.join(self.models_dir_path, f'{model_name}_buy.xz'))
            regression_models['sell'][model_name] = joblib.load(os.path.join(self.models_dir_path, f'{model_name}_sell.xz'))
        for model_name in self.classification_model_names:
            classification_models['buy'][model_name] = joblib.load(os.path.join(self.models_dir_path, f'{model_name}_buy.xz'))
            classification_models['sell'][model_name] = joblib.load(os.path.join(self.models_dir_path, f'{model_name}_sell.xz'))

        assert len(regression_models['buy']) == 4
        assert len(regression_models['sell']) == 4
        assert len(classification_models['buy']) == 3
        assert len(classification_models['sell']) == 3

        return regression_models, classification_models


class FeatureCreator:
    def create_features(
            self, 
            df_btc_15m: pd.DataFrame,
            df_btc_5m: pd.DataFrame,
            df_btc_7_5m: pd.DataFrame,
            df_btc_2_5m: pd.DataFrame,
            df_eth_15m: pd.DataFrame,
            df_eth_5m: pd.DataFrame) -> pd.DataFrame:
        # 各タイムスケールで特徴量を計算
        df = self._calc_features(df_ohlcvs=df_btc_15m)
        df_btc_5mf = self._calc_fine_timescale_features(df_ohlcvs=df_btc_5m)
        df_btc_2_5mf = self._calc_fine_timescale_features(df_ohlcvs=df_btc_2_5m)
        df_btc_7_5mf = self._calc_fine_timescale_features(df_ohlcvs=df_btc_7_5m)
        df_ethf = self._calc_features(df_ohlcvs=df_eth_15m)
        df_eth_5mf = self._calc_fine_timescale_features(df_ohlcvs=df_eth_5m)
        # 15分間隔に合わせて、dfを結合
        df = pd.merge(df, self._get_every_15min_datas(df_btc_5mf), on='timestamp', suffixes=['', '_btc5m'])
        df = pd.merge(df, self._get_every_15min_datas(df_btc_2_5mf), on='timestamp', suffixes=['', '_btc2_5m'])
        df = pd.merge(df, self._get_every_15min_datas(df_btc_7_5mf), on='timestamp', suffixes=['', '_btc7_5m'])
        df = pd.merge(df, df_ethf, on='timestamp', suffixes=['', '_eth'])
        df = pd.merge(df, self._get_every_15min_datas(df_eth_5mf), on='timestamp', suffixes=['', '_eth5m'])
        df = df.set_index('timestamp')

        logger.info('Created features')
        logger.info(df[['timestamp', 'open', 'high', 'low', 'close']].tail(2))
        df_features = df.dropna()
        return df_features

    def _calc_features(self, df_ohlcvs: pd.DataFrame) -> pd.DataFrame:
        df = df_ohlcvs.copy()
        open = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']

        hilo = (df['high'] + df['low']) / 2
        df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        df['BBANDS_upperband'] -= hilo
        df['BBANDS_middleband'] -= hilo
        df['BBANDS_lowerband'] -= hilo
        df['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
        df['EMA'] = talib.EMA(close, timeperiod=30) - hilo
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close) - hilo
        df['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
        df['MA'] = talib.MA(close, timeperiod=30, matype=0) - hilo
        df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
        df['SMA'] = talib.SMA(close, timeperiod=30) - hilo
        df['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
        df['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
        df['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
        df['WMA'] = talib.WMA(close, timeperiod=30) - hilo

        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
        df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
        df['BOP'] = talib.BOP(open, high, low, close)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        df['DX'] = talib.DX(high, low, close, timeperiod=14)
        df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        # skip MACDEXT MACDFIX たぶん同じなので
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
        df['MOM'] = talib.MOM(close, timeperiod=10)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
        df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df['TRIX'] = talib.TRIX(close, timeperiod=30)
        df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        df['AD'] = talib.AD(high, low, close, volume)
        df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        df['OBV'] = talib.OBV(close, volume)

        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
        df['TRANGE'] = talib.TRANGE(high, low, close)

        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
        df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
        df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
        df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close)
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        df['BETA'] = talib.BETA(high, low, timeperiod=5)
        df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
        df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
        df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
        df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
        df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
        df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)

        df['uphige_size'] = self._up_hige_size(df)
        df['downhige_size'] = self._down_hige_size(df)

        df['pct'] = df['close'].pct_change()  # 変化率
        # 平均足を使った戦略: https://note.com/btcml/n/n6198a3714fe5
        df['heikin_cl'] = 0.25 * (df['open'] + df['high'] + df['low'] + df['close'])
        df['heikin_op'] = df['heikin_cl'].ewm(1, adjust=False).mean().shift(1)
        for span in [5, 15, 25]:
            df[f'pct_mean{span}']= df['pct'].rolling(span).mean()  # 変化率の偏差
            df[f'pct_std{span}']= df['pct'].rolling(span).std()  # 変化率の偏差
            df[f'heikin_cl_mean{span}'] = df['heikin_cl'].rolling(span).mean()
            df[f'heikin_cl_std{span}'] = df['heikin_cl'].rolling(span).std()
            df[f'heikin_op_mean{span}'] = df['heikin_op'].rolling(span).mean()
            df[f'heikin_op_std{span}'] = df['heikin_op'].rolling(span).std()

        return df

    def _calc_fine_timescale_features(self, df_ohlcs: pd.DataFrame) -> pd.DataFrame:
        df = df_ohlcs.copy()

        df['pct'] = df['close'].pct_change()  # 変化率
        df['pct_mean5']= df['pct'].rolling(5).std()  # 変化率の平均
        df['pct_std5']= df['pct'].rolling(5).std()  # 変化率の偏差
        
        df['uphige_size'] = self._up_hige_size(df)
        df['downhige_size'] = self._down_hige_size(df)

        # 平均足を使った戦略: https://note.com/btcml/n/n6198a3714fe5
        df['heikin_cl'] = 0.25 * (df['open'] + df['high'] + df['low'] + df['close'])
        df['heikin_op'] = df['heikin_cl'].ewm(1, adjust=False).mean().shift(1)
        df['heikin_cl_mean5'] = df['heikin_cl'].rolling(5).mean()
        df['heikin_cl_std5'] = df['heikin_cl'].rolling(5).std()
        df['heikin_op_mean5'] = df['heikin_op'].rolling(5).mean()
        df['heikin_op_std5'] = df['heikin_op'].rolling(5).std()

        return df

    def _up_hige_size(df: pd.DataFrame) -> NDArray[float]:
        """上ヒゲの大きさ"""
        df = df.copy()
        uphige = np.zeros(len(df))
        high = df.high.values
        close = df.close.values
        open_ = df.open.values
        # close とopenの高い方を判定
        close_or_open = df.open.values - df.close.values
        close_or_open_sign = np.sign(close_or_open)
        close_or_open_sign = np.where(close_or_open_sign == 0, 1, close_or_open_sign)
        # 陽線
        for i in range(len(close_or_open_sign)):
            sig = close_or_open_sign[i]
            h = high[i]
            o = open_[i]
            c = close[i]
            if sig == 1:
                uphige[i] = (h - o) / c
            else:
                uphige[i] = (h - c) / c
        uphige = uphige / close
        return uphige

    def _down_hige_size(df: pd.DataFrame) -> NDArray[float]:
        """下ヒゲの大きさ"""
        df = df.copy()
        downhige = np.zeros(len(df))
        low = df.low.values
        close = df.close.values
        open_ = df.open.values
        # close とopenの高い方を判定
        close_or_open = df.open.values - df.close.values
        close_or_open_sign = np.sign(close_or_open)
        close_or_open_sign = np.where(close_or_open_sign == 0, 1, close_or_open_sign)
        # 陽線
        for i in range(len(close_or_open_sign)):
            sig = close_or_open_sign[i]
            l = low[i]
            o = open_[i]
            c = close[i]

            if sig == 1:
                downhige[i] = (c - l) / c
            else:
                downhige[i] = (o - l) / c
        downhige = downhige/close
        return downhige

    def _get_every_15min_datas(df: pd.DataFrame) -> pd.DataFrame:
        """15分ごとのにデータを整形する"""

        def parse_to_minute(timestamp):
            return timestamp.minute

        df = df.copy()
        df['temp'] = np.vectorize(parse_to_minute)(df['timestamp'])
        return df[df['temp'] % 15 == 0].drop(columns=['temp'])