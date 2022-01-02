from datetime import datetime
from typing import List, Dict, Tuple, Any, Union
import os

import numpy as np
import pandas as pd
from nptyping import NDArray
import joblib
import talib

from constants import BYBIT_FEAURES
from logger import Logger

logger = Logger()

Model = Any


class MLJudgement:
    """MLによりトレードを行うべきか判断を下すクラス

    Attributes
    ----------
    features : List[str]
        MLに用いる特徴量
    regression_models : Dict[str, Dict[str, Model]]
        用いる回帰モデル key: 'buy' or 'sell'
    classification_models : Dict[str, Dict[str, Model]]
        用いる分類モデル key: 'buy' or 'sell'
    blending_weights : NDArray[float]
        Blendingの重みパラメータ

    Methods
    -------
    predict -> pd.DataFrame
        MLによる予測結果と、orderを出す価格を算出
    """
    def __init__(self) -> None:
        self.features : List[str] = BYBIT_FEAURES  # 使用する特徴量
        self.models_dir_path : str = os.path.join('trained_models', 'bybit_ceeling')
        self.regression_model_names : List[str] = ['gbdt']  # Blending重み最適化を行った順番
        self.regression_models : List[Model] = self._load_models()

    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """MLによる予測結果と、orderを出す価格を算出

        Parameters
        ----------
        df_features : pd.DataFrame
            特徴量をもつdf

        Returns
        -------
        pd.DataFrame
            MLによる予測結果と、orderを出す価格をもつdf
        """
        X = df_features[self.features]
        # 回帰モデルで予測
        model : Model = self.regression_models[0]
        y_pred = model.predict(X)
        
        df = df_features.copy()
        df['y_pred'] = y_pred
        return df

    def _load_models(self) -> List[Model]:
        regression_models : List[Model] = []
        for model_name in self.regression_model_names:
            regression_models.append(joblib.load(os.path.join(self.models_dir_path, f'{model_name}.xz')))

        return regression_models


class FeatureCreator:
    """ohlcvの情報から特徴量を算出するクラス
    
    Methods
    -------
    create_features -> pd.DataFrame
        特徴量を算出する
    """
    def create_features(self, df_btc_15m: pd.DataFrame) -> pd.DataFrame:
        # 特徴量を計算
        df = self._calc_features(df_ohlcvs=df_btc_15m).dropna()
        df = df.set_index('timestamp')

        df_features = df.dropna()
        logger.info('Created features')
        logger.info(df_features[['open', 'high', 'low', 'close']].tail(2))
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
        df['DEMA'] = talib.DEMA(close, timeperiod=20) - hilo
        df['EMA'] = talib.EMA(close, timeperiod=20) - hilo
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close) - hilo
        df['KAMA'] = talib.KAMA(close, timeperiod=20) - hilo
        df['MA'] = talib.MA(close, timeperiod=20, matype=0) - hilo
        df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
        df['SMA'] = talib.SMA(close, timeperiod=20) - hilo
        df['SMA_'] = talib.SMA(close, timeperiod=20)
        df['up_down'] = np.log(df['close'] / df['SMA_'])
        for i in range(1, 40):
            df[f'up_down_{i}'] = df['up_down'].shift(i)
        df['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
        df['TEMA'] = talib.TEMA(close, timeperiod=20) - hilo
        df['TRIMA'] = talib.TRIMA(close, timeperiod=20) - hilo
        df['WMA'] = talib.WMA(close, timeperiod=20) - hilo

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
        df['TRIX'] = talib.TRIX(close, timeperiod=20)
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
        df['CORREL'] = talib.CORREL(high, low, timeperiod=20)
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

    def _up_hige_size(self, df: pd.DataFrame) -> NDArray[float]:
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

    def _down_hige_size(self, df: pd.DataFrame) -> NDArray[float]:
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
