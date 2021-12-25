
DURATION_1M = '1'
# DURATION_3M = '3'
DURATION_5M = '5'
DURATION_15M = '15'
DURATION_30M = '30'
DURATION_1H = '60'
# DURATION_2H = '120'
DURATION_4H = '240'
# DURATION_6H = '360'
# DURATION_12H = '720'
DURATION_1DAY = 'D'
# DURATION_1WEEK = 'W'
# DURATION_1MANTH = 'M'

BUY = 'Buy'
SELL = 'Sell'
NONE = 'None'

LIMIT = 'Limit'
MARKET = 'Market'
STOP = 'Stop'

NUMBER_OF_OHLCS = 1000
UPDATE_INTERVAL = 20



BYBIT_FEAURES = sorted([
    'ADX',
    'ADXR',
    'APO',
    'AROON_aroondown',
    'AROON_aroonup',
    'AROONOSC',
    'CCI',
    'DX',
    'MACD_macd',
    'MACD_macdsignal',
    'MACD_macdhist',
    'MFI',
#     'MINUS_DI',
#     'MINUS_DM',
    'MOM',
#     'PLUS_DI',
#     'PLUS_DM',
    'RSI',
    'STOCH_slowk',
    'STOCH_slowd',
    'STOCHF_fastk',
#     'STOCHRSI_fastd',
    'ULTOSC',
    'WILLR',
#     'ADOSC',
#     'NATR',
    'HT_DCPERIOD',
    'HT_DCPHASE',
    'HT_PHASOR_inphase',
    'HT_PHASOR_quadrature',
    'HT_TRENDMODE',
    'BETA',
    'LINEARREG',
    'LINEARREG_ANGLE',
    'LINEARREG_INTERCEPT',
    'LINEARREG_SLOPE',
    'STDDEV',
    'BBANDS_upperband',
    'BBANDS_middleband',
    'BBANDS_lowerband',
    'DEMA',
    'EMA',
    'HT_TRENDLINE',
    'KAMA',
    'MA',
    'MIDPOINT',
    'T3',
    'TEMA',
    'TRIMA',
    'WMA',

    'pct',
    'pct_mean5',
    'pct_mean15',
    'pct_mean25',
    'pct_std5',
    'pct_std15',
    'pct_std25',
    'uphige_size',
    'downhige_size',

    'heikin_cl',
    'heikin_op',
    'heikin_cl_mean5',
    'heikin_op_mean5',
    'heikin_cl_mean15',
    'heikin_op_mean15',
    'heikin_cl_mean25',
    'heikin_op_mean25',
    'heikin_cl_std5',
    'heikin_op_std5',
    'heikin_cl_std15',
    'heikin_op_std15',
    'heikin_cl_std25',
    'heikin_op_std25',
])

BYBIT_FINE_FEAURES = sorted([
    'pct',
    'pct_mean5',
    'pct_std5',
    'uphige_size',
    'downhige_size',
    'heikin_cl',
    'heikin_op',
    'heikin_cl_mean5',
    'heikin_op_mean5',
    'heikin_cl_std5',
    'heikin_op_std5',
])