import pandas as pd
import numpy as np
import ta
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
from collections import deque

from utils import log_and_print

# Global variable to track drift history
drift_window = deque(maxlen=10)

def identify_market_structure(data):
    """Identifies swing highs/lows and market structure (HH, HL, LH, LL)."""
    data['swing_high'] = (data['high'].shift(1) < data['high']) & (data['high'].shift(-1) < data['high'])
    data['swing_low'] = (data['low'].shift(1) > data['low']) & (data['low'].shift(-1) > data['low'])

    swing_highs = data[data['swing_high']]
    swing_lows = data[data['swing_low']]

    data['HH'] = (data['swing_high']) & (data['high'] > swing_highs['high'].shift(1))
    data['HL'] = (data['swing_low']) & (data['low'] > swing_lows['low'].shift(1))
    data['LH'] = (data['swing_high']) & (data['high'] < swing_highs['high'].shift(1))
    data['LL'] = (data['swing_low']) & (data['low'] < swing_lows['low'].shift(1))

    # Fill NaN values
    for col in ['swing_high', 'swing_low', 'HH', 'HL', 'LH', 'LL']:
        data[col] = data[col].fillna(False)
    return data

def identify_candlestick_patterns(data):
    """Identifies common candlestick patterns."""
    patterns = {
        'engulfing': talib.CDLENGULFING,
        'doji': talib.CDLDOJI,
        'hammer': talib.CDLHAMMER,
        'shooting_star': talib.CDLSHOOTINGSTAR
    }
    for name, pattern_func in patterns.items():
        data[name] = pattern_func(data['open'], data['high'], data['low'], data['close'])
    return data

def identify_trends(data):
    """Identifies the current trend based on market structure."""
    if 'HH' not in data.columns or 'HL' not in data.columns or 'LH' not in data.columns or 'LL' not in data.columns:
        log_and_print("Warning: Market structure columns (HH, HL, etc.) are missing. Cannot identify trends.", is_error=True)
        data['trend'] = 'sideways'
    else:
        data['trend'] = np.where(data['HH'] & data['HL'], 'uptrend',
                                 np.where(data['LH'] & data['LL'], 'downtrend', 'sideways'))
    return data

def detect_breakouts(data, window=20):
    """Detects breakouts from recent highs and lows."""
    data['breakout'] = np.logical_or(
        np.greater(data['close'], data['high'].rolling(window=window).max().shift(1)),
        np.less(data['close'], data['low'].rolling(window=window).min().shift(1))
    )
    return data

def add_indicators(data):
    """Adds a set of technical indicators to the dataframe."""
    # Add basic indicators
    data['RSI'] = ta.momentum.rsi(data['close'], window=14)
    data['MACD'] = ta.trend.macd(data['close'])
    data['MACD_signal'] = ta.trend.macd_signal(data['close'])
    data['Bollinger_high'] = ta.volatility.bollinger_hband(data['close'])
    data['Bollinger_low'] = ta.volatility.bollinger_lband(data['close'])
    data['MA20'] = data['close'].rolling(window=20).mean()
    data['MA50'] = data['close'].rolling(window=50).mean()
    data['MA200'] = data['close'].rolling(window=200).mean()
    data['volatility_atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
    
    # Add advanced indicators
    data['ichimoku_a'] = ta.trend.ichimoku_a(data['high'], data['low'])
    data['ichimoku_b'] = ta.trend.ichimoku_b(data['high'], data['low'])
    data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'])
    if 'tick_volume' in data.columns:
        data['vwap'] = ta.volume.volume_weighted_average_price(data['high'], data['low'], data['close'], data['tick_volume'])
    
    return data

def detect_market_regime(data):
    """Detects market regime (trending or ranging) based on ADX."""
    try:
        adx = data.get('adx', ta.trend.adx(data['high'], data['low'], data['close']))
        data['regime'] = np.where(adx > 25, 'trending', 'ranging')
    except Exception as e:
        log_and_print(f"Error in detect_market_regime: {e}", is_error=True)
        data['regime'] = 'ranging'  # Default to 'ranging'
    return data

def add_all_features(df):
    """A master function to add all features to the dataframe."""
    df = add_indicators(df)
    df = identify_market_structure(df)
    df = identify_candlestick_patterns(df)
    df = identify_trends(df)
    df = detect_breakouts(df)
    df = detect_market_regime(df)
    df = df.fillna(0)
    return df

def select_features(X, y, max_features=20):
    """Selects the best features using a RandomForestRegressor."""
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=max_features).fit(X, y)
    selected_features = X.columns[selector.get_support()]
    log_and_print(f"Selected Features: {list(selected_features)}")
    return selected_features

def encode_categorical(df):
    """Encodes categorical columns like 'trend' and 'regime'."""
    le = LabelEncoder()
    for col in ['trend', 'regime']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
    return df

def calculate_future_returns(df):
    """Calculates future returns for training the model."""
    df["future_returns"] = df["close"].pct_change().shift(-1)
    return df.dropna()

def ensure_same_columns(df1, df2):
    """Ensures two DataFrames have the same columns in the same order."""
    missing_in_df1 = set(df2.columns) - set(df1.columns)
    for c in missing_in_df1:
        df1[c] = 0
    missing_in_df2 = set(df1.columns) - set(df2.columns)
    for c in missing_in_df2:
        df2[c] = 0
    return df1[df2.columns], df2

def detect_feature_drift(current_df, reference_df):
    """Detects drift between current data and a reference dataset."""
    current_df, reference_df = ensure_same_columns(current_df.copy(), reference_df.copy())
    
    drift_features = []
    for column in current_df.columns:
        if column not in reference_df.columns:
            continue
        stat, p_value = ks_2samp(current_df[column].dropna(), reference_df[column].dropna())
        if p_value < 0.05:
            drift_features.append(column)

    drift_detected = 1 if drift_features else 0
    drift_window.append(drift_detected)
    
    drift_ratio = sum(drift_window) / len(drift_window) if drift_window else 0
    log_and_print(f"Rolling drift ratio: {drift_ratio:.2f}")

    if drift_ratio > 0.5:
        log_and_print(f"Significant feature drift detected in: {drift_features}")
        return drift_features
    else:
        return []